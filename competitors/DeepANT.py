import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

class DeepAnt:
    """
    DeepAnT: CNN-based predictor + anomaly score.

    Uses MAE loss, 2x (Conv1D -> MaxPool) with 32 filters,
    Euclidean distance scoring, and training-set normalization.

    Thresholding methods:
      - "ksigma": mean(val_scores) + k * std(val_scores)
      - "quantile": quantile(val_scores, 1-alpha)
      - "kde": KDE tail threshold
    """

    def __init__(
        self,
        past_len: int,
        horizon: int,
        n_features: int,
        alpha: float = 0.05,
        n_filters: int = 32,
        kernel_size: int = 3,
        pool_size: int = 2,
        lr: float = 1e-2,
        momentum: float = 0.9,
        threshold_method: str = "ksigma",
        k_sigma: float = 3.0,
        validation_split: float = 0.10,
        random_state: int = 0,
    ):
        self.past_len = int(past_len)
        self.horizon = int(horizon)
        self.n_features = int(n_features)
        self.alpha = float(alpha)

        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)

        self.lr = float(lr)
        self.momentum = float(momentum)

        self.threshold_method = str(threshold_method).lower()
        self.k_sigma = float(k_sigma)
        self.validation_split = float(validation_split)
        self.random_state = int(random_state)

        self.model = None
        self.mu_ = None
        self.sigma_ = None
        self.threshold_ = None

        self._build()

    def _build(self):
        tf.random.set_seed(self.random_state)

        inp = layers.Input(shape=(self.past_len, self.n_features))

        x = layers.Conv1D(self.n_filters, self.kernel_size, padding="same", activation="relu")(inp)
        x = layers.MaxPooling1D(pool_size=self.pool_size)(x)

        x = layers.Conv1D(self.n_filters, self.kernel_size, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=self.pool_size)(x)

        x = layers.Flatten()(x)
        out = layers.Dense(self.horizon * self.n_features)(x)
        out = layers.Reshape((self.horizon, self.n_features))(out)

        self.model = Model(inp, out)

        opt = optimizers.SGD(learning_rate=self.lr, momentum=self.momentum)
        self.model.compile(optimizer=opt, loss="mae")

    def _fit_normalizer(self, Xp, Yf):
        X_flat = Xp.reshape(-1, self.n_features)
        Y_flat = Yf.reshape(-1, self.n_features)
        Z = np.vstack([X_flat, Y_flat])

        self.mu_ = Z.mean(axis=0)
        self.sigma_ = Z.std(axis=0)
        self.sigma_[self.sigma_ < 1e-8] = 1e-8

    def _norm_X(self, Xp):
        return (Xp - self.mu_[None, None, :]) / self.sigma_[None, None, :]

    def _norm_Y(self, Yf):
        return (Yf - self.mu_[None, None, :]) / self.sigma_[None, None, :]

    def _euclidean_score(self, Y_true, Y_pred):
        diff = (Y_true - Y_pred).reshape(len(Y_true), -1)
        return np.sqrt(np.sum(diff * diff, axis=1))

    def _compute_threshold(self, val_scores):
        m = float(np.mean(val_scores))
        s = float(np.std(val_scores)) if float(np.std(val_scores)) > 1e-12 else 1e-12

        if self.threshold_method == "ksigma":
            return m + self.k_sigma * s

        if self.threshold_method == "quantile":
            return float(np.quantile(val_scores, 1.0 - self.alpha))

        if self.threshold_method == "kde":
            xs = np.linspace(val_scores.min(), val_scores.max(), 512)
            bw = 1.06 * s * (len(val_scores) ** (-1 / 5))  # Silverman
            bw = max(bw, 1e-3)

            diffs = (xs[:, None] - val_scores[None, :]) / bw
            pdf = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
            cdf = np.cumsum(pdf)
            cdf = cdf / (cdf[-1] + 1e-12)

            idx = np.searchsorted(cdf, 1.0 - self.alpha)
            idx = int(np.clip(idx, 0, len(xs) - 1))
            return float(xs[idx])

        raise ValueError(f"Unknown threshold_method: {self.threshold_method}")

    def fit(self, Xp, Yf, epochs=20, batch_size=32, verbose=0):
        Xp = np.asarray(Xp, dtype=np.float32)
        Yf = np.asarray(Yf, dtype=np.float32)

        if Xp.ndim != 3:
            raise ValueError("Xp must be (N, past_len, n_features)")
        if Yf.ndim != 3:
            raise ValueError("Yf must be (N, horizon, n_features)")

        self._fit_normalizer(Xp, Yf)
        Xn = self._norm_X(Xp)
        Yn = self._norm_Y(Yf)

        hist = self.model.fit(
            Xn, Yn,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=self.validation_split,
            shuffle=True
        )

        if "val_loss" in hist.history and self.validation_split > 0 and len(Xn) >= 20:
            n_val = max(1, int(np.floor(len(Xn) * self.validation_split)))
            X_val = Xn[-n_val:]
            Y_val = Yn[-n_val:]
        else:
            X_val = Xn
            Y_val = Yn

        Yhat_val = self.model.predict(X_val, verbose=0)
        val_scores = self._euclidean_score(Y_val, Yhat_val)

        self.threshold_ = self._compute_threshold(val_scores)
        return self

    def decision_function(self, Xp, Yf):
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Call fit() first.")

        Xp = np.asarray(Xp, dtype=np.float32)
        Yf = np.asarray(Yf, dtype=np.float32)

        Xn = self._norm_X(Xp)
        Yn = self._norm_Y(Yf)

        Yhat = self.model.predict(Xn, verbose=0)
        scores = self._euclidean_score(Yn, Yhat)
        return scores

    def predict(self, Xp, Yf):
        if self.threshold_ is None:
            raise RuntimeError("Call fit() first.")
        scores = self.decision_function(Xp, Yf)
        return (scores > self.threshold_).astype(int)