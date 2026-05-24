# ============================================================
# TranAD Baseline
# (Tuli et al., 2022)
# ============================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle = pos * angle_rates

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe)

    def call(self, x):
        return x + self.pe[: tf.shape(x)[1]]


class TranAD:
    """
    TranAD: Transformer-based adversarial anomaly detection.

    Input : window X_t in R^{W x D}
    Output: prediction of x_{t+1}

    Stage 1: y1 = Dec1(Enc(X_t))
    Stage 2: y2 = Dec2(Enc(X_t), residual = x_{t+1} - y1)

    Loss: L = eps^n * MSE(x, y1) + (1 - eps^n) * MSE(x, y2)
    Score: MSE(x_{t+1}, y2)
    """

    def __init__(
        self,
        n_window,
        n_features,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        batch_size=32,
        anneal_epochs=80,
        rank_top_frac=0.05,
    ):
        self.n_window = n_window
        self.n_features = n_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.batch_size = batch_size
        self.anneal_epochs = anneal_epochs
        self.rank_top_frac = float(rank_top_frac)
        self.rank_score_cutoff = None

        self.encoder = None
        self.dec1 = None
        self.dec2 = None
        self.model = None

    def _build(self):
        inp = layers.Input(shape=(self.n_window, self.n_features))

        x = layers.Dense(self.d_model)(inp)
        x = PositionalEncoding(self.n_window, self.d_model)(x)

        att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model
        )(x, x)
        x = layers.LayerNormalization()(x + att)

        ff = layers.Dense(self.ff_dim, activation="relu")(x)
        ff = layers.Dense(self.d_model)(ff)
        enc = layers.LayerNormalization()(x + ff)

        z = layers.GlobalAveragePooling1D()(enc)
        self.encoder = Model(inp, z, name="encoder")

        z_in_d1 = layers.Input(shape=(self.d_model,))
        y1 = layers.Dense(self.ff_dim, activation="relu")(z_in_d1)
        y1 = layers.Dense(self.n_features)(y1)
        self.dec1 = Model(z_in_d1, y1, name="decoder1")

        z_in_d2 = layers.Input(shape=(self.d_model,))
        res_in = layers.Input(shape=(self.n_features,))

        z2 = layers.Concatenate()([z_in_d2, res_in])
        y2 = layers.Dense(self.ff_dim, activation="relu")(z2)
        y2 = layers.Dense(self.n_features)(y2)
        self.dec2 = Model([z_in_d2, res_in], y2, name="decoder2")

        class _TranAD(Model):
            def __init__(self, encoder, dec1, dec2, anneal_epochs):
                super().__init__()
                self.encoder = encoder
                self.dec1 = dec1
                self.dec2 = dec2
                self.anneal_epochs = anneal_epochs
                self.epoch_counter = tf.Variable(0.0)

            def train_step(self, data):
                x_win, x_next = data

                with tf.GradientTape() as tape:
                    z = self.encoder(x_win, training=True)
                    y_hat1 = self.dec1(z, training=True)

                    residual = x_next - y_hat1
                    y_hat2 = self.dec2([z, residual], training=True)

                    eps = tf.pow(0.95, self.epoch_counter)
                    l1 = tf.reduce_mean(tf.square(x_next - y_hat1))
                    l2 = tf.reduce_mean(tf.square(x_next - y_hat2))
                    loss = eps * l1 + (1 - eps) * l2

                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return {"loss": loss}

        self.model = _TranAD(self.encoder, self.dec1, self.dec2, self.anneal_epochs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    def fit(self, X_window, epochs=80, verbose=1):
        """
        X_window: (N, W+1, D) -- last timestep is the prediction target.
        """
        X_window = np.asarray(X_window, dtype=np.float32)
        X_in = X_window[:, :-1, :]
        X_next = X_window[:, -1, :]

        if self.model is None:
            self._build()

        dataset = tf.data.Dataset.from_tensor_slices((X_in, X_next))
        dataset = dataset.batch(self.batch_size)

        for e in range(epochs):
            self.model.epoch_counter.assign(float(e))
            self.model.fit(dataset, epochs=1, verbose=verbose)

        # Calibrate fixed rank cutoff on pre-shock training windows only.
        train_scores = self.decision_function(X_window, batch_size=256)
        frac = float(np.clip(self.rank_top_frac, 1e-6, 0.999999))
        self.rank_score_cutoff = float(np.quantile(train_scores, 1.0 - frac))

    def decision_function(self, X_window, batch_size=256):
        X_window = np.asarray(X_window, dtype=np.float32)
        X_in = X_window[:, :-1, :]
        X_next = X_window[:, -1, :]

        z = self.encoder.predict(X_in, batch_size=batch_size, verbose=0)
        y_hat1 = self.dec1.predict(z, batch_size=batch_size, verbose=0)
        residual = X_next - y_hat1
        y_hat2 = self.dec2.predict([z, residual], batch_size=batch_size, verbose=0)

        return np.mean((X_next - y_hat2) ** 2, axis=1)

    def predict(
        self,
        X_window,
        top_frac=0.05,
        batch_size=256,
        return_scores=False
    ):
        """
        Rank-based prediction.

        Parameters
        ----------
        X_window : array, shape (N, W+1, D)
        top_frac : float
            Fraction of windows to flag as anomalous (fallback only when no calibrated cutoff exists).
        return_scores : bool
            If True, return (pred, scores).
        """
        scores = self.decision_function(X_window, batch_size=batch_size)

        n = len(scores)
        if self.rank_score_cutoff is not None:
            pred = (scores >= float(self.rank_score_cutoff)).astype(int)
        else:
            k = max(1, int(np.floor(top_frac * n)))
            idx = np.argsort(scores)[-k:]
            pred = np.zeros(n, dtype=int)
            pred[idx] = 1

        if return_scores:
            return pred, scores

        return pred
