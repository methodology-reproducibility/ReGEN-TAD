# ============================================================
# LSTM-NDT Baseline
# Forecasting + nonparametric dynamic thresholding
# (Malhotra et al.-style)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


class LSTM_NDT:
    """
    LSTM-NDT: LSTM forecaster with MAE training loss
    and MAD-based threshold from training data.
    """

    def __init__(
        self,
        past_len,
        horizon,
        n_features,
        alpha=0.05,
        hidden_dim=64,
        use_mae=True,
        k_mad=3.5
    ):
        self.past_len = past_len
        self.horizon = horizon
        self.n_features = n_features
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.k_mad = k_mad

        inp = layers.Input(shape=(past_len, n_features))
        x = layers.LSTM(hidden_dim, return_sequences=False)(inp)
        x = layers.Dense(hidden_dim, activation="relu")(x)

        out = layers.Dense(horizon * n_features)(x)
        out = layers.Reshape((horizon, n_features))(out)

        self.model = Model(inp, out)
        self.model.compile(
            optimizer="adam",
            loss="mae" if use_mae else "mse"
        )

        self.med = None
        self.mad = None
        self.threshold = None

    def fit(self, Xp, Yf, epochs=20, batch_size=32, verbose=0):
        self.model.fit(
            Xp, Yf,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True
        )

        Yhat = self.model.predict(Xp, verbose=0)
        err = np.mean((Yf - Yhat) ** 2, axis=(1, 2))

        self.med = np.median(err)
        self.mad = np.median(np.abs(err - self.med)) + 1e-8
        self.threshold = self.med + self.k_mad * self.mad

    def decision_function(self, Xp, Yf):
        Yhat = self.model.predict(Xp, verbose=0)
        err = np.mean((Yf - Yhat) ** 2, axis=(1, 2))
        return err

    def predict(self, Xp, Yf):
        scores = self.decision_function(Xp, Yf)
        return (scores > self.threshold).astype(int)