# ============================================================
# Alioghli and Okay (2025) Implementation
# (Enhanced Transformer-based Anomaly Detection)
# Focus: Positional Encoding & Attention-based Forecasting
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers


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


class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ff = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(dropout)
        self.do2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        att = self.attn(x, x)
        att = self.do1(att, training=training)
        out1 = self.ln1(x + att)
        ff = self.ff(out1)
        ff = self.do2(ff, training=training)
        return self.ln2(out1 + ff)


class AlioghliOkay2025:
    """
    Implementation based on Alioghli and Okay (2025).
    Features: Specialized positional encoding and Transformer-only 
    forecasting to identify contextual and structural anomalies.
    """
    def __init__(
        self, 
        past_len, 
        horizon, 
        n_features, 
        d_model=128, 
        num_heads=8, 
        ff_dim=256, 
        dropout=0.1, 
        alpha=0.05,
        k_sigma=3.0
    ):
        self.past_len = past_len
        self.horizon = horizon
        self.n_features = n_features
        self.alpha = alpha
        self.k_sigma = k_sigma

        # Build the Transformer-Only Forecasting Model
        inp = layers.Input(shape=(past_len, n_features))
        
        # 1. Feature Projection & Positional Encoding (Key 2025 Feature)
        x = layers.Dense(d_model)(inp)
        x = PositionalEncoding(past_len, d_model)(x)
        
        # 2. Multi-Head Self-Attention (Capture Global Context)
        x = TransformerEncoder(d_model, num_heads, ff_dim, dropout)(x)
        
        # 3. Global Temporal Summary
        x = layers.GlobalAveragePooling1D()(x)
        
        # 4. Forecasting Head
        out = layers.Dense(horizon * n_features)(x)
        out = layers.Reshape((horizon, n_features))(out)

        self.model = Model(inp, out)
        self.model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
        
        self.threshold = None
        self.mu_err = None
        self.std_err = None

    def fit(self, Xp, Yf, epochs=40, batch_size=32, verbose=0):
        # Training as a self-supervised forecaster
        self.model.fit(
            Xp, Yf, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose, 
            shuffle=True
        )

        # Calibration for Anomaly Detection
        Yhat = self.model.predict(Xp, verbose=0)
        err = np.mean((Yf - Yhat)**2, axis=(1, 2))
        
        # Calculate threshold using K-Sigma (standard baseline approach)
        self.mu_err = np.mean(err)
        self.std_err = np.std(err)
        self.threshold = self.mu_err + self.k_sigma * self.std_err
        return self

    def decision_function(self, Xp, Yf):
        Yhat = self.model.predict(Xp, verbose=0)
        # Anomaly score is the squared prediction error
        return np.mean((Yf - Yhat)**2, axis=(1, 2))

    def predict(self, Xp, Yf):
        scores = self.decision_function(Xp, Yf)
        return (scores > self.threshold).astype(int)
