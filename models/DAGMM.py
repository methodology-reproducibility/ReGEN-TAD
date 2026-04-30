# ============================================================
# DAGMM Baseline
# Deep Autoencoding Gaussian Mixture Model
# (Zong et al., ICLR 2018)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


class DAGMM:
    """
    DAGMM: Deep Autoencoding Gaussian Mixture Model.

    Expected X shape: (N, window_len, n_features)
    """

    def __init__(
        self,
        past_len,
        n_features,
        latent_dim=10,
        hidden_dim=64,
        n_components=2,
        alpha=0.05
    ):
        self.past_len = past_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.alpha = alpha

        self.ae = None
        self.estimator = None
        self.model = None

        self.phi = None
        self.mu = None
        self.cov = None

        self.threshold = None

        self._build()

    def _build(self):
        # Autoencoder
        x_in = layers.Input(shape=(self.past_len, self.n_features))
        x = layers.Flatten()(x_in)
        h = layers.Dense(self.hidden_dim, activation="relu")(x)
        z = layers.Dense(self.latent_dim)(h)

        h_dec = layers.Dense(self.hidden_dim, activation="relu")(z)
        x_hat = layers.Dense(self.past_len * self.n_features)(h_dec)
        x_hat = layers.Reshape((self.past_len, self.n_features))(x_hat)

        self.ae = Model(x_in, [z, x_hat], name="autoencoder")

        # Estimation network
        z_in = layers.Input(shape=(self.latent_dim + 2,))

        h_est = layers.Dense(self.hidden_dim, activation="relu")(z_in)
        gamma = layers.Dense(self.n_components, activation="softmax")(h_est)

        self.estimator = Model(z_in, gamma, name="estimator")

        # Full DAGMM
        z_out, x_hat_out = self.ae(x_in)

        def compute_dagmm_features(args):
            x, x_rec, z_emb = args
            rec_err = tf.reduce_mean((x - x_rec) ** 2, axis=[1, 2])
            rec_err = tf.expand_dims(rec_err, axis=1)

            num = tf.reduce_sum(x * x_rec, axis=[1, 2])
            denom = tf.norm(x, axis=[1, 2]) * tf.norm(x_rec, axis=[1, 2]) + 1e-8
            cos_sim = tf.expand_dims(num / denom, axis=1)

            return tf.concat([z_emb, rec_err, cos_sim], axis=1)

        z_feat = layers.Lambda(compute_dagmm_features, output_shape=(self.latent_dim + 2,))([x_in, x_hat_out, z_out])

        gamma_out = self.estimator(z_feat)

        self.model = Model(x_in, gamma_out)
        self.model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss=self._dagmm_loss
        )

    def _dagmm_loss(self, y_true, gamma):
        return tf.reduce_mean(gamma)

    def fit(self, X, epochs=30, batch_size=128, verbose=0):
        dummy = np.zeros((len(X), self.n_components))

        self.model.fit(
            X,
            dummy,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=verbose
        )

        z, x_hat = self.ae.predict(X, batch_size=256, verbose=0)

        rec_err = np.mean((X - x_hat) ** 2, axis=(1, 2))
        cos_sim = np.sum(X * x_hat, axis=(1, 2)) / (
            np.linalg.norm(X, axis=(1, 2)) *
            np.linalg.norm(x_hat, axis=(1, 2)) + 1e-8
        )

        z_feat = np.concatenate(
            [z, rec_err[:, None], cos_sim[:, None]],
            axis=1
        )

        gamma = self.estimator.predict(z_feat, verbose=0)

        # GMM parameter estimation (paper Eq. 10-12)
        self.phi = gamma.mean(axis=0)

        self.mu = np.zeros((self.n_components, z_feat.shape[1]))
        self.cov = np.zeros((self.n_components, z_feat.shape[1]))

        for k in range(self.n_components):
            w = gamma[:, k][:, None]
            norm = np.sum(w) + 1e-8
            self.mu[k] = np.sum(w * z_feat, axis=0) / norm
            diff = z_feat - self.mu[k]
            self.cov[k] = np.sum(w * diff * diff, axis=0) / norm

        scores = self.decision_function(X)
        self.threshold = np.quantile(scores, 1 - self.alpha)

    def decision_function(self, X):
        """Energy-based anomaly score (paper Eq. 13)."""
        z, x_hat = self.ae.predict(X, batch_size=256, verbose=0)

        rec_err = np.mean((X - x_hat) ** 2, axis=(1, 2))
        cos_sim = np.sum(X * x_hat, axis=(1, 2)) / (
            np.linalg.norm(X, axis=(1, 2)) *
            np.linalg.norm(x_hat, axis=(1, 2)) + 1e-8
        )

        z_feat = np.concatenate(
            [z, rec_err[:, None], cos_sim[:, None]],
            axis=1
        )

        energy = np.zeros(len(X))

        for k in range(self.n_components):
            diff = z_feat - self.mu[k]
            inv_cov = 1.0 / (self.cov[k] + 1e-8)
            exp_term = np.sum(diff * diff * inv_cov, axis=1)
            energy += self.phi[k] * np.exp(-0.5 * exp_term)

        energy = -np.log(energy + 1e-12)
        return energy

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)