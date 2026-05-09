import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
import os
import time
from statistics import NormalDist
from typing import Any, Dict, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# ============================================================
# ReGEN-TAD / TSCOUT with Phase-I Purification
# ============================================================

# -----------------------------
# Positional encoding + transformer encoder
# -----------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle = pos * angle_rates
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.constant(pe, dtype=tf.float32)

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


# -----------------------------
# Backbone network (Transformer + BiLSTM)
# Two-pass forecast + reconstruction
# -----------------------------
class ReGENTAD_Backbone(Model):
    def __init__(
        self,
        past_len,
        horizon,
        n_features,
        d_model=128,
        num_heads=6,
        ff_dim=128,
        lstm_units=32,
        dropout=0.1,
        loss_w_y1=0.2,
        loss_w_y2=0.8,
        loss_w_recon=0.5,
        latent_l2=0.0,
    ):
        super().__init__()
        self.past_len = int(past_len)
        self.horizon = int(horizon)
        self.n_features = int(n_features)

        self.loss_w_y1 = float(loss_w_y1)
        self.loss_w_y2 = float(loss_w_y2)
        self.loss_w_recon = float(loss_w_recon)
        self.latent_l2 = float(latent_l2)

        self.ln_in = layers.LayerNormalization(epsilon=1e-6)
        self.conv1 = layers.Conv1D(64, 3, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(64, 3, padding="same", activation="relu")
        self.to_d = layers.Dense(d_model)

        self.pe = PositionalEncoding(self.past_len, d_model)
        self.trans = TransformerEncoder(d_model, num_heads, ff_dim, dropout=dropout)
        self.pool_t = layers.GlobalAveragePooling1D()

        self.lstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
        )
        self.pool_l = layers.GlobalAveragePooling1D()

        self.concat = layers.Concatenate()
        self.dense_z = layers.Dense(ff_dim, activation="relu")
        self.z_drop = layers.Dropout(dropout)

        self.head_y1 = layers.Dense(self.horizon * self.n_features)
        self.head_recon = layers.Dense(self.past_len * self.n_features)

        self.flat_res = layers.Flatten()
        self.dense_refine = layers.Dense(ff_dim, activation="relu")
        self.head_y2 = layers.Dense(self.horizon * self.n_features)

        self.reshape_y = layers.Reshape((self.horizon, self.n_features))
        self.reshape_x = layers.Reshape((self.past_len, self.n_features))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def set_loss_weights(self, w_y1, w_y2, w_recon, latent_l2=None):
        self.loss_w_y1 = float(w_y1)
        self.loss_w_y2 = float(w_y2)
        self.loss_w_recon = float(w_recon)
        if latent_l2 is not None:
            self.latent_l2 = float(latent_l2)

    def encode(self, x, training=False):
        x = self.ln_in(x)
        x = self.conv1(x)
        x = self.conv2(x)
        xd = self.to_d(x)

        xt = self.pe(xd)
        xt = self.trans(xt, training=training)
        ht = self.pool_t(xt)

        xl = self.lstm(xd, training=training)
        hl = self.pool_l(xl)

        z = self.concat([ht, hl])
        z = self.dense_z(z)
        z = self.z_drop(z, training=training)

        if self.latent_l2 > 0:
            self.add_loss(self.latent_l2 * tf.reduce_mean(tf.square(z)))

        return z

    def call(self, x, training=False):
        z = self.encode(x, training=training)
        y1 = self.reshape_y(self.head_y1(z))
        rec = self.reshape_x(self.head_recon(z))
        return y1, rec, z

    def refine(self, z, residual):
        r = self.flat_res(residual)
        h = self.dense_refine(tf.concat([z, r], axis=-1))
        return self.reshape_y(self.head_y2(h))

    def train_step(self, data):
        x, targets = data
        y_true = targets["target"]
        x_true = targets["recon"]

        with tf.GradientTape() as tape:
            y1, rec, z = self(x, training=True)
            y2 = self.refine(z, y_true - y1)

            l1 = tf.reduce_mean(tf.square(y_true - y1))
            l2 = tf.reduce_mean(tf.square(y_true - y2))
            lr = tf.reduce_mean(tf.square(x_true - rec))

            loss = self.loss_w_y1 * l1 + self.loss_w_y2 * l2 + self.loss_w_recon * lr
            if self.losses:
                loss += tf.add_n(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# ============================================================
# Main Detector Class
# ============================================================
class ReGENTAD:
    """
    ReGEN-AD / TSCOUT with Phase-I purification (VSCOUT-style)
    + Option A (recommended for SMD): rank/top-frac prediction.
    """

    def __init__(
        self,
        past_len,
        horizon,
        n_features,

        # backbone architecture
        d_model=128,
        num_heads=6,
        ff_dim=128,
        lstm_units=32,
        dropout=0.1,

        # optimizer
        lr=1e-3,

        # loss weights (Stage B defaults)
        loss_w_y1=0.2,
        loss_w_y2=0.8,
        loss_w_recon=0.5,
        latent_l2=0.0,

        # ensemble parts + smoothing
        alpha=0.05,
        n_neighbors=20,
        dyn_lag=5,
        smooth_span=5,

        # scoring weights 
        weights=None,
        ndt_mode="static",         # "static" | "adaptive" | "hybrid"
        window_size=200,
        lag=10,
        sensitivity=1.25,

        # persistence filter (for threshold modes)
        min_duration=1,

        # Option A: rank/top-frac prediction (SMD recommended)
        #   - If decision_rule="rank", predict flags the top `rank_top_frac` of windows as anomalies.
        #   - If rank_top_frac="auto": uses alpha (i.e., alpha=0.05 -> top 5%).
        #   - If rank_use_smoothed=True: uses smoothed scores for ranking (default).
        rank_top_frac=0.052,
        rank_min_duration=1,
        rank_dilate=0,

        # regime latch 
        flag_persistent_regime=False,
        regime_quantile=0.995,
        regime_confirm_len=5,
        regime_persist_max=50,

        random_state=42,
        decision_rule="rank",
    ):
        self.past_len = int(past_len)
        self.horizon = int(horizon)
        self.n_features = int(n_features)

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.lstm_units = int(lstm_units)
        self.dropout = float(dropout)

        self.lr = float(lr)

        self.loss_w_y1 = float(loss_w_y1)
        self.loss_w_y2 = float(loss_w_y2)
        self.loss_w_recon = float(loss_w_recon)
        self.latent_l2 = float(latent_l2)

        self.alpha = float(alpha)
        self.n_neighbors = int(n_neighbors)
        self.dyn_lag = int(dyn_lag)
        self.smooth_span = int(smooth_span)

        self.ndt_mode = str(ndt_mode)
        self.window_size = int(window_size)
        self.lag = int(lag)
        self.sensitivity = float(sensitivity)
        self.min_duration = int(min_duration)

        self.rank_top_frac = rank_top_frac  # "auto" or float in (0,1)
        self.rank_min_duration = int(rank_min_duration)
        self.rank_dilate = int(rank_dilate)

        self.flag_persistent_regime = flag_persistent_regime
        self.regime_quantile = float(regime_quantile)
        self.regime_confirm_len = int(regime_confirm_len)
        self.regime_persist_max = None if regime_persist_max is None else int(regime_persist_max)

        self.random_state = int(random_state)
        self.decision_rule = str(decision_rule).lower()

        if self.decision_rule not in {"rank", "adaptive_threshold", "adaptive_quantile_threshold"}:
            raise ValueError(
                "decision_rule must be one of: 'rank', 'adaptive_threshold', "
                "'adaptive_quantile_threshold'."
            )

        if weights is None:
            weights = {
                "err": 0.6,
                "recon": 1.2,
                "knn": 0.2,
                "dyn": 0.2,
                "regime": 0.7,
                "vol": 0.6,
            }
        self.weights = dict(weights)

        # learned
        self.network = None
        self.nn = None
        self.stats = {}
        self.thresh_static = None
        self.z_threshold = None
        self._z_mu = None
        self._z_inv_cov = None

    # -----------------------------
    # Build / reset model
    # -----------------------------
    def _build_network(self):
        net = ReGENTAD_Backbone(
            past_len=self.past_len,
            horizon=self.horizon,
            n_features=self.n_features,
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            loss_w_y1=self.loss_w_y1,
            loss_w_y2=self.loss_w_y2,
            loss_w_recon=self.loss_w_recon,
            latent_l2=self.latent_l2,
        )
        net.compile(optimizer=optimizers.Adam(self.lr))
        return net

    @staticmethod
    def _ewma(scores, span):
        if span is None or span <= 1:
            return np.asarray(scores)
        return pd.Series(scores).ewm(span=int(span), adjust=False).mean().values

    @staticmethod
    def _run_length_filter(mask, min_len):
        mask = np.asarray(mask).astype(bool)
        if min_len <= 1:
            return mask.astype(int)
        out = np.zeros_like(mask, dtype=int)
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                if (j - i) >= min_len:
                    out[i:j] = 1
                i = j
            else:
                i += 1
        return out

    @staticmethod
    def _dilate_mask(mask, k):
        mask = np.asarray(mask).astype(bool)
        if k <= 0 or mask.size == 0:
            return mask
        out = mask.copy()
        idx = np.where(mask)[0]
        for i in idx:
            lo = max(0, i - k)
            hi = min(len(mask), i + k + 1)
            out[lo:hi] = True
        return out

    def _latent_residual_dynamics(self, Z):
        out = np.zeros(len(Z), dtype=float)
        L = int(self.dyn_lag)
        if len(Z) <= L:
            return out
        for t in range(L, len(Z)):
            out[t] = np.sum((Z[t] - Z[t - L : t].mean(axis=0)) ** 2)
        out[:L] = out[L]
        return out

    def _regime_score(self, Z):
        if self._z_mu is None:
            return np.zeros(len(Z), dtype=float)
        delta = Z - self._z_mu
        return np.sqrt(np.einsum("nj,jk,nk->n", delta, self._z_inv_cov, delta))

    @staticmethod
    def _iqr_fit(v):
        q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        iqr = max(float(q75 - q25), 1e-6)
        return float(q50), float(iqr)

    @staticmethod
    def _iqr_norm(v, med, iqr):
        iqr = max(float(iqr), 1e-6)
        return np.abs((v - med) / iqr)

    def make_windows(self, series, point_labels=None):
        series = np.asarray(series, dtype=np.float32)
        if series.ndim == 1:
            series = series[:, None]
        if series.ndim != 2:
            raise ValueError("series must have shape (T, n_features) or (T,).")
        if series.shape[1] != self.n_features:
            raise ValueError(
                f"series has n_features={series.shape[1]}, expected {self.n_features}."
            )
        if len(series) < (self.past_len + self.horizon):
            raise ValueError(
                "series is too short for the configured past_len and horizon."
            )

        if point_labels is not None:
            point_labels = np.asarray(point_labels)
            if point_labels.ndim != 1:
                raise ValueError("point_labels must be 1D.")
            if len(point_labels) != len(series):
                raise ValueError("point_labels must have the same length as series.")

        X, Y = [], []
        window_labels = []
        end_indices = []

        for t in range(self.past_len, len(series) - self.horizon + 1):
            X.append(series[t - self.past_len : t])
            Y.append(series[t : t + self.horizon])
            if point_labels is not None:
                window_labels.append(int(np.max(point_labels[t : t + self.horizon]) > 0))
                end_indices.append(t + self.horizon - 1)

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        if point_labels is None:
            return X, Y

        return (
            X,
            Y,
            np.asarray(window_labels, dtype=int),
            np.asarray(end_indices, dtype=int),
        )

    def _coerce_window_data(self, data, point_labels=None):
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError("Windowed data must be provided as (X, Y).")
            X = np.asarray(data[0], dtype=np.float32)
            Y = np.asarray(data[1], dtype=np.float32)
            if X.ndim != 3 or Y.ndim != 3:
                raise ValueError("Windowed X and Y must both be 3D.")
            if X.shape[0] != Y.shape[0]:
                raise ValueError("Windowed X and Y must have the same number of samples.")
            return X, Y, None, None

        windows = self.make_windows(data, point_labels=point_labels)
        if point_labels is None:
            X, Y = windows
            return X, Y, None, None
        return windows

    # -----------------------------
    # Phase I Purification
    # -----------------------------
    def _purify_indices(self, Xtr, Ytr, q=0.97, max_remove=0.30, epochs=20, batch_size=64, verbose=0):
        tf.keras.backend.clear_session()
        net = self._build_network()

        # Stage A: recon-only
        net.set_loss_weights(w_y1=0.0, w_y2=0.0, w_recon=1.0, latent_l2=self.latent_l2)

        net.fit(
            Xtr,
            {"target": Ytr, "recon": Xtr},
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
            shuffle=True,
        )

        _y1, Xrec, _Z = net.predict(Xtr, batch_size=256, verbose=0)
        recon_err = np.mean((Xtr - Xrec) ** 2, axis=(1, 2))

        thr = float(np.quantile(recon_err, q))
        cand = recon_err >= thr

        if max_remove is not None:
            max_k = int(np.floor(max_remove * len(Xtr)))
            if max_k < cand.sum():
                idx = np.argsort(recon_err)[::-1][:max_k]
                cand = np.zeros_like(cand, dtype=bool)
                cand[idx] = True

        keep_idx = np.where(~cand)[0]
        return keep_idx

    # -----------------------------
    # fit
    # -----------------------------
    def fit(
        self,
        X,
        Y,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0,
        purify=True,
        purify_q=0.97,
        purify_max_remove=0.30,
        purify_epochs=20,
        purify_iters=1,
    ):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        n = len(X)
        split = int(n * (1 - float(validation_split)))
        Xtr, Xcal = X[:split], X[split:]
        Ytr, Ycal = Y[:split], Y[split:]

        # ---- Stage I: purification on TRAIN ONLY ----
        if purify:
            keep = np.arange(len(Xtr))
            for _ in range(int(purify_iters)):
                if len(keep) < max(50, self.n_neighbors + 5):
                    break
                keep2 = self._purify_indices(
                    Xtr[keep],
                    Ytr[keep],
                    q=float(purify_q),
                    max_remove=float(purify_max_remove) if purify_max_remove is not None else None,
                    epochs=int(purify_epochs),
                    batch_size=max(32, int(batch_size)),
                    verbose=verbose,
                )
                keep = keep[keep2]
            Xtr2, Ytr2 = Xtr[keep], Ytr[keep]
        else:
            Xtr2, Ytr2 = Xtr, Ytr

        # ---- Stage II: refit full model on cleaned training set ----
        tf.keras.backend.clear_session()
        self.network = self._build_network()

        self.network.set_loss_weights(
            w_y1=self.loss_w_y1,
            w_y2=self.loss_w_y2,
            w_recon=self.loss_w_recon,
            latent_l2=self.latent_l2,
        )

        self.network.fit(
            Xtr2,
            {"target": Ytr2, "recon": Xtr2},
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=verbose,
            shuffle=True,
        )

        y1_c, xr_c, Zcal = self.network.predict(Xcal, batch_size=256, verbose=0)
        y2_c = self.network.refine(Zcal, Ycal - y1_c).numpy()

        self._z_mu = Zcal.mean(axis=0)

        try:
            # Original behavior (unchanged)
            cov = np.cov(Zcal.T) + 1e-5 * np.eye(Zcal.shape[1])
            self._z_inv_cov = np.linalg.pinv(cov)

        except Exception:
            try:
                # Fallback 1: LedoitÃ¢â‚¬â€œWolf shrinkage
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(Zcal)
                self._z_inv_cov = np.linalg.inv(lw.covariance_)

            except Exception:
                # Fallback 2: diagonal covariance
                var = Zcal.var(axis=0) + 1e-6
                self._z_inv_cov = np.diag(1.0 / var)

        Z_knn = Zcal.copy()
        finite_mask = np.isfinite(Z_knn).all(axis=1)

        if finite_mask.sum() < max(5, self.n_neighbors):
            self.nn = None
        else:
            Z_knn = Z_knn[finite_mask]
            self.nn = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, len(Z_knn))
            ).fit(Z_knn)


        resid = (Ycal - y2_c)

        if self.nn is not None:
            knn_score = np.log1p(self.nn.kneighbors(Zcal)[0].mean(axis=1))
        else:
            knn_score = np.zeros(len(Zcal))

        raw = {
            "err": np.log1p(np.mean((Ycal - y2_c) ** 2, axis=(1, 2))),
            "recon": np.log1p(np.mean((Xcal - xr_c) ** 2, axis=(1, 2))),
            "knn": knn_score,
            "dyn": np.log1p(self._latent_residual_dynamics(Zcal)),
            "regime": np.log1p(self._regime_score(Zcal)),
            "vol": np.log1p(np.std(resid, axis=(1, 2))),
        }


        self.stats = {}
        parts = {}
        for k, v in raw.items():
            med, iqr = self._iqr_fit(v)
            self.stats[k] = {"med": med, "iqr": iqr}
            parts[k] = self._iqr_norm(v, med, iqr) * self.weights.get(k, 1.0)

        scores = np.mean(list(parts.values()), axis=0)
        scores = self._ewma(scores, self.smooth_span)

        self._calibrate_static_threshold(scores)
        self._baseline_mu = Xtr2.mean(axis=(0, 1))
        self._baseline_std = Xtr2.std(axis=(0, 1)) + 1e-8
        return self

    # -----------------------------
    # decision function
    # -----------------------------
    def decision_function(self, X, Y, batch_size=256, return_parts=False):
        if self.network is None:
            raise RuntimeError("Call fit() before decision_function().")


        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        y1, xr, Z = self.network.predict(X, batch_size=int(batch_size), verbose=0)
        y2 = self.network.refine(Z, Y - y1).numpy()
        resid = (Y - y2)

        if self.nn is not None:
            knn_score = np.log1p(
                self.nn.kneighbors(Z)[0].mean(axis=1)
            )
        else:
            knn_score = np.zeros(len(Z))

        raw = {
            "err": np.log1p(np.mean((Y - y2) ** 2, axis=(1, 2))),
            "recon": np.log1p(np.mean((X - xr) ** 2, axis=(1, 2))),
            "knn": knn_score,
            "dyn": np.log1p(self._latent_residual_dynamics(Z)),
            "regime": np.log1p(self._regime_score(Z)),
            "vol": np.log1p(np.std(resid, axis=(1, 2))),
        }


        parts = {}
        for k, v in raw.items():
            med = self.stats[k]["med"]
            iqr = self.stats[k]["iqr"]
            parts[k] = self._iqr_norm(v, med, iqr) * self.weights.get(k, 1.0)

        scores = np.mean(list(parts.values()), axis=0)
        scores = self._ewma(scores, self.smooth_span)

        return (scores, parts) if return_parts else scores


    def _calibrate_static_threshold(self, scores):
        """
        Conservative static threshold from clean calibration scores.

        Keeps alpha unchanged, but makes the empirical threshold stricter by:
        1) using a finite-sample 'higher' quantile
        2) comparing it against a robust MAD-based floor
        3) taking the maximum of the two
        """
        scores = np.asarray(scores, dtype=float)
        n = len(scores)

        if n == 0:
            raise ValueError("Cannot calibrate threshold from empty score array.")

        # Finite-sample conservative quantile at level 1 - alpha
        # Equivalent to a slightly more conservative empirical quantile choice
        q = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q = min(max(q, 0.0), 1.0)

        try:
            q_thr = float(np.quantile(scores, q, method="higher"))
        except TypeError:
            # older NumPy fallback
            q_thr = float(np.quantile(scores, q, interpolation="higher"))

        # Robust floor
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)) + 1e-8)
        mad_thr = float(med + 3.5 * mad)

        # Final conservative threshold
        self.thresh_static = max(q_thr, mad_thr)

        # Keep z-threshold for adaptive logic, but derive it from the final static threshold
        mu = float(np.mean(scores))
        sd = float(np.std(scores) + 1e-8)
        self.z_threshold = (self.thresh_static - mu) / sd



    # -----------------------------
    # threshold series builder
    # -----------------------------
    def _threshold_series(self, scores, adaptive=True, sensitivity=None):
        """
        Conservative thresholding:
        - uses the calibrated threshold directly (no division by sensitivity)
        - uses only past information
        - uses the stricter of:
            median + 1.4826 * z * MAD
            mean   + z * std
        - never lets rolling threshold fall below static threshold
        - avoids bfill/future borrowing
        """
        scores = np.asarray(scores, dtype=float)

        # Static calibrated threshold: use directly
        thr_static = np.full_like(scores, self.thresh_static, dtype=float)

        if (not adaptive) or (self.ndt_mode == "static"):
            return thr_static

        # Use only past information
        lag = max(int(self.lag), 1)
        s = pd.Series(scores).shift(lag)

        # Require more history before trusting adaptive threshold
        min_hist = max(30, self.window_size)

        # Robust rolling threshold
        roll_med = s.rolling(self.window_size, min_periods=min_hist).median()
        roll_mad = s.rolling(self.window_size, min_periods=min_hist).apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True,
        )

        # Classical rolling threshold
        roll_mean = s.rolling(self.window_size, min_periods=min_hist).mean()
        roll_std = s.rolling(self.window_size, min_periods=min_hist).std()

        zref = max(float(self.z_threshold), 1.0)

        thr_mad = (roll_med + 1.4826 * zref * roll_mad).to_numpy()
        thr_std = (roll_mean + zref * roll_std).to_numpy()

        # Use the stricter rolling threshold
        thr_roll = np.maximum(thr_mad, thr_std)

        # No future leakage via bfill; use static threshold until enough history exists
        thr_roll = np.where(np.isfinite(thr_roll), thr_roll, self.thresh_static)

        # Final conservative guard
        thr_roll = np.maximum(thr_roll, thr_static)

        if self.ndt_mode == "adaptive":
            return thr_roll
        if self.ndt_mode == "hybrid":
            return np.maximum(thr_static, thr_roll)

        raise ValueError("ndt_mode must be one of: 'static' | 'adaptive' | 'hybrid'")
    
    # -----------------------------
    # Option A: rank/top-frac prediction
    # -----------------------------
    def _rank_predict(self, scores):
        """
        Select top-k windows by score.
        k = ceil(top_frac * N), where top_frac comes from:
          - rank_top_frac="auto" -> alpha
          - else -> float(rank_top_frac)
        """
        N = len(scores)
        if N == 0:
            return np.zeros(0, dtype=int)

        if isinstance(self.rank_top_frac, str) and self.rank_top_frac.lower() == "auto":
            top_frac = float(self.alpha)
        else:
            top_frac = float(self.rank_top_frac)

        top_frac = float(np.clip(top_frac, 1e-6, 0.999999))
        k = int(np.ceil(top_frac * N))
        k = max(1, min(N, k))

        idx = np.argsort(scores)[-k:]  # top-k
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True

        # Optional: expand candidates slightly to catch short bursts
        mask = self._dilate_mask(mask, self.rank_dilate)

        # Optional: require persistence (rank_min_duration)
        pred = self._run_length_filter(mask, self.rank_min_duration)
        return pred

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ValueError("scores must be non-empty.")
        if not np.isfinite(scores).all():
            raise ValueError("scores must contain only finite values.")
        return scores

    @staticmethod
    def _tail_z_value(alpha: float) -> float:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        return float(NormalDist().inv_cdf(1.0 - float(alpha)))

    @staticmethod
    def _postprocess_mask(mask: np.ndarray, min_duration: int, dilate: int) -> np.ndarray:
        pred = ReGENTAD._run_length_filter(mask, min_len=min_duration)
        pred = ReGENTAD._dilate_mask(pred, k=dilate)
        return np.asarray(pred, dtype=int)

    def _predict_rank_from_scores(
        self,
        scores: np.ndarray,
        *,
        alpha: Optional[float] = None,
        top_frac: Any = None,
        min_duration: Optional[int] = None,
        dilate: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        scores = self._validate_scores(scores)
        alpha = float(self.alpha if alpha is None else alpha)
        top_frac = self.rank_top_frac if top_frac is None else top_frac
        min_duration = self.rank_min_duration if min_duration is None else int(min_duration)
        dilate = self.rank_dilate if dilate is None else int(dilate)

        if isinstance(top_frac, str):
            if top_frac.lower() != "auto":
                raise ValueError("top_frac must be 'auto' or a float in (0, 1).")
            top_frac_used = alpha
        else:
            top_frac_used = float(top_frac)

        top_frac_used = float(np.clip(top_frac_used, 1e-6, 0.999999))
        n = scores.size
        k = max(1, min(n, int(math.ceil(top_frac_used * n))))

        idx = np.argsort(scores)[-k:]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        pred = self._postprocess_mask(mask, min_duration=min_duration, dilate=dilate)

        meta = {
            "threshold_used": float(np.min(scores[idx])) if idx.size else float("nan"),
            "top_frac_used": float(top_frac_used),
            "k_selected": float(k),
        }
        return pred, meta

    def _predict_adaptive_threshold_from_scores(
        self,
        scores: np.ndarray,
        *,
        alpha: Optional[float] = None,
        min_duration: Optional[int] = None,
        dilate: int = 0,
        window: Optional[int] = None,
        robust: bool = True,
        min_history: int = 5,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        scores = self._validate_scores(scores)
        alpha = float(self.alpha if alpha is None else alpha)
        min_duration = self.min_duration if min_duration is None else int(min_duration)
        dilate = int(dilate)
        window = int(self.window_size if window is None else window)
        if window < 2:
            raise ValueError("window must be >= 2.")
        if min_history < 1:
            raise ValueError("min_history must be >= 1.")

        z_value = self._tail_z_value(alpha)
        thresholds = np.full(scores.shape[0], np.inf, dtype=float)

        for t in range(scores.shape[0]):
            hist = scores[max(0, t - window) : t]
            if hist.size < min_history:
                continue

            if robust:
                q25, q50, q75 = np.quantile(hist, [0.25, 0.5, 0.75])
                sigma = max((q75 - q25) / 1.349, 1e-8)
                thresholds[t] = float(q50 + z_value * sigma)
            else:
                mu = float(np.mean(hist))
                sigma = max(float(np.std(hist, ddof=0)), 1e-8)
                thresholds[t] = float(mu + z_value * sigma)

        mask = scores >= thresholds
        pred = self._postprocess_mask(mask, min_duration=min_duration, dilate=dilate)

        finite_thresholds = thresholds[np.isfinite(thresholds)]
        meta = {
            "threshold_used": float(np.median(finite_thresholds)) if finite_thresholds.size else float("nan"),
            "top_frac_used": float("nan"),
            "threshold_series": thresholds,
            "z_value": z_value,
        }
        return pred, meta

    def _predict_adaptive_quantile_threshold_from_scores(
        self,
        scores: np.ndarray,
        *,
        alpha: Optional[float] = None,
        min_duration: Optional[int] = None,
        dilate: int = 0,
        window: Optional[int] = None,
        min_history: int = 25,
        quantile_buffer: float = 0.005,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        scores = self._validate_scores(scores)
        alpha = float(self.alpha if alpha is None else alpha)
        min_duration = self.min_duration if min_duration is None else int(min_duration)
        dilate = int(dilate)
        window = int(self.window_size if window is None else window)

        if window < 2:
            raise ValueError("window must be >= 2.")
        if min_history < 1:
            raise ValueError("min_history must be >= 1.")

        thresholds = np.full(scores.shape[0], np.inf, dtype=float)

        q = min(0.999, 1.0 - alpha + quantile_buffer)

        for t in range(scores.shape[0]):
            hist = scores[max(0, t - window):t]
            if hist.size < min_history:
                continue

            thresholds[t] = float(np.quantile(hist, q))

        mask = scores >= thresholds
        pred = self._postprocess_mask(mask, min_duration=min_duration, dilate=dilate)

        finite_thresholds = thresholds[np.isfinite(thresholds)]
        meta = {
            "threshold_used": float(np.median(finite_thresholds)) if finite_thresholds.size else float("nan"),
            "top_frac_used": float("nan"),
            "threshold_series": thresholds,
            "quantile_used": q,
        }
        return pred, meta

    def _predict_from_scores(
        self,
        scores: np.ndarray,
        *,
        decision_rule: str = "adaptive_threshold",
        alpha: Optional[float] = None,
        min_duration: Optional[int] = None,
        dilate: Optional[int] = None,
        window: Optional[int] = None,
        robust: bool = True,
        min_history: int = 5,
        quantile_buffer: float = 0.005,
        return_metadata: bool = False,
    ):
        decision_rule = str(decision_rule).lower()

        if decision_rule == "adaptive_threshold":
            pred, meta = self._predict_adaptive_threshold_from_scores(
                scores,
                alpha=alpha,
                min_duration=min_duration,
                dilate=0 if dilate is None else dilate,
                window=window,
                robust=robust,
                min_history=min_history,
            )
        elif decision_rule == "adaptive_quantile_threshold":
            pred, meta = self._predict_adaptive_quantile_threshold_from_scores(
                scores,
                alpha=alpha,
                min_duration=min_duration,
                dilate=0 if dilate is None else dilate,
                window=window,
                min_history=min_history,
                quantile_buffer=quantile_buffer,
            )
        else:
            raise ValueError(
                "decision_rule must be one of: 'adaptive_threshold', "
                "'adaptive_quantile_threshold'."
            )

        return (pred, meta) if return_metadata else pred

    # -----------------------------
    # predict
    # -----------------------------
    def predict(
        self,
        X,
        Y,
        alpha: Optional[float] = None,
        min_duration: Optional[int] = None,
        dilate: Optional[int] = None,
        window: Optional[int] = None,
        robust: bool = True,
        min_history: int = 5,
        quantile_buffer: float = 0.005,
        return_scores=False,
        return_parts=False,
        return_metadata: bool = False,
    ):
        scores, parts = self.decision_function(X, Y, return_parts=True)

        active_rule = self.decision_rule

        if active_rule == "rank":
            pred = self._rank_predict(scores)

        else:
            pred, meta = self._predict_from_scores(
                scores,
                decision_rule=active_rule,
                alpha=alpha,
                min_duration=min_duration,
                dilate=dilate,
                window=window,
                robust=robust,
                min_history=min_history,
                quantile_buffer=quantile_buffer,
                return_metadata=True,
            )

            outputs = [pred]
            if return_scores:
                outputs.append(scores)
            if return_parts:
                outputs.append(parts)
            if return_metadata:
                outputs.append(meta)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        if self.flag_persistent_regime:
            r = parts["regime"]
            rthr = float(np.quantile(r, self.regime_quantile))
            rmask = r > rthr

            start = None
            cnt = 0
            for i in range(len(rmask)):
                if rmask[i]:
                    cnt += 1
                    if cnt >= self.regime_confirm_len:
                        start = i - self.regime_confirm_len + 1
                        break
                else:
                    cnt = 0

            if start is not None:
                end = len(pred) if self.regime_persist_max is None else min(len(pred), start + self.regime_persist_max)
                pred[start:end] = 1

        outputs = [pred]
        if return_scores:
            outputs.append(scores)
        if return_parts:
            outputs.append(parts)
        if return_metadata:
            _, meta = self._predict_rank_from_scores(scores)
            outputs.append(meta)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def fit_predict(
        self,
        train_data,
        test_data=None,
        *,
        train_point_labels=None,
        test_point_labels=None,
        fit_kwargs=None,
        predict_kwargs=None,
        return_window_data: bool = False,
    ):
        fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)

        X_train, Y_train, train_window_labels, train_end_indices = self._coerce_window_data(
            train_data,
            point_labels=train_point_labels,
        )

        if test_data is None:
            X_test, Y_test = X_train, Y_train
            test_window_labels = train_window_labels
            test_end_indices = train_end_indices
        else:
            X_test, Y_test, test_window_labels, test_end_indices = self._coerce_window_data(
                test_data,
                point_labels=test_point_labels,
            )

        self.fit(X_train, Y_train, **fit_kwargs)
        pred_outputs = self.predict(X_test, Y_test, **predict_kwargs)

        if not return_window_data:
            return pred_outputs

        info = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_test": X_test,
            "Y_test": Y_test,
            "train_window_labels": train_window_labels,
            "train_end_indices": train_end_indices,
            "test_window_labels": test_window_labels,
            "test_end_indices": test_end_indices,
        }
        return pred_outputs, info
    

    def factor_contribution(
        self,
        x_window,
        ref_windows=None,
        scores=None,
        t_idx=None,
        top_k=None,
        normalize=True,
        plot=False,
        figsize=(12, 6),
        feature_names=None,
        indices=None,          # <<< NEW (optional)
    ):
        """
        Factor-level attribution for detected anomalies.
        """

        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt

        # -----------------------------
        # Input checks
        # -----------------------------
        x_window = np.asarray(x_window)
        if x_window.ndim != 2:
            raise ValueError("x_window must have shape (past_len, n_features).")

        n_features = x_window.shape[1]

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError("feature_names must have length equal to n_features.")
            labels = list(feature_names)
        else:
            labels = list(range(n_features))

        # --------------------------------------------------
        # 1. Baseline
        # --------------------------------------------------
        if ref_windows is not None:
            ref_windows = np.asarray(ref_windows)
            mu_ref = ref_windows.mean(axis=(0, 1))
            std_ref = ref_windows.std(axis=(0, 1))
        elif hasattr(self, "_baseline_mu") and hasattr(self, "_baseline_std"):
            mu_ref = np.asarray(self._baseline_mu)
            std_ref = np.asarray(self._baseline_std)
        else:
            raise RuntimeError("No baseline available for factor attribution.")

        std_ref = np.maximum(std_ref, 1e-8)

        # --------------------------------------------------
        # 2. Standardized deviation
        # --------------------------------------------------
        x_mean = x_window.mean(axis=0)
        delta = np.abs(x_mean - mu_ref) / std_ref

        # --------------------------------------------------
        # 3. Gradient-based sensitivity
        # --------------------------------------------------
        x_tensor = tf.convert_to_tensor(x_window[None, ...], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            _, x_recon, _ = self.network(x_tensor, training=False)
            recon_loss = tf.reduce_mean(tf.square(x_tensor - x_recon))

        grads = tape.gradient(recon_loss, x_tensor)

        if grads is None:
            sensitivity = np.ones(n_features)
        else:
            sensitivity = tf.reduce_mean(tf.abs(grads[0]), axis=0).numpy()

        # --------------------------------------------------
        # 4. Contribution
        # --------------------------------------------------
        contrib = delta * sensitivity
        contrib = np.where(np.isfinite(contrib), contrib, 0.0)

        if normalize:
            s = contrib.sum()
            if s > 0:
                contrib = contrib / (s + 1e-8)

        idx_sorted = np.argsort(contrib)[::-1]
        if top_k is not None:
            top_k = max(1, min(int(top_k), len(idx_sorted)))
            idx_sorted = idx_sorted[:top_k]

        # --------------------------------------------------
        # 5. Plot
        # --------------------------------------------------
        if plot:
            fig, axes = plt.subplots(
                2, 1,
                figsize=figsize,
                gridspec_kw={"height_ratios": [2, 1]}
            )

            # ----- Top: anomaly score -----
            if scores is not None:
                scores = np.asarray(scores, dtype=float)

                if indices is not None:
                    if len(indices) != len(scores):
                        raise ValueError("indices must have same length as scores.")
                    t = indices
                else:
                    t = np.arange(len(scores))

                finite = np.isfinite(scores)

                if finite.sum() > 0:
                    s_min, s_max = scores[finite].min(), scores[finite].max()
                    scores_plot = scores.copy()
                    if s_max > s_min:
                        scores_plot[finite] = (
                            scores_plot[finite] - s_min
                        ) / (s_max - s_min)

                    axes[0].plot(
                        t[finite],
                        scores_plot[finite],
                        color="black",
                        lw=1.5,
                        label="ReGEN-TAD score (scaled)"
                    )

                    if self.decision_rule == "rank":
                        preds = self._rank_predict(scores)
                        idx_anom = np.where(preds == 1)[0]
                        idx_anom = idx_anom[np.isfinite(scores[idx_anom])]
                        axes[0].scatter(
                            np.asarray(t)[idx_anom],
                            scores_plot[idx_anom],
                            color="red",
                            s=18,
                            alpha=0.85,
                            label="Predicted anomaly"
                        )

                    if t_idx is not None:
                        axes[0].axvline(
                            t[t_idx] if indices is not None else int(t_idx),
                            color="red",
                            linestyle="--",
                            lw=1.5,
                            label="Explained window"
                        )

                    axes[0].legend()
                    axes[0].grid(True, linestyle=":", alpha=0.6)

            # ----- Bottom: attribution -----
            contrib_plot = contrib[idx_sorted]
            label_plot = [labels[i] for i in idx_sorted]

            axes[1].bar(
                range(len(idx_sorted)),
                contrib_plot,
                color="tab:orange",
                alpha=0.85
            )
            axes[1].set_xticks(range(len(idx_sorted)))
            axes[1].set_xticklabels(label_plot, rotation=45, ha="right")
            axes[1].set_ylabel("Contribution")
            axes[1].set_xlabel("Factor")
            axes[1].set_title(f"Top-{len(idx_sorted)} factor contributions")
            axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

            ymax = contrib_plot.max() if contrib_plot.size else 1.0
            axes[1].set_ylim(0, max(1e-6, ymax * 1.2))

            plt.tight_layout()
            plt.show()

        return contrib
