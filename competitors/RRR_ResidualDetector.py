import numpy as np
import pandas as pd


class RRR_ResidualDetector:
    """
    Reduced-Rank Regression (RRR) residual-based anomaly detector.

    Designed for your windowed protocol (Xp, Yf):
      - Fit B on TRAIN only
      - Calibrate threshold on held-out CALIBRATION windows (still pre-shock / clean region)
      - Score test windows using residual MSE

    This avoids the common "in-sample MAD threshold -> FPR ~ 1" failure mode.
    """

    def __init__(
        self,
        rank=None,                  # None -> sqrt rule
        alpha=0.05,                 # target FPR on clean calibration (quantile threshold)
        threshold_method="quantile", # "quantile" (recommended) or "mad"
        k_mad=3.5,                  # used only if threshold_method="mad"
        standardize=True,           # standardize X and Y using TRAIN stats
        ridge=1e-6                  # small ridge for numerical stability
    ):
        self.rank = rank
        self.alpha = float(alpha)
        self.threshold_method = str(threshold_method).lower()
        self.k_mad = float(k_mad)
        self.standardize = bool(standardize)
        self.ridge = float(ridge)

        # learned params
        self.B = None
        self.threshold = None

        # for MAD option
        self.med = None
        self.mad = None

        # standardization stats (train only)
        self.x_mu_ = None
        self.x_sd_ = None
        self.y_mu_ = None
        self.y_sd_ = None

    @staticmethod
    def _flatten_windows(Xp, Yf):
        X = Xp.reshape(len(Xp), -1)
        Y = Yf.reshape(len(Yf), -1)
        return X, Y

    def _fit_standardizer(self, X, Y):
        self.x_mu_ = X.mean(axis=0)
        self.x_sd_ = X.std(axis=0) + 1e-8
        self.y_mu_ = Y.mean(axis=0)
        self.y_sd_ = Y.std(axis=0) + 1e-8

    def _apply_standardizer(self, X, Y):
        Xs = (X - self.x_mu_) / self.x_sd_
        Ys = (Y - self.y_mu_) / self.y_sd_
        return Xs, Ys

    def _prepare(self, Xp, Yf):
        X, Y = self._flatten_windows(Xp, Yf)

        if self.standardize:
            if self.x_mu_ is None:
                raise RuntimeError("Standardizer not fit. Call fit() first.")
            X, Y = self._apply_standardizer(X, Y)

        # add intercept
        X = np.hstack([np.ones((len(X), 1)), X])
        return X, Y

    def _fit_B_rrr(self, X, Y):
        # ridge-stabilized OLS: B = (X'X + lambda*I)^-1 X'Y
        XtX = X.T @ X
        if self.ridge is not None and self.ridge > 0:
            XtX = XtX + self.ridge * np.eye(XtX.shape[0])
        XtY = X.T @ Y
        B_full = np.linalg.pinv(XtX) @ XtY

        # reduced-rank via SVD truncation
        U, S, Vt = np.linalg.svd(B_full, full_matrices=False)

        if self.rank is None:
            r = int(np.sqrt(min(B_full.shape)))
            r = max(1, min(r, len(S)))
        else:
            r = int(self.rank)
            r = max(1, min(r, len(S)))

        S_trunc = S.copy()
        S_trunc[r:] = 0.0
        B_rrr = U @ np.diag(S_trunc) @ Vt
        return B_rrr

    @staticmethod
    def _resid_score(Y, Yhat):
        # per-window residual MSE (one scalar per window)
        return np.mean((Y - Yhat) ** 2, axis=1)

    def fit(self, Xp, Yf, calib_frac=0.2):
        """
        Fit on early TRAIN windows, calibrate threshold on later CAL windows.
        Assumes the input passed here is pre-shock (your protocol uses Xp_tr_contam, Yf_tr_contam).
        """
        X_raw, Y_raw = self._flatten_windows(Xp, Yf)

        n = len(X_raw)
        if n < 20:
            raise ValueError("Not enough training windows for RRR calibration.")

        split = int(np.floor(n * (1.0 - float(calib_frac))))
        split = max(5, min(split, n - 5))

        Xtr, Ytr = X_raw[:split], Y_raw[:split]
        Xcal, Ycal = X_raw[split:], Y_raw[split:]

        # standardize using TRAIN ONLY
        if self.standardize:
            self._fit_standardizer(Xtr, Ytr)
            Xtr, Ytr = self._apply_standardizer(Xtr, Ytr)
            Xcal, Ycal = self._apply_standardizer(Xcal, Ycal)

        # add intercept
        Xtr_i = np.hstack([np.ones((len(Xtr), 1)), Xtr])
        Xcal_i = np.hstack([np.ones((len(Xcal), 1)), Xcal])

        # fit B on TRAIN
        self.B = self._fit_B_rrr(Xtr_i, Ytr)

        # calibrate threshold on CALIBRATION residuals (out-of-sample)
        Yhat_cal = Xcal_i @ self.B
        cal_scores = self._resid_score(Ycal, Yhat_cal)

        if self.threshold_method == "quantile":
            # target FPR ~ alpha on clean calibration
            self.threshold = float(np.quantile(cal_scores, 1.0 - self.alpha))

        elif self.threshold_method == "mad":
            self.med = float(np.median(cal_scores))
            self.mad = float(np.median(np.abs(cal_scores - self.med)) + 1e-8)
            self.threshold = float(self.med + self.k_mad * self.mad)

        else:
            raise ValueError("threshold_method must be 'quantile' or 'mad'.")

        return self

    def decision_function(self, Xp, Yf):
        if self.B is None:
            raise RuntimeError("Call fit() first.")
        X, Y = self._prepare(Xp, Yf)
        Yhat = X @ self.B
        return self._resid_score(Y, Yhat)

    def predict(self, Xp, Yf):
        if self.threshold is None:
            raise RuntimeError("Call fit() first.")
        scores = self.decision_function(Xp, Yf)
        return (scores > self.threshold).astype(int)
