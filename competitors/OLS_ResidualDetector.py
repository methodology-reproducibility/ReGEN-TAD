import numpy as np

class OLS_ResidualDetector:
    """
    Full-rank multivariate OLS residual-based anomaly detector.

    - Fits Y = X B via OLS
    - Scores via residual mean squared error
    - Threshold calibrated on training residuals (MAD)
    """

    def __init__(self, k_mad=3.5):
        self.k_mad = k_mad

        self.B = None
        self.med = None
        self.mad = None
        self.threshold = None

    def _prepare(self, Xp, Yf):
        """
        Flatten window tensors:
        Xp: (N, L, p)
        Yf: (N, H, p)
        """
        X = Xp.reshape(len(Xp), -1)
        Y = Yf.reshape(len(Yf), -1)

        X = np.hstack([np.ones((len(X), 1)), X])

        return X, Y

    def fit(self, Xp, Yf):
        """
        Fit OLS and calibrate threshold on training data only.
        """
        X, Y = self._prepare(Xp, Yf)

        self.B = np.linalg.pinv(X) @ Y

        Yhat = X @ self.B
        resid = Y - Yhat
        err = np.mean(resid**2, axis=1)

        self.med = np.median(err)
        self.mad = np.median(np.abs(err - self.med)) + 1e-8
        self.threshold = self.med + self.k_mad * self.mad

        return self

    def decision_function(self, Xp, Yf):
        """
        Returns residual score (MSE per window).
        """
        X, Y = self._prepare(Xp, Yf)
        Yhat = X @ self.B
        resid = Y - Yhat
        err = np.mean(resid**2, axis=1)
        return err

    def predict(self, Xp, Yf):
        """
        Binary anomaly labels.
        """
        scores = self.decision_function(Xp, Yf)
        return (scores > self.threshold).astype(int)
