# ============================================================
# Isolation Forest Baseline
# (Classical Ensemble-based Outlier Detection)
# ============================================================

import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    """
    Isolation Forest baseline.
    Flattens the temporal windows into vectors to perform
    classical partitioning-based anomaly detection.
    """
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

    def _flatten(self, X):
        # X is (N, past_len, n_features) -> transform to (N, past_len * n_features)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X_tr, Y_tr=None, verbose=0):
        X_flat = self._flatten(X_tr)
        self.model.fit(X_flat)
        return self

    def decision_function(self, X):
        X_flat = self._flatten(X)
        # sklearn's decision_function returns lower values for more anomalous samples.
        # We negate it so that higher values = higher anomaly probability,
        # matching the AUROC requirements in eval_metrics.
        return -self.model.decision_function(X_flat)

    def predict(self, X):
        X_flat = self._flatten(X)
        # sklearn returns -1 for outliers and 1 for inliers.
        # We map -1 -> 1 (Anomaly) and 1 -> 0 (Normal).
        preds = self.model.predict(X_flat)
        return np.where(preds == -1, 1, 0)
