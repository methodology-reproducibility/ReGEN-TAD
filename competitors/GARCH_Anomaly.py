# ============================================================
# GARCH Baseline
# (Standard Econometric Volatility-Burst Detector)
# ============================================================

import numpy as np

try:
    from arch import arch_model
except ImportError:  # pragma: no cover - optional dependency guard
    arch_model = None

class GARCH_Baseline:
    """
    Standard GARCH(1,1) baseline for multivariate returns.
    Fits a GARCH model to the cross-sectional mean (Market Factor) 
    to detect systemic volatility-based anomalies.
    """
    def __init__(self, alpha=0.05, k_sigma=3.0):
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.model_res = None
        self.threshold = None

    def _get_market_factor(self, X_win):
        # X_win is (N, seq_len, p) or (N, p)
        # We take the mean across the p features to get the 'Market Return'
        if X_win.ndim == 3:
            return np.mean(X_win[:, -1, :], axis=1)
        return np.mean(X_win, axis=1)

    def fit(self, X_tr_win, Y_f_win, verbose=0):
        if arch_model is None:
            raise ImportError(
                "arch is required for GARCH_Baseline. Install with `pip install arch`."
            )

        # Fit on the market factor of the training data
        market_returns = self._get_market_factor(X_tr_win)
        
        # Rescale returns if they are too small (improves GARCH convergence)
        self.scale = 1.0
        if np.std(market_returns) < 0.01:
            self.scale = 100.0
            
        scaled_returns = market_returns * self.scale
        
        # GARCH(1,1) is the industry standard for financial time series [cite: 74]
        am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
        self.model_res = am.fit(disp='off')
        
        # Thresholding based on standardized residuals (Innovations / Volatility)
        # This flags observations that 'significantly exceed predicted variance' [cite: 143]
        std_resid = self.model_res.resid / self.model_res.conditional_volatility
        self.threshold = np.quantile(np.abs(std_resid), 1 - self.alpha)
        return self

    def decision_function(self, X_win, Y_f_win):
        # Forecast volatility for the evaluation windows
        market_eval = self._get_market_factor(X_win) * self.scale
        
        # For simplicity in a baseline, we use the model's standardized residual 
        # relative to the fitted parameters. 
        mu = self.model_res.params['mu']
        omega = self.model_res.params['omega']
        alpha1 = self.model_res.params['alpha[1]']
        beta1 = self.model_res.params['beta[1]']
        
        # Compute conditional volatility recursively (Standard GARCH recursion)
        vols = np.zeros_like(market_eval)
        vols[0] = np.var(market_eval) # Seed with variance
        for t in range(1, len(market_eval)):
            vols[t] = omega + alpha1 * (market_eval[t-1] - mu)**2 + beta1 * vols[t-1]
            
        scores = np.abs(market_eval - mu) / np.sqrt(vols + 1e-8)
        return scores

    def predict(self, X_win, Y_f_win):
        scores = self.decision_function(X_win, Y_f_win)
        return (scores > self.threshold).astype(int)
