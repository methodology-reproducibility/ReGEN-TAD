import pandas as pd


class TimeGPTMultivariateDetector:
    def __init__(self, client, model="timegpt-1", level=95):
        self.client = client
        self.model = model
        self.level = level

    def score(self, X, y_eval, past_len, horizon):
        n = len(X)
        idx = pd.date_range("2023-01-01", periods=n, freq="D")

        df = pd.DataFrame(X, index=idx)
        df.index.name = "ds"
        df_long = df.reset_index().melt(id_vars=["ds"], var_name="unique_id", value_name="y")

        detection_size = min(max(n - 60, 1), 440)

        out = self.client.detect_anomalies_online(
            df_long,
            freq="D",
            model=self.model,
            level=self.level,
            h=horizon,
            detection_size=detection_size,
            threshold_method="multivariate",
        )

        sys = out.groupby("ds").first().reset_index()
        score_cols = [c for c in sys.columns if self.model in c.lower()]
        score_col = score_cols[0] if score_cols else "anomaly"

        scores = pd.to_numeric(sys[score_col], errors="coerce").fillna(0.0).values
        preds = sys["anomaly"].astype(int).values

        L = min(len(scores), len(y_eval))
        return scores[-L:], preds[-L:], y_eval[-L:]
