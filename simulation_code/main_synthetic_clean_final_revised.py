# pyrefly: ignore [missing-import]
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITORS_DIR = PROJECT_ROOT / "competitors"
for path in (PROJECT_ROOT, COMPETITORS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DAGMM import DAGMM
from DeepANT import DeepAnt
from GARCH_Anomaly import GARCH_Baseline
from IsolationForestDetector import IsolationForestDetector
from LSTM_NDT import LSTM_NDT
from RRR_ResidualDetector import RRR_ResidualDetector
from TGANAD import TGANAD
from TimeGPTMultivariateDetector import TimeGPTMultivariateDetector
from TranAD import TranAD
from ReGENTAD import ReGENTAD
from AlioghliOkay2025 import AlioghliOkay2025

try:
    from nixtla import NixtlaClient
except ImportError:
    NixtlaClient = None

timegpt_api_key = os.environ.get("NIXTLA_API_KEY") or os.environ.get("TIMEGPT_API_KEY") or ""
nixtla_client = NixtlaClient(api_key=timegpt_api_key) if NixtlaClient and timegpt_api_key else None



class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start





import gc

def set_all_seeds(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def generate_clean_data(
    n=500,
    p=100,
    dgp="iid_normal",
    n_factors=3,
    df=5,
    seed=0,
):
    """
    Generate clean (no-anomaly) multivariate time series data.

    Parameters
    ----------
    n : int
        Number of time steps.
    p : int
        Number of dimensions/assets.
    dgp : str
        Data generating process.
    n_factors : int
        Number of latent factors (for factor models).
    df : int
        Degrees of freedom for t-distribution.
    seed : int
        Random seed.

    Returns
    -------
    X : ndarray, shape (n, p)
    y : ndarray, shape (n,), all zeros
    """
    rng = np.random.default_rng(seed)

    if dgp == "iid_normal":
        mu = rng.uniform(-0.0005, 0.0005, size=p)
        sigma = rng.uniform(0.007, 0.015, size=p)
        X = rng.normal(mu, sigma, size=(n, p))

    elif dgp == "iid_t":
        scale = rng.uniform(0.007, 0.015, size=p)
        X = rng.standard_t(df, size=(n, p))
        X = X / np.sqrt(df / (df - 2))
        X *= scale

    elif dgp == "garch":
        omega = 0.000001
        alpha = 0.05
        beta = 0.9

        X = np.zeros((n, p))
        sigma2 = np.ones(p) * 0.0001

        for t in range(1, n):
            eps = rng.normal(size=p)
            sigma2 = omega + alpha * (X[t - 1] ** 2) + beta * sigma2
            X[t] = np.sqrt(sigma2) * eps

    elif dgp == "factor":
        B = rng.normal(0, 0.5, size=(p, n_factors))
        F = rng.normal(0, 0.01, size=(n, n_factors))
        U = rng.normal(0, 0.01, size=(n, p))
        X = F @ B.T + U

    elif dgp == "factor_garch":
        B = rng.normal(0, 0.5, size=(p, n_factors))

        F = np.zeros((n, n_factors))
        sigma2 = np.ones(n_factors) * 0.0001
        omega, alpha, beta = 0.000001, 0.05, 0.9

        for t in range(1, n):
            eps = rng.normal(size=n_factors)
            sigma2 = omega + alpha * (F[t - 1] ** 2) + beta * sigma2
            F[t] = np.sqrt(sigma2) * eps

        U = rng.normal(0, 0.01, size=(n, p))
        X = F @ B.T + U

    elif dgp == "var":
        A = rng.normal(0, 0.05, size=(p, p))
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig >= 1:
            A = A / (1.1 * max_eig)

        X = np.zeros((n, p))
        eps = rng.normal(0, 0.01, size=(n, p))

        for t in range(1, n):
            X[t] = A @ X[t - 1] + eps[t]

    elif dgp == "seasonal_var":
        A = np.eye(p) * 0.3
        X = np.zeros((n, p))
        eps = rng.normal(0, 0.01, size=(n, p))

        period = 50
        t_grid = np.arange(n)

        base_season = np.sin(2 * np.pi * t_grid / period)        # (n,)
        base_season = base_season[:, None]                       # (n,1)

        amplitudes = rng.uniform(0.001, 0.003, size=(1, p))      # (1,p)

        seasonal = base_season @ amplitudes                      # (n,p)

        for t in range(1, n):
            X[t] = A @ X[t - 1] + eps[t]

        X = X + seasonal

    elif dgp == "smooth_vol_drift":
        base_sigma = 0.01
        t_grid = np.linspace(0, 1, n)
        sigma_t = base_sigma * (1 + 0.5 * np.sin(2 * np.pi * t_grid))
        X = rng.normal(0, 1, size=(n, p))
        X *= sigma_t[:, None]

    else:
        raise ValueError(f"Unknown dgp: {dgp}")

    y = np.zeros(n, dtype=int)
    return X, y


def make_windows(X, y, past_len, horizon):
    Xp, Yf, yw = [], [], []
    for t in range(past_len, len(X) - horizon):
        Xp.append(X[t - past_len : t])
        Yf.append(X[t : t + horizon])
        yw.append(y[t])
    return np.asarray(Xp), np.asarray(Yf), np.asarray(yw)


def eval_metrics(y_true, y_pred, scores):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores, dtype=float)

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    fpr = ((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())
    au = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
    alert_rate = float((y_pred == 1).mean())
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_p95 = float(np.quantile(scores, 0.95))
    score_p99 = float(np.quantile(scores, 0.99))
    return (
        float(p),
        float(r),
        float(f),
        float(au),
        float(fpr),
        alert_rate,
        score_mean,
        score_std,
        score_p95,
        score_p99,
    )


def contaminate_training_data(Xp_tr, Yf_tr, contam_rate, rng):
    if contam_rate <= 0:
        return Xp_tr, Yf_tr

    n_train = len(Xp_tr)
    n_contam = int(np.ceil(contam_rate * n_train))
    contam_idx = rng.choice(n_train, n_contam, replace=False)

    Xp_contam = Xp_tr.copy()
    Yf_contam = Yf_tr.copy()

    for idx in contam_idx:
        contam_type = rng.choice(["shift", "scale", "spike", "noise"])
        if contam_type == "shift":
            shift = rng.uniform(-0.05, 0.05)
            Xp_contam[idx] += shift
            Yf_contam[idx] += shift
        elif contam_type == "scale":
            scale = rng.uniform(1.5, 3.0)
            Xp_contam[idx] *= scale
            Yf_contam[idx] *= scale
        elif contam_type == "spike":
            n_spikes = rng.integers(1, 4)
            spike_t = rng.choice(Xp_contam.shape[1], n_spikes, replace=False)
            spike_f = rng.choice(Xp_contam.shape[2], n_spikes, replace=True)
            for t, f in zip(spike_t, spike_f):
                Xp_contam[idx, t, f] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
        else:
            noise_scale = rng.uniform(2.0, 4.0)
            Xp_contam[idx] += rng.normal(0, 0.01 * noise_scale, Xp_contam[idx].shape)
            Yf_contam[idx] += rng.normal(0, 0.01 * noise_scale, Yf_contam[idx].shape)

    return Xp_contam, Yf_contam


def run_one_model(model_name, past_len, horizon, dim, Xp_tr_contam, Yf_tr_contam, Xp, Yf):
    if model_name == "ReGENTAD":
        model = ReGENTAD(past_len, horizon, dim)
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, verbose=0)
        yhat, scores = model.predict(Xp, Yf, return_scores=True)
        return yhat, scores

    if model_name == "LSTM_NDT":
        model = LSTM_NDT(past_len, horizon, dim)
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, verbose=0)
        scores = model.decision_function(Xp, Yf)
        yhat = model.predict(Xp, Yf)
        return yhat, scores

    if model_name == "DeepANT":
        model = DeepAnt(past_len, horizon, dim)
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, verbose=0)
        scores = model.decision_function(Xp, Yf)
        yhat = model.predict(Xp, Yf)
        return yhat, scores

    if model_name == "TranAD":
        x_tranad = np.concatenate([Xp, Yf[:, :1, :]], axis=1)
        x_tranad_tr = np.concatenate([Xp_tr_contam, Yf_tr_contam[:, :1, :]], axis=1)
        model = TranAD(past_len, dim, rank_top_frac=0.05)
        model.fit(x_tranad_tr, epochs=40, verbose=0)
        yhat, scores = model.predict(x_tranad, return_scores=True)
        return yhat, scores

    if model_name == "DAGMM":
        model = DAGMM(past_len, dim)
        model.fit(Xp_tr_contam, epochs=40, verbose=0)
        scores = model.decision_function(Xp)
        yhat = model.predict(Xp)
        return yhat, scores

    if model_name == "AlioghliOkay2025":
        model = AlioghliOkay2025(
            past_len=past_len,
            horizon=horizon,
            n_features=dim,
            d_model=128,
            num_heads=8,
            ff_dim=256,
            dropout=0.1,
            alpha=0.05,
            k_sigma=3.0,
        )
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, batch_size=32, verbose=0)
        scores = model.decision_function(Xp, Yf)
        yhat = model.predict(Xp, Yf)
        return yhat, scores

    if model_name == "TGANAD":
        model = TGANAD(
            past_len=past_len,
            n_features=dim,
            d_model=64,
            num_heads=4,
            ff_dim=128,
            lambda_adv=0.1,
        )
        model.fit(Xp_tr_contam, epochs=40, batch_size=32, verbose=0)
        scores = model.decision_function(Xp)
        yhat = model.predict(Xp)
        return yhat, scores

    if model_name == "IsolationForestDetector":
        model = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=42)
        model.fit(Xp_tr_contam, None, verbose=0)
        scores = model.decision_function(Xp)
        yhat = model.predict(Xp)
        return yhat, scores

    if model_name == "GARCH_Anomaly":
        model = GARCH_Baseline(alpha=0.05, k_sigma=3.0)
        model.fit(Xp_tr_contam, Yf_tr_contam, verbose=0)
        scores = model.decision_function(Xp, Yf)
        yhat = model.predict(Xp, Yf)
        return yhat, scores

    if model_name == "RRR_ResidualDetector":
        model = RRR_ResidualDetector(rank=None, k_mad=3.5)
        model.fit(Xp_tr_contam, Yf_tr_contam)
        scores = model.decision_function(Xp, Yf)
        yhat = model.predict(Xp, Yf)
        return yhat, scores

    if model_name == "TimeGPT":
        if NixtlaClient is None:
            raise RuntimeError("TimeGPT requires the nixtla package to be installed.")
        if not timegpt_api_key:
            raise RuntimeError("TimeGPT requires NIXTLA_API_KEY or TIMEGPT_API_KEY to be set.")
        client = NixtlaClient(api_key=timegpt_api_key)

        model = TimeGPTMultivariateDetector(
            client,
            model="timegpt-1",
            level=95
        )

        X_series = np.vstack([Xp[0], Yf[:, 0, :]])

        y_eval_local = np.asarray([
            (np.asarray(y).astype(int).max() if np.asarray(y).size else 0)
            for y in Yf
        ], dtype=int)

        scores, yhat, _ = model.score(
            X_series,
            y_eval_local,
            past_len,
            horizon
        )

        L = min(len(scores), len(y_eval_local))
        scores = scores[-L:]
        yhat = yhat[-L:]

        pad = len(y_eval_local) - L
        if pad > 0:
            scores = np.concatenate([np.zeros(pad), scores])
            yhat = np.concatenate([np.zeros(pad, dtype=int), yhat])

        return yhat.astype(int), scores

    raise ValueError(f"Unknown model: {model_name}")

    raise ValueError(f"Unknown model: {model_name}")


def save_checkpoint_atomic(results, checkpoint_file):
    tmp_file = checkpoint_file + ".tmp"
    pd.DataFrame(results).to_csv(tmp_file, index=False)
    os.replace(tmp_file, checkpoint_file)


PAST_LEN = 24
HORIZON = 6
N_ITER = 10

DIMENSIONS = [100]
SAMPLE_SIZES = [
    220,
    550,
    1100,
    2050,
    650,
]
TRAIN_FRACTIONS = [0.7]
CONTAMINATION_RATES = [0.0]
DGPS = [
    "iid_normal",
    "iid_t",
    "garch",
    "factor",
    "factor_garch",
    "var",
    "seasonal_var",
    "smooth_vol_drift",
]
MODELS = [
"ReGENTAD",
]


STOP_AFTER_NEW_ROWS = None  # e.g., 20 to stop after writing 20 new rows
VERBOSE_MODEL_ERRORS = True


def run_clean_study(
    past_len=PAST_LEN,
    horizon=HORIZON,
    n_iter=N_ITER,
    dimensions=DIMENSIONS,
    sample_sizes=SAMPLE_SIZES,
    train_fractions=TRAIN_FRACTIONS,
    contamination_rates=CONTAMINATION_RATES,
    dgps=DGPS,
    models=MODELS,
    stop_after_new_rows=STOP_AFTER_NEW_ROWS,
    verbose_model_errors=VERBOSE_MODEL_ERRORS,
):
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "synthetic_clean_ReGENTAD.csv"
    final_file = output_dir / "synthetic_clean_ReGENTAD_final.csv"

    results = []
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        try:
            ckpt = pd.read_csv(checkpoint_file)
            results = ckpt.to_dict("records")
            for row in results:
                key = (
                    row["DGP"],
                    int(row["Iteration"]),
                    int(row["Dim"]),
                    int(row["N_Total"]),
                    round(float(row["TrainFrac"]), 6),
                    round(float(row["ContamRate"]), 6),
                    row["Model"],
                )
                completed_keys.add(key)
            print(f"Resuming from checkpoint: {checkpoint_file} ({len(results)} rows)")
        except Exception as e:
            print(f"Checkpoint read failed ({e}); starting fresh.")

    total_iters = (
        len(dimensions)
        * len(sample_sizes)
        * len(train_fractions)
        * len(dgps)
        * len(contamination_rates)
        * n_iter
    )
    total_jobs = total_iters * len(models)
    current_iter = 0
    rows_added = 0
    global_start = time.perf_counter()

    for dim in dimensions:
        for n_total in sample_sizes:
            for train_frac in train_fractions:
                for dgp in dgps:
                    for contam_rate in contamination_rates:
                        for it in range(n_iter):
                            current_iter += 1
                            print(
                                f"[{current_iter}/{total_iters}] "
                                f"dim={dim}, n_total={n_total}, train_frac={train_frac}, "
                                f"dgp={dgp}, contam={contam_rate}, iter={it}, "
                                f"elapsed={time.perf_counter() - global_start:.1f}s"
                            )

                            seed = 1000 + it
                            rng = np.random.default_rng(seed)
                            set_all_seeds(seed)

                            try:
                                X, y = generate_clean_data(
                                    n=n_total,
                                    p=dim,
                                    dgp=dgp,
                                    seed=seed,
                                )
                                Xp, Yf, y_eval = make_windows(X, y, past_len, horizon)

                                if len(Xp) < 40:
                                    print(f"  Not enough windows ({len(Xp)}); skipping scenario.")
                                    continue

                                split_idx = int(np.floor(train_frac * len(Xp)))
                                split_idx = min(max(split_idx, 20), len(Xp) - 20)

                                Xp_tr = Xp[:split_idx]
                                Yf_tr = Yf[:split_idx]
                                Xp_eval = Xp[split_idx:]
                                Yf_eval = Yf[split_idx:]
                                y_eval_out = y_eval[split_idx:]

                                Xp_tr_contam, Yf_tr_contam = contaminate_training_data(
                                    Xp_tr, Yf_tr, contam_rate, rng
                                )

                                base_result = dict(
                                    DGP=dgp,
                                    Iteration=it,
                                    Dim=dim,
                                    N_Total=n_total,
                                    N_Train=len(Xp_tr),
                                    N_Eval=len(Xp_eval),
                                    TrainFrac=train_frac,
                                    ContamRate=contam_rate,
                                )

                                for model_idx, model_name in enumerate(models):
                                    key = (
                                        dgp,
                                        it,
                                        dim,
                                        n_total,
                                        round(float(train_frac), 6),
                                        round(float(contam_rate), 6),
                                        model_name,
                                    )
                                    if key in completed_keys:
                                        continue

                                    try:
                                        tf.keras.backend.clear_session()
                                        gc.collect()
                                        set_all_seeds(seed + model_idx)
                                        with Timer() as t:
                                            yhat, scores = run_one_model(
                                                model_name=model_name,
                                                past_len=past_len,
                                                horizon=horizon,
                                                dim=dim,
                                                Xp_tr_contam=Xp_tr_contam,
                                                Yf_tr_contam=Yf_tr_contam,
                                                Xp=Xp_eval,
                                                Yf=Yf_eval,
                                            )

                                        (
                                            p,
                                            r,
                                            f1,
                                            au,
                                            fpr,
                                            alert_rate,
                                            score_mean,
                                            score_std,
                                            score_p95,
                                            score_p99,
                                        ) = eval_metrics(
                                            y_eval_out,
                                            yhat,
                                            scores,
                                        )
                                        row = {
                                            **base_result,
                                            "Model": model_name,
                                            "Precision": p,
                                            "Recall": r,
                                            "F1": f1,
                                            "AUROC": au,
                                            "FPR": fpr,
                                            "AlertRate": alert_rate,
                                            "ScoreMean": score_mean,
                                            "ScoreStd": score_std,
                                            "ScoreP95": score_p95,
                                            "ScoreP99": score_p99,
                                            "Time": t.elapsed,
                                        }
                                        results.append(row)
                                        completed_keys.add(key)
                                        rows_added += 1
                                        save_checkpoint_atomic(results, checkpoint_file)
                                        del yhat, scores
                                        gc.collect()

                                        if (
                                            stop_after_new_rows is not None
                                            and rows_added >= int(stop_after_new_rows)
                                        ):
                                            print(f"Stopping early after {rows_added} new rows (manual limit).")
                                            df = pd.DataFrame(results)
                                            df.to_csv(final_file, index=False)
                                            return df, checkpoint_file, final_file

                                    except Exception as model_e:
                                        if verbose_model_errors:
                                            print(f"  {model_name} ERROR: {model_e}")

                            except Exception as scenario_e:
                                print(f"  Scenario ERROR: {scenario_e}")

    df = pd.DataFrame(results)
    df.to_csv(final_file, index=False)
    save_checkpoint_atomic(results, checkpoint_file)

    print("\nStudy complete.")
    print(f"Total jobs configured: {total_jobs}")
    print(f"Total rows produced: {len(df)}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Final output: {final_file}")
    return df, checkpoint_file, final_file


df, checkpoint_file, final_file = run_clean_study()
print("\nRows:", len(df))
if len(df) > 0:
    summary_cols = [
        "Precision",
        "Recall",
        "F1",
        "AUROC",
        "FPR",
        "AlertRate",
        "ScoreMean",
        "ScoreStd",
    ]
    display(df.groupby(["DGP", "Model"])[summary_cols].mean().round(4))


df

df.to_csv(PROJECT_ROOT / "results" / "clean_regentad.csv", index=False)
