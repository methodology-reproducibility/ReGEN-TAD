#!/usr/bin/env python
# coding: utf-8

# # Synthetic Stock Simulations
# 
# This notebook runs the stock simulation using the unified `ReGENTAD` class and compares the three new decision-rule variants with the competing benchmark methodologies in one self-contained notebook.
# 
# Evaluation is controlled by a single flag:
# 
# - `EVALUATION_SCHEME = "whole set"`
# - `EVALUATION_SCHEME = "test set"`
# 
# Notes:
# 
# - the stock DGP, anomaly generation, contamination logic, seeds, dimensions, and simulation grid are preserved
# - legacy `ReGENTAD` classes and wrapper logic are not used
# - each model is fit on pre-shock training windows only
# - results from different evaluation modes are stored in one shared output/checkpoint file using the `EvaluationScheme` flag
# 

# In[1]:


import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPETITORS_DIR = PROJECT_ROOT / "competitors"
for path in (PROJECT_ROOT, COMPETITORS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AlioghliOkay2025 import AlioghliOkay2025
from DAGMM import DAGMM
from DeepANT import DeepAnt
from GARCH_Anomaly import GARCH_Baseline
from IsolationForestDetector import IsolationForestDetector
from LSTM_NDT import LSTM_NDT
from OLS_ResidualDetector import OLS_ResidualDetector
from RRR_ResidualDetector import RRR_ResidualDetector
from ReGENTAD import ReGENTAD
from TGANAD import TGANAD
from TimeGPTMultivariateDetector import TimeGPTMultivariateDetector
from TranAD import TranAD

try:
    from nixtla import NixtlaClient
except ImportError:
    NixtlaClient = None

timegpt_api_key = os.environ.get("NIXTLA_API_KEY") or os.environ.get("TIMEGPT_API_KEY") or ""
if timegpt_api_key:
    os.environ["NIXTLA_API_KEY"] = timegpt_api_key
    os.environ.setdefault("TIMEGPT_API_KEY", timegpt_api_key)

if NixtlaClient is None or not timegpt_api_key:
    nixtla_client = None
else:
    nixtla_client = NixtlaClient(api_key=timegpt_api_key)


# In[5]:


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def set_all_seeds(seed):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def generate_stock_like_data(
    n_normal=450,
    n_shock=50,
    p=250,
    anomaly_type="bear_market",
    shock_sign="random",
    frac_affected=0.5,
    seed=0,
):
    rng = np.random.default_rng(seed)

    mu0 = rng.uniform(-0.0005, 0.0005)
    sig0 = rng.uniform(0.007, 0.015)
    R0 = rng.normal(mu0, sig0, size=(n_normal, p))

    if shock_sign == "random":
        sign = rng.choice([-1, 1])
    elif shock_sign == "positive":
        sign = 1
    else:
        sign = -1

    n_aff = max(1, int(np.ceil(frac_affected * p)))
    affected = rng.choice(p, n_aff, replace=False)
    R1 = np.zeros((n_shock, p))

    if anomaly_type in {"bear_market", "bull_market", "mean_shift"}:
        mu = sign * rng.uniform(0.01, 0.04)
        market = rng.normal(mu, sig0, size=(n_shock, 1))
        idio = rng.normal(0, sig0 * 0.5, size=(n_shock, n_aff))
        R1[:, affected] = market + idio
    elif anomaly_type == "volatility_spike":
        sig = rng.uniform(0.03, 0.06)
        R1[:, affected] = rng.normal(mu0, sig, size=(n_shock, n_aff))
    elif anomaly_type == "trend_reversal":
        mu_trend = -mu0 * rng.uniform(4, 6)
        market = rng.normal(mu_trend, sig0 * 1.5, size=(n_shock, 1))
        R1[:, affected] = market
    elif anomaly_type == "flash_crash":
        R1[:] = rng.normal(mu0, sig0, size=(n_shock, p))
        crash_t = rng.integers(0, n_shock)
        R1[crash_t, affected] -= rng.uniform(0.15, 0.30)
    elif anomaly_type == "sector_shock":
        mu = sign * rng.uniform(0.02, 0.05)
        R1[:, affected] = rng.normal(mu, sig0 * 1.5, size=(n_shock, n_aff))
    elif anomaly_type == "liquidity_dryup":
        R1[:, affected] = rng.normal(mu0, sig0 * 4.0, size=(n_shock, n_aff))
    elif anomaly_type == "regime_switch":
        mu = sign * rng.uniform(0.01, 0.03)
        sig = rng.uniform(0.03, 0.06)
        market = rng.normal(mu, sig, size=(n_shock, 1))
        idio = rng.normal(0, sig, size=(n_shock, n_aff))
        R1[:, affected] = market + idio
    elif anomaly_type == "correlation_breakdown":
        sig_shock = sig0 * rng.uniform(2.0, 3.0)
        R1[:, affected] = rng.normal(0, sig_shock, size=(n_shock, n_aff))
        n_spikes = max(1, n_shock // 10)
        spike_times = rng.choice(n_shock, n_spikes, replace=False)
        spike_assets = rng.choice(n_aff, n_spikes, replace=True)
        for t, a in zip(spike_times, spike_assets):
            R1[t, affected[a]] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
    elif anomaly_type == "contagion":
        n_initial = max(1, n_aff // 5)
        spread_rate = (n_aff - n_initial) / max(1, n_shock - 1)
        mu_shock = sign * rng.uniform(0.02, 0.04)
        sig_shock = sig0 * 1.5
        for t in range(n_shock):
            n_affected_t = min(n_aff, int(n_initial + spread_rate * t))
            affected_t = affected[:n_affected_t]
            R1[t, affected_t] = rng.normal(mu_shock, sig_shock, size=n_affected_t)
            unaffected_t = affected[n_affected_t:]
            if len(unaffected_t) > 0:
                R1[t, unaffected_t] = rng.normal(mu0, sig0, size=len(unaffected_t))
    elif anomaly_type == "momentum_crash":
        n_winners = n_aff // 2
        winners = affected[:n_winners]
        losers = affected[n_winners:]
        mu_reversal = rng.uniform(0.03, 0.06)
        sig_shock = sig0 * 2.0
        R1[:, winners] = rng.normal(-mu_reversal, sig_shock, size=(n_shock, len(winners)))
        if len(losers) > 0:
            R1[:, losers] = rng.normal(mu_reversal, sig_shock, size=(n_shock, len(losers)))
    elif anomaly_type == "fat_tail_event":
        df_t = rng.uniform(2.5, 4.0)
        scale = sig0 * 1.5
        R1[:, affected] = rng.standard_t(df_t, size=(n_shock, n_aff)) * scale
        n_extreme = max(1, n_shock // 5)
        extreme_times = rng.choice(n_shock, n_extreme, replace=False)
        extreme_assets = rng.choice(n_aff, n_extreme, replace=True)
        for t, a in zip(extreme_times, extreme_assets):
            R1[t, affected[a]] += rng.choice([-1, 1]) * rng.uniform(0.10, 0.25)
    elif anomaly_type == "microstructure_noise":
        base_returns = rng.normal(mu0, sig0, size=(n_shock, n_aff))
        bounce_amplitude = rng.uniform(0.005, 0.015)
        bounce = np.zeros((n_shock, n_aff))
        for i in range(n_aff):
            phase = rng.uniform(0, 2 * np.pi)
            freq = rng.uniform(0.3, 0.7)
            bounce[:, i] = bounce_amplitude * np.sin(freq * np.arange(n_shock) + phase)
        noise_bursts = rng.choice(n_shock, size=max(1, n_shock // 10), replace=False)
        burst_noise = np.zeros((n_shock, n_aff))
        for t in noise_bursts:
            burst_noise[t, :] = rng.normal(0, sig0 * 3, size=n_aff)
        R1[:, affected] = base_returns + bounce + burst_noise
    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    X = np.vstack([R0, R1])
    y = np.zeros(len(X), dtype=int)
    y[n_normal:] = 1
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
    try:
        aucroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        aucroc = float("nan")
    return float(p), float(r), float(f), float(fpr), aucroc


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


MODEL_VARIANTS = {
    "ReGENTAD_rank": {
        "decision_rule": "rank",
        "predict_kwargs": {},
    },
    "ReGENTAD_threshold": {
        "decision_rule": "adaptive_threshold",
        "predict_kwargs": {},
    },
    "ReGENTAD_threshold_quantile": {
        "decision_rule": "adaptive_quantile_threshold",
        "predict_kwargs": {
            "min_history": 25,
            "quantile_buffer": 0.015,
        },
    },
}

REGENTADT_MODELS = list(MODEL_VARIANTS)
COMPETING_MODELS = [
    "DeepANT",
    "TranAD",
    "DAGMM",
    "AlioghliOkay2025",
    "TGANAD",
    "IsolationForestDetector",
    "GARCH_Anomaly",
    "OLS_ResidualDetector",
    "RRR_ResidualDetector",
    "TimeGPT",  # optional: requires `nixtla` and an API key in the notebook kernel
]
MODELS = REGENTADT_MODELS + COMPETING_MODELS

SHARED_D_MODEL = 128
SHARED_NUM_HEADS = 6
SHARED_FF_DIM = 128
SHARED_DROPOUT = 0.1


def build_regentadt_model(model_name, past_len, horizon, dim, seed):
    config = MODEL_VARIANTS[model_name]
    model = ReGENTAD(
        past_len=past_len,
        horizon=horizon,
        n_features=dim,
        d_model=SHARED_D_MODEL,
        num_heads=SHARED_NUM_HEADS,
        ff_dim=SHARED_FF_DIM,
        dropout=SHARED_DROPOUT,
        decision_rule=config["decision_rule"],
        random_state=seed,
    )
    return model, dict(config["predict_kwargs"])


def run_regentadt_model(model_name, past_len, horizon, dim, Xp_tr_contam, Yf_tr_contam, Xp_eval, Yf_eval, seed):
    model, predict_kwargs = build_regentadt_model(
        model_name=model_name,
        past_len=past_len,
        horizon=horizon,
        dim=dim,
        seed=seed,
    )
    model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, verbose=0)
    yhat, scores, parts, meta = model.predict(
        Xp_eval,
        Yf_eval,
        return_scores=True,
        return_parts=True,
        return_metadata=True,
        **predict_kwargs,
    )
    return yhat, scores, parts, meta


def run_competing_model(model_name, past_len, horizon, dim, Xp_tr_contam, Yf_tr_contam, Xp_eval, Yf_eval):
    if model_name == "DeepANT":
        model = DeepAnt(past_len, horizon, dim)
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, verbose=0)
        scores = model.decision_function(Xp_eval, Yf_eval)
        yhat = model.predict(Xp_eval, Yf_eval)
        return yhat, scores, None, None

    if model_name == "TranAD":
        x_tranad_eval = np.concatenate([Xp_eval, Yf_eval[:, :1, :]], axis=1)
        x_tranad_tr = np.concatenate([Xp_tr_contam, Yf_tr_contam[:, :1, :]], axis=1)
        model = TranAD(
            past_len,
            dim,
            d_model=SHARED_D_MODEL,
            num_heads=SHARED_NUM_HEADS,
            ff_dim=SHARED_FF_DIM,
            rank_top_frac=0.05,
        )
        model.fit(x_tranad_tr, epochs=40, verbose=0)
        yhat, scores = model.predict(x_tranad_eval, return_scores=True)
        return yhat, scores, None, None

    if model_name == "DAGMM":
        model = DAGMM(past_len, dim)
        model.fit(Xp_tr_contam, epochs=40, verbose=0)
        scores = model.decision_function(Xp_eval)
        yhat = model.predict(Xp_eval)
        return yhat, scores, None, None

    if model_name == "AlioghliOkay2025":
        model = AlioghliOkay2025(
            past_len=past_len,
            horizon=horizon,
            n_features=dim,
            d_model=SHARED_D_MODEL,
            num_heads=SHARED_NUM_HEADS,
            ff_dim=SHARED_FF_DIM,
            dropout=SHARED_DROPOUT,
            alpha=0.05,
            k_sigma=3.0,
        )
        model.fit(Xp_tr_contam, Yf_tr_contam, epochs=40, batch_size=32, verbose=0)
        scores = model.decision_function(Xp_eval, Yf_eval)
        yhat = model.predict(Xp_eval, Yf_eval)
        return yhat, scores, None, None

    if model_name == "TGANAD":
        model = TGANAD(
            past_len=past_len,
            n_features=dim,
            d_model=SHARED_D_MODEL,
            num_heads=SHARED_NUM_HEADS,
            ff_dim=SHARED_FF_DIM,
            dropout=SHARED_DROPOUT,
            lambda_adv=0.1,
        )
        model.fit(Xp_tr_contam, epochs=40, batch_size=32, verbose=0)
        scores = model.decision_function(Xp_eval)
        yhat = model.predict(Xp_eval)
        return yhat, scores, None, None

    if model_name == "IsolationForestDetector":
        model = IsolationForestDetector(contamination=0.05, n_estimators=100, random_state=42)
        model.fit(Xp_tr_contam, None, verbose=0)
        scores = model.decision_function(Xp_eval)
        yhat = model.predict(Xp_eval)
        return yhat, scores, None, None

    if model_name == "GARCH_Anomaly":
        model = GARCH_Baseline(alpha=0.05, k_sigma=3.0)
        model.fit(Xp_tr_contam, Yf_tr_contam, verbose=0)
        scores = model.decision_function(Xp_eval, Yf_eval)
        yhat = model.predict(Xp_eval, Yf_eval)
        return yhat, scores, None, None

    if model_name == "OLS_ResidualDetector":
        model = OLS_ResidualDetector()
        model.fit(Xp_tr_contam, Yf_tr_contam)
        scores = model.decision_function(Xp_eval, Yf_eval)
        yhat = model.predict(Xp_eval, Yf_eval)
        return yhat, scores, None, None

    if model_name == "RRR_ResidualDetector":
        model = RRR_ResidualDetector(rank=None, k_mad=3.5)
        model.fit(Xp_tr_contam, Yf_tr_contam)
        scores = model.decision_function(Xp_eval, Yf_eval)
        yhat = model.predict(Xp_eval, Yf_eval)
        return yhat, scores, None, None

    if model_name == "TimeGPT":
        client = nixtla_client
        if client is None:
            if NixtlaClient is None:
                raise RuntimeError("TimeGPT requires the nixtla package.")
            if not timegpt_api_key:
                raise RuntimeError("TimeGPT requires NIXTLA_API_KEY or TIMEGPT_API_KEY in the notebook kernel.")
            client = NixtlaClient(api_key=timegpt_api_key)

        model = TimeGPTMultivariateDetector(client, model="timegpt-1", level=95)
        X_series = np.vstack([Xp_eval[0], Yf_eval[:, 0, :]])
        alignment_stub = np.zeros(len(Xp_eval), dtype=int)
        scores, yhat, _ = model.score(X_series, alignment_stub, past_len, horizon)

        L = min(len(scores), len(Xp_eval))
        scores = np.asarray(scores[-L:], dtype=float)
        yhat = np.asarray(yhat[-L:], dtype=int)

        pad = len(Xp_eval) - L
        if pad > 0:
            scores = np.concatenate([np.zeros(pad, dtype=float), scores])
            yhat = np.concatenate([np.zeros(pad, dtype=int), yhat])

        return yhat.astype(int), scores, None, None

    raise ValueError(f"Unknown model: {model_name}")


def run_one_model(model_name, past_len, horizon, dim, Xp_tr_contam, Yf_tr_contam, Xp_eval, Yf_eval, seed):
    if model_name in MODEL_VARIANTS:
        return run_regentadt_model(
            model_name=model_name,
            past_len=past_len,
            horizon=horizon,
            dim=dim,
            Xp_tr_contam=Xp_tr_contam,
            Yf_tr_contam=Yf_tr_contam,
            Xp_eval=Xp_eval,
            Yf_eval=Yf_eval,
            seed=seed,
        )
    return run_competing_model(
        model_name=model_name,
        past_len=past_len,
        horizon=horizon,
        dim=dim,
        Xp_tr_contam=Xp_tr_contam,
        Yf_tr_contam=Yf_tr_contam,
        Xp_eval=Xp_eval,
        Yf_eval=Yf_eval,
    )


def save_checkpoint_atomic(results, checkpoint_file):
    checkpoint_file = Path(checkpoint_file)
    tmp_file = checkpoint_file.with_suffix(checkpoint_file.suffix + ".tmp")
    pd.DataFrame(results).to_csv(tmp_file, index=False)
    os.replace(tmp_file, checkpoint_file)


PAST_LEN = 24
HORIZON = 6
N_ITER = 2

DIMENSIONS = [100]
SAMPLE_SIZES = [
    (200, 20),
    (500, 50),
    (1000, 100),
    # (2000, 50),
    (500, 150),
]
CONTAMINATION_RATES = [0.01, 0.03, 0.05, 0.10, 0.12, 0.15]
ANOMALIES = [
    "bear_market",
    "bull_market",
    "volatility_spike",
    "trend_reversal",
    "flash_crash",
    "sector_shock",
    "liquidity_dryup",
    "regime_switch",
    "correlation_breakdown",
    "contagion",
    "momentum_crash",
    "fat_tail_event",
    "microstructure_noise",
]

EVALUATION_SCHEME = "test mixed"  # or "test set"

STOP_AFTER_NEW_ROWS = None
VERBOSE_MODEL_ERRORS = True


def normalize_evaluation_scheme(eval_scheme):
    raw = str(eval_scheme).strip().lower()
    aliases = {
        "whole set": "whole set",
        "whole_dataset": "whole set",
        "whole-dataset": "whole set",
        "test set": "test set",
        "test_set_only": "test set",
        "test-set-only": "test set",
        "test mixed": "test mixed",
        "test_mixed": "test mixed",
        "test-mixed": "test mixed",
    }
    if raw not in aliases:
        raise ValueError("evaluation_scheme must be 'whole set', 'test set', or 'test mixed'.")
    return aliases[raw]


def get_evaluation_data(eval_scheme, Xp, Yf, y_eval, shock_start):
    eval_scheme = normalize_evaluation_scheme(eval_scheme)
    if eval_scheme == "whole set":
        eval_idx = np.arange(len(y_eval))
        return Xp, Yf, eval_idx
    if eval_scheme == "test set":
        eval_idx = np.arange(shock_start, len(y_eval))
        if len(eval_idx) == 0:
            raise ValueError("No test windows found for test-set evaluation.")
        return Xp, Yf, eval_idx
    if eval_scheme == "test mixed":
        test_start = max(0, shock_start - 100)
        eval_idx = np.arange(test_start, len(y_eval))
        if len(eval_idx) == 0:
            raise ValueError("No test windows found for test-mixed evaluation.")
        return Xp, Yf, eval_idx
    raise ValueError("evaluation_scheme must be 'whole set', 'test set', or 'test mixed'.")


def project_root() -> Path:
    return PROJECT_ROOT


def output_dir() -> Path:
    env_override = os.environ.get("DECISION_RULES_OUTPUT_DIR")
    out = Path(env_override).expanduser().resolve() if env_override else project_root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_stock_study(
    past_len=PAST_LEN,
    horizon=HORIZON,
    n_iter=N_ITER,
    dimensions=None,
    sample_sizes=None,
    contamination_rates=None,
    anomalies=None,
    models=None,
    evaluation_scheme=EVALUATION_SCHEME,
    stop_after_new_rows=STOP_AFTER_NEW_ROWS,
    verbose_model_errors=VERBOSE_MODEL_ERRORS,
):
    dimensions = DIMENSIONS if dimensions is None else dimensions
    sample_sizes = SAMPLE_SIZES if sample_sizes is None else sample_sizes
    contamination_rates = CONTAMINATION_RATES if contamination_rates is None else contamination_rates
    anomalies = ANOMALIES if anomalies is None else anomalies
    models = MODELS if models is None else models
    evaluation_scheme = EVALUATION_SCHEME if evaluation_scheme is None else evaluation_scheme
    stop_after_new_rows = STOP_AFTER_NEW_ROWS if stop_after_new_rows is None else stop_after_new_rows
    verbose_model_errors = VERBOSE_MODEL_ERRORS if verbose_model_errors is None else verbose_model_errors

    evaluation_scheme = normalize_evaluation_scheme(evaluation_scheme)

    out_dir = output_dir()
    checkpoint_file = out_dir / "synthetic_stocks_simulations_final_checkpoint.csv"
    final_file = out_dir / "synthetic_stocks_simulations_final.csv"
    results = []
    completed_keys = set()

    if os.path.exists(checkpoint_file):
        try:
            ckpt = pd.read_csv(checkpoint_file)
            results = ckpt.to_dict("records")
            for row in results:
                key = (
                    row.get("EvaluationScheme", ""),
                    row["Anomaly"],
                    int(row["Iteration"]),
                    int(row["Dim"]),
                    int(row["N_Normal"]),
                    int(row["N_Shock"]),
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
        * len(anomalies)
        * len(contamination_rates)
        * n_iter
    )
    total_jobs = total_iters * len(models)
    current_iter = 0
    rows_added = 0
    global_start = time.perf_counter()

    for dim in dimensions:
        for (n_normal, n_shock) in sample_sizes:
            for anomaly in anomalies:
                for contam_rate in contamination_rates:
                    for it in range(n_iter):
                        current_iter += 1
                        print(
                            f"[{current_iter}/{total_iters}] "
                            f"dim={dim}, samples=({n_normal},{n_shock}), "
                            f"anomaly={anomaly}, contam={contam_rate}, iter={it}, "
                            f"elapsed={time.perf_counter() - global_start:.1f}s"
                        )

                        seed = 1000 + it
                        rng = np.random.default_rng(seed)
                        set_all_seeds(seed)

                        try:
                            X, y = generate_stock_like_data(
                                n_normal=n_normal,
                                n_shock=n_shock,
                                p=dim,
                                anomaly_type=anomaly,
                                seed=seed,
                            )
                            Xp, Yf, y_eval = make_windows(X, y, past_len, horizon)

                            shock_positions = np.where(y_eval == 1)[0]
                            if len(shock_positions) == 0:
                                print("  No shock windows found; skipping scenario.")
                                continue

                            shock_start = int(shock_positions[0])
                            Xp_tr = Xp[:shock_start]
                            Yf_tr = Yf[:shock_start]
                            if len(Xp_tr) < 20:
                                print(f"  Not enough pre-shock windows ({len(Xp_tr)}); skipping.")
                                continue

                            Xp_tr_contam, Yf_tr_contam = contaminate_training_data(
                                Xp_tr, Yf_tr, contam_rate, rng
                            )

                            try:
                                Xp_eval, Yf_eval, eval_idx = get_evaluation_data(
                                    evaluation_scheme,
                                    Xp=Xp,
                                    Yf=Yf,
                                    y_eval=y_eval,
                                    shock_start=shock_start,
                                )
                            except Exception as eval_e:
                                if verbose_model_errors:
                                    print(f"  {evaluation_scheme} ERROR: {eval_e}")
                                continue

                            base_result = dict(
                                EvaluationScheme=evaluation_scheme,
                                Anomaly=anomaly,
                                Iteration=it,
                                Dim=dim,
                                N_Normal=n_normal,
                                N_Shock=n_shock,
                                ContamRate=contam_rate,
                            )

                            for model_idx, model_name in enumerate(models):
                                key = (
                                    evaluation_scheme,
                                    anomaly,
                                    it,
                                    dim,
                                    n_normal,
                                    n_shock,
                                    round(float(contam_rate), 6),
                                    model_name,
                                )
                                if key in completed_keys:
                                    continue

                                try:
                                    tf.keras.backend.clear_session()
                                    gc.collect()
                                    model_seed = seed + model_idx
                                    set_all_seeds(model_seed)

                                    with Timer() as t:
                                        yhat, scores, parts, meta = run_one_model(
                                            model_name=model_name,
                                            past_len=past_len,
                                            horizon=horizon,
                                            dim=dim,
                                            Xp_tr_contam=Xp_tr_contam,
                                            Yf_tr_contam=Yf_tr_contam,
                                            Xp_eval=Xp_eval,
                                            Yf_eval=Yf_eval,
                                            seed=model_seed,
                                        )

                                    p, r, f1, fpr, aucroc = eval_metrics(y_eval[eval_idx], yhat[eval_idx], scores[eval_idx])

                                    row = {
                                        **base_result,
                                        "Model": model_name,
                                        "Precision": p,
                                        "Recall": r,
                                        "F1": f1,
                                        "FPR": fpr,
                                        "AUCROC": aucroc,
                                        "Time": t.elapsed,
                                    }
                                    results.append(row)
                                    completed_keys.add(key)
                                    rows_added += 1
                                    save_checkpoint_atomic(results, checkpoint_file)

                                    del yhat, scores, parts, meta
                                    gc.collect()

                                    if stop_after_new_rows is not None and rows_added >= int(stop_after_new_rows):
                                        print(f"Stopping early after {rows_added} new rows (manual limit).")
                                        df = pd.DataFrame(results)
                                        df.to_csv(final_file, index=False)
                                        return df, checkpoint_file, final_file

                                except Exception as model_e:
                                    if verbose_model_errors:
                                        print(f"  {evaluation_scheme} | {model_name} ERROR: {model_e}")

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


# In[6]:


df, checkpoint_file, final_file = run_stock_study()
print("\nRows:", len(df))
if len(df) > 0:
    summary = (
        df.groupby(["EvaluationScheme", "Model"])[["Precision", "Recall", "F1", "FPR", "AUCROC"]]
        .mean()
        .round(4)
        .sort_values(["EvaluationScheme", "F1"], ascending=[True, False])
    )
    print(summary)


# In[ ]:



