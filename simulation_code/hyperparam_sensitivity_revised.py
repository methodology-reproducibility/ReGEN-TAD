#!/usr/bin/env python
"""
Faithful two-stage hyperparameter sensitivity analysis for ReGENTAD.

This script keeps the ReGENTAD detector logic in `ReGENTAD.py` as the sole
source of truth and mirrors the synthetic simulation protocol used in:

  - synthetic_structural_simulations_final.ipynb
  - synthetic_stocks_simulations_final.ipynb

Only the requested architecture/optimizer parameters and scoring weights are
varied. All other detector settings are left at the defaults defined in
`ReGENTAD.py`.
"""

from __future__ import annotations

import argparse
import ast
import gc
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ReGENTAD import ReGENTAD


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    notebook_path: Path
    anomalies: Tuple[str, ...]
    sample_sizes: Tuple[Tuple[int, int], ...]
    dimensions: Tuple[int, ...]
    contamination_rates: Tuple[float, ...]
    n_iter: int
    notebook_default_eval: str


NOTEBOOK_PATHS = {
    "structural": HERE / "synthetic_structural_simulations_final_revised.py",
    "stocks": HERE / "synthetic_stocks_simulations_final_revised.py",
}

PAST_LEN = 24
HORIZON = 6
BASE_SEED = 1000
TOP_N = 20

# `notebook_default` resolves to:
#   structural -> anomalous_segment  (notebook "test set")
#   stocks     -> mixed_test_segment (notebook "test mixed")
EVALUATION_SCHEME = "notebook_default"
ACTIVE_PROTOCOLS = ("structural", "stocks")
STAGE2_ARCH_SOURCE = "best"  # "best" or "baseline"

ARCH_GRID: Dict[str, Sequence[float]] = {
    "d_model": (64, 128),
    "num_heads": (4, 6),
    "ff_dim": (64, 128),
    "lstm_units": (16, 32),
    "lr": (1e-3, 5e-4),
}

WEIGHT_GRID: Dict[str, Sequence[float]] = {
    "err": (0.4, 0.6, 0.8),
    "recon": (0.8, 1.0, 1.2),
    "knn": (0.1, 0.2, 0.3),
    "dyn": (0.1, 0.2, 0.3),
    "regime": (0.5, 0.7, 0.9),
    "vol": (0.4, 0.6, 0.8),
}

METRIC_COLS = ("Precision", "Recall", "F1", "FPR", "AUROC", "Runtime")

PROTOCOLS: Dict[str, ProtocolSpec] = {
    "structural": ProtocolSpec(
        name="structural",
        notebook_path=NOTEBOOK_PATHS["structural"],
        anomalies=(
            "mean_shift",
            "variance",
            "trend",
            "spike",
            "collective",
            "contextual",
        ),
        sample_sizes=((200, 20), (500, 100)),
        dimensions=(100,),
        contamination_rates=(0.01, 0.15),
        n_iter=5,
        notebook_default_eval="anomalous_segment",
    ),
    "stocks": ProtocolSpec(
        name="stocks",
        notebook_path=NOTEBOOK_PATHS["stocks"],
        anomalies=(
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
        ),
        sample_sizes=((200, 20), (500, 100)),
        dimensions=(100,),
        contamination_rates=(0.01, 0.15),
        n_iter=5,
        notebook_default_eval="mixed_test_segment",
    ),
}

EVAL_ALIASES = {
    "notebook_default": "notebook_default",
    "whole set": "whole_set",
    "whole_set": "whole_set",
    "whole-set": "whole_set",
    "whole dataset": "whole_set",
    "whole_dataset": "whole_set",
    "whole-dataset": "whole_set",
    "anomalous segment": "anomalous_segment",
    "anomalous_segment": "anomalous_segment",
    "anomalous-segment": "anomalous_segment",
    "test set": "anomalous_segment",
    "test_set": "anomalous_segment",
    "test-set": "anomalous_segment",
    "mixed test segment": "mixed_test_segment",
    "mixed_test_segment": "mixed_test_segment",
    "mixed-test-segment": "mixed_test_segment",
    "test mixed": "mixed_test_segment",
    "test_mixed": "mixed_test_segment",
    "test-mixed": "mixed_test_segment",
}

EVAL_DISPLAY = {
    "whole_set": "Whole Set",
    "anomalous_segment": "Anomalous Segment",
    "mixed_test_segment": "Mixed Test Segment",
}

CONFIG_GROUP_COLS = [
    "d_model",
    "num_heads",
    "ff_dim",
    "lstm_units",
    "lr",
    "err",
    "recon",
    "knn",
    "dyn",
    "regime",
    "vol",
]

_CLASS_BALANCE_LOGGED: set[Tuple[str, str, int, int, int]] = set()


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def results_dir() -> Path:
    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _function_defaults(fn: ast.FunctionDef) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    args = list(fn.args.args)[1:]  # skip self
    raw_defaults = list(fn.args.defaults)
    offset = len(args) - len(raw_defaults)
    for idx, node in enumerate(raw_defaults):
        value = _literal(node)
        defaults[args[offset + idx].arg] = value
    return defaults


def _extract_default_weights(init_fn: ast.FunctionDef) -> Dict[str, float]:
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "weights"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        for child in node.body:
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id == "weights":
                    value = _literal(child.value)
                    if isinstance(value, dict):
                        return {str(k): float(v) for k, v in value.items()}
    raise ValueError("Could not extract default ReGENTAD weights from ReGENTAD.py")


def extract_regentad_defaults(source_path: Path) -> Dict[str, Dict[str, Any]]:
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ReGENTAD"
        ),
        None,
    )
    if class_node is None:
        raise ValueError("Class `ReGENTAD` not found in ReGENTAD.py")

    fn_map = {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "fit", "predict"}
    }
    init_fn = fn_map["__init__"]
    return {
        "init": _function_defaults(init_fn),
        "fit": _function_defaults(fn_map["fit"]),
        "predict": _function_defaults(fn_map["predict"]),
        "weights": _extract_default_weights(init_fn),
    }


def _load_notebook_source(notebook_path: Path) -> str:
    if notebook_path.suffix == ".py":
        return notebook_path.read_text()
    nb = json.loads(notebook_path.read_text())
    chunks: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            chunks.append("".join(cell.get("source", [])))
    return "\n\n".join(chunks)


def extract_notebook_constants(notebook_path: Path) -> Dict[str, Any]:
    wanted = {
        "PAST_LEN",
        "HORIZON",
        "N_ITER",
        "DIMENSIONS",
        "SAMPLE_SIZES",
        "CONTAMINATION_RATES",
        "ANOMALIES",
        "EVALUATION_SCHEME",
        "SHARED_D_MODEL",
        "SHARED_NUM_HEADS",
        "SHARED_FF_DIM",
        "SHARED_DROPOUT",
    }
    tree = ast.parse(_load_notebook_source(notebook_path), filename=str(notebook_path))
    out: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = _literal(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted and value is not None:
                out[target.id] = value
    return out


def load_notebook_helpers(notebook_path: Path) -> Dict[str, Any]:
    helper_names = {
        "generate_stock_like_data",
        "make_windows",
        "contaminate_training_data",
        "normalize_evaluation_scheme",
        "get_evaluation_data",
    }
    tree = ast.parse(_load_notebook_source(notebook_path), filename=str(notebook_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {"np": np}
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace


def normalize_evaluation_scheme(eval_scheme: str) -> str:
    raw = str(eval_scheme).strip().lower()
    if raw not in EVAL_ALIASES:
        raise ValueError(
            "evaluation_scheme must be one of: "
            "'Whole Set', 'Anomalous Segment', 'Mixed Test Segment', or 'notebook_default'."
        )
    return EVAL_ALIASES[raw]


def resolve_evaluation_scheme(eval_scheme: str, protocol: ProtocolSpec) -> str:
    canonical = normalize_evaluation_scheme(eval_scheme)
    if canonical == "notebook_default":
        return protocol.notebook_default_eval
    return canonical


def generate_synthetic_series(
    *,
    n_normal: int,
    n_shock: int,
    p: int,
    anomaly_type: str,
    seed: int,
    shock_sign: str = "random",
    frac_affected: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    mu0 = rng.uniform(-0.0005, 0.0005)
    sig0 = rng.uniform(0.007, 0.015)
    r0 = rng.normal(mu0, sig0, size=(n_normal, p))

    if shock_sign == "random":
        sign = rng.choice([-1, 1])
    elif shock_sign == "positive":
        sign = 1
    else:
        sign = -1

    n_aff = max(1, int(np.ceil(frac_affected * p)))
    affected = rng.choice(p, n_aff, replace=False)
    r1 = np.zeros((n_shock, p))

    if anomaly_type in {"bear_market", "bull_market", "mean_shift"}:
        mu = sign * rng.uniform(0.01, 0.04)
        market = rng.normal(mu, sig0, size=(n_shock, 1))
        idio = rng.normal(0, sig0 * 0.5, size=(n_shock, n_aff))
        r1[:, affected] = market + idio
    elif anomaly_type in {"volatility_spike", "variance"}:
        sig = rng.uniform(0.03, 0.06)
        r1[:, affected] = rng.normal(mu0, sig, size=(n_shock, n_aff))
    elif anomaly_type in {"trend_reversal", "trend"}:
        mu_trend = -mu0 * rng.uniform(4, 6)
        market = rng.normal(mu_trend, sig0 * 1.5, size=(n_shock, 1))
        r1[:, affected] = market
    elif anomaly_type in {"flash_crash", "spike"}:
        r1[:] = rng.normal(mu0, sig0, size=(n_shock, p))
        crash_t = rng.integers(0, n_shock)
        r1[crash_t, affected] -= rng.uniform(0.15, 0.30)
    elif anomaly_type == "collective":
        collective_level = sign * rng.uniform(0.01, 0.04)
        collective_noise = rng.normal(0, sig0 * 0.15, size=(1, n_aff))
        r1[:, affected] = collective_level + collective_noise
    elif anomaly_type == "contextual":
        base = rng.normal(mu0, sig0, size=(n_shock, n_aff))
        context_boost = rng.uniform(0.02, 0.05)
        r1[:, affected] = base + context_boost * (base > 0)
    elif anomaly_type == "sector_shock":
        mu = sign * rng.uniform(0.02, 0.05)
        r1[:, affected] = rng.normal(mu, sig0 * 1.5, size=(n_shock, n_aff))
    elif anomaly_type == "liquidity_dryup":
        r1[:, affected] = rng.normal(mu0, sig0 * 4.0, size=(n_shock, n_aff))
    elif anomaly_type == "regime_switch":
        mu = sign * rng.uniform(0.01, 0.03)
        sig = rng.uniform(0.03, 0.06)
        market = rng.normal(mu, sig, size=(n_shock, 1))
        idio = rng.normal(0, sig, size=(n_shock, n_aff))
        r1[:, affected] = market + idio
    elif anomaly_type == "correlation_breakdown":
        sig_shock = sig0 * rng.uniform(2.0, 3.0)
        r1[:, affected] = rng.normal(0, sig_shock, size=(n_shock, n_aff))
        n_spikes = max(1, n_shock // 10)
        spike_times = rng.choice(n_shock, n_spikes, replace=False)
        spike_assets = rng.choice(n_aff, n_spikes, replace=True)
        for t, a in zip(spike_times, spike_assets):
            r1[t, affected[a]] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
    elif anomaly_type == "contagion":
        n_initial = max(1, n_aff // 5)
        spread_rate = (n_aff - n_initial) / max(1, n_shock - 1)
        mu_shock = sign * rng.uniform(0.02, 0.04)
        sig_shock = sig0 * 1.5
        for t in range(n_shock):
            n_affected_t = min(n_aff, int(n_initial + spread_rate * t))
            affected_t = affected[:n_affected_t]
            r1[t, affected_t] = rng.normal(mu_shock, sig_shock, size=n_affected_t)
            unaffected_t = affected[n_affected_t:]
            if len(unaffected_t) > 0:
                r1[t, unaffected_t] = rng.normal(mu0, sig0, size=len(unaffected_t))
    elif anomaly_type == "momentum_crash":
        n_winners = n_aff // 2
        winners = affected[:n_winners]
        losers = affected[n_winners:]
        mu_reversal = rng.uniform(0.03, 0.06)
        sig_shock = sig0 * 2.0
        r1[:, winners] = rng.normal(-mu_reversal, sig_shock, size=(n_shock, len(winners)))
        if len(losers) > 0:
            r1[:, losers] = rng.normal(mu_reversal, sig_shock, size=(n_shock, len(losers)))
    elif anomaly_type == "fat_tail_event":
        df_t = rng.uniform(2.5, 4.0)
        scale = sig0 * 1.5
        r1[:, affected] = rng.standard_t(df_t, size=(n_shock, n_aff)) * scale
        n_extreme = max(1, n_shock // 5)
        extreme_times = rng.choice(n_shock, n_extreme, replace=False)
        extreme_assets = rng.choice(n_aff, n_extreme, replace=True)
        for t, a in zip(extreme_times, extreme_assets):
            r1[t, affected[a]] += rng.choice([-1, 1]) * rng.uniform(0.10, 0.25)
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
        r1[:, affected] = base_returns + bounce + burst_noise
    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    x = np.vstack([r0, r1])
    y = np.zeros(len(x), dtype=int)
    y[n_normal:] = 1
    return x, y


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    past_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xp, yf, yw = [], [], []
    for t in range(past_len, len(x) - horizon):
        xp.append(x[t - past_len : t])
        yf.append(x[t : t + horizon])
        yw.append(y[t])
    return np.asarray(xp), np.asarray(yf), np.asarray(yw)


def contaminate_training_data(
    xp_tr: np.ndarray,
    yf_tr: np.ndarray,
    contam_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if contam_rate <= 0:
        return xp_tr, yf_tr

    n_train = len(xp_tr)
    n_contam = int(np.ceil(contam_rate * n_train))
    contam_idx = rng.choice(n_train, n_contam, replace=False)

    xp_contam = xp_tr.copy()
    yf_contam = yf_tr.copy()

    for idx in contam_idx:
        contam_type = rng.choice(["shift", "scale", "spike", "noise"])
        if contam_type == "shift":
            shift = rng.uniform(-0.05, 0.05)
            xp_contam[idx] += shift
            yf_contam[idx] += shift
        elif contam_type == "scale":
            scale = rng.uniform(1.5, 3.0)
            xp_contam[idx] *= scale
            yf_contam[idx] *= scale
        elif contam_type == "spike":
            n_spikes = rng.integers(1, 4)
            spike_t = rng.choice(xp_contam.shape[1], n_spikes, replace=False)
            spike_f = rng.choice(xp_contam.shape[2], n_spikes, replace=True)
            for t, f in zip(spike_t, spike_f):
                xp_contam[idx, t, f] += rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
        else:
            noise_scale = rng.uniform(2.0, 4.0)
            xp_contam[idx] += rng.normal(0, 0.01 * noise_scale, xp_contam[idx].shape)
            yf_contam[idx] += rng.normal(0, 0.01 * noise_scale, yf_contam[idx].shape)

    return xp_contam, yf_contam


def get_evaluation_indices(
    eval_scheme: str,
    y_eval: np.ndarray,
    shock_start: int,
) -> np.ndarray:
    if eval_scheme == "whole_set":
        eval_idx = np.arange(len(y_eval))
    elif eval_scheme == "anomalous_segment":
        eval_idx = np.arange(shock_start, len(y_eval))
    elif eval_scheme == "mixed_test_segment":
        eval_idx = np.arange(max(0, shock_start - 100), len(y_eval))
    else:
        raise ValueError(f"Unknown canonical evaluation scheme: {eval_scheme}")

    if len(eval_idx) == 0:
        raise ValueError("No evaluation windows available for the selected scheme.")
    return eval_idx


def eval_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores, dtype=float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    fpr = ((y_pred == 1) & (y_true == 0)).sum() / max(1, (y_true == 0).sum())

    if np.unique(y_true).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, scores))
        if not np.isfinite(auroc):
            raise ValueError("AUROC is not finite despite both classes being present.")

    return float(precision), float(recall), float(f1), float(fpr), float(auroc)


def validate_notebook_alignment(regentad_defaults: Mapping[str, Dict[str, Any]]) -> None:
    for protocol in (PROTOCOLS[name] for name in ACTIVE_PROTOCOLS):
        nb_constants = extract_notebook_constants(protocol.notebook_path)

        protocol_mismatches = []
        if nb_constants.get("PAST_LEN") != PAST_LEN:
            protocol_mismatches.append(("PAST_LEN", nb_constants.get("PAST_LEN"), PAST_LEN))
        if nb_constants.get("HORIZON") != HORIZON:
            protocol_mismatches.append(("HORIZON", nb_constants.get("HORIZON"), HORIZON))
        if tuple(nb_constants.get("DIMENSIONS", ())) != protocol.dimensions:
            protocol_mismatches.append(("DIMENSIONS", nb_constants.get("DIMENSIONS"), protocol.dimensions))
        if tuple(tuple(x) for x in nb_constants.get("SAMPLE_SIZES", ())) != protocol.sample_sizes:
            protocol_mismatches.append(("SAMPLE_SIZES", nb_constants.get("SAMPLE_SIZES"), protocol.sample_sizes))
        if tuple(nb_constants.get("ANOMALIES", ())) != protocol.anomalies:
            protocol_mismatches.append(("ANOMALIES", nb_constants.get("ANOMALIES"), protocol.anomalies))
        if tuple(nb_constants.get("CONTAMINATION_RATES", ())) != protocol.contamination_rates:
            protocol_mismatches.append(
                ("CONTAMINATION_RATES", nb_constants.get("CONTAMINATION_RATES"), protocol.contamination_rates)
            )
        if nb_constants.get("N_ITER") != protocol.n_iter:
            protocol_mismatches.append(("N_ITER", nb_constants.get("N_ITER"), protocol.n_iter))

        if protocol_mismatches:
            mismatch_text = "; ".join(
                f"{name}: notebook={nb!r}, script={script!r}"
                for name, nb, script in protocol_mismatches
            )
            raise ValueError(f"{protocol.name} protocol no longer matches notebook constants: {mismatch_text}")

        helpers = load_notebook_helpers(protocol.notebook_path)

        for anomaly in protocol.anomalies:
            n_normal, n_shock = protocol.sample_sizes[0]
            dim = protocol.dimensions[0]
            seed = BASE_SEED
            nb_x, nb_y = helpers["generate_stock_like_data"](
                n_normal=n_normal,
                n_shock=n_shock,
                p=dim,
                anomaly_type=anomaly,
                seed=seed,
            )
            py_x, py_y = generate_synthetic_series(
                n_normal=n_normal,
                n_shock=n_shock,
                p=dim,
                anomaly_type=anomaly,
                seed=seed,
            )
            if not np.array_equal(nb_y, py_y) or not np.allclose(nb_x, py_x):
                raise ValueError(
                    f"{protocol.name} anomaly generator mismatch for anomaly={anomaly}, seed={seed}"
                )

        sample_x, sample_y = generate_synthetic_series(
            n_normal=protocol.sample_sizes[0][0],
            n_shock=protocol.sample_sizes[0][1],
            p=protocol.dimensions[0],
            anomaly_type=protocol.anomalies[0],
            seed=BASE_SEED + 1,
        )
        nb_xp, nb_yf, nb_yw = helpers["make_windows"](sample_x, sample_y, PAST_LEN, HORIZON)
        py_xp, py_yf, py_yw = make_windows(sample_x, sample_y, PAST_LEN, HORIZON)
        if not (np.allclose(nb_xp, py_xp) and np.allclose(nb_yf, py_yf) and np.array_equal(nb_yw, py_yw)):
            raise ValueError(f"{protocol.name} window construction mismatch")

        nb_rng = np.random.default_rng(BASE_SEED + 2)
        py_rng = np.random.default_rng(BASE_SEED + 2)
        nb_xc, nb_yc = helpers["contaminate_training_data"](nb_xp[:25], nb_yf[:25], 0.10, nb_rng)
        py_xc, py_yc = contaminate_training_data(py_xp[:25], py_yf[:25], 0.10, py_rng)
        if not (np.allclose(nb_xc, py_xc) and np.allclose(nb_yc, py_yc)):
            raise ValueError(f"{protocol.name} contamination function mismatch")

        shock_start = int(np.where(py_yw == 1)[0][0])
        dummy_xp = py_xp
        dummy_yf = py_yf
        for notebook_name, canonical in (
            ("whole set", "whole_set"),
            ("test set", "anomalous_segment"),
            ("test mixed", "mixed_test_segment"),
        ):
            _, _, nb_eval_idx = helpers["get_evaluation_data"](
                notebook_name,
                dummy_xp,
                dummy_yf,
                py_yw,
                shock_start,
            )
            py_eval_idx = get_evaluation_indices(canonical, py_yw, shock_start)
            if not np.array_equal(nb_eval_idx, py_eval_idx):
                raise ValueError(
                    f"{protocol.name} evaluation indexing mismatch for scheme={canonical}"
                )

        notebook_model_defaults = {
            "d_model": nb_constants.get("SHARED_D_MODEL"),
            "num_heads": nb_constants.get("SHARED_NUM_HEADS"),
            "ff_dim": nb_constants.get("SHARED_FF_DIM"),
            "dropout": nb_constants.get("SHARED_DROPOUT"),
        }
        overlapping = []
        for key, notebook_value in notebook_model_defaults.items():
            regentad_value = regentad_defaults["init"].get(key)
            if notebook_value != regentad_value:
                overlapping.append((key, notebook_value, regentad_value))

        if overlapping:
            print(
                f"[sanity] WARNING: notebook shared model defaults differ from ReGENTAD in {protocol.name}:"
            )
            for key, notebook_value, regentad_value in overlapping:
                print(f"  - {key}: notebook={notebook_value!r}, ReGENTAD={regentad_value!r}")
        else:
            print(f"[sanity] {protocol.name}: notebook helper functions match the script exactly.")


def default_architecture(regentad_defaults: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    init_defaults = regentad_defaults["init"]
    return {
        "d_model": int(init_defaults["d_model"]),
        "num_heads": int(init_defaults["num_heads"]),
        "ff_dim": int(init_defaults["ff_dim"]),
        "lstm_units": int(init_defaults["lstm_units"]),
        "lr": float(init_defaults["lr"]),
    }


def build_arch_configs(regentad_defaults: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    keys = list(ARCH_GRID.keys())
    for combo in itertools.product(*(ARCH_GRID[key] for key in keys)):
        cfg = dict(zip(keys, combo))
        cfg.update(regentad_defaults["weights"])
        cfg["stage"] = "arch"
        cfg["vary_param"] = ""
        configs.append(cfg)
    return configs


def build_weight_configs(
    stage2_arch: Mapping[str, Any],
    regentad_defaults: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for param, levels in WEIGHT_GRID.items():
        default_value = float(regentad_defaults["weights"][param])
        for value in levels:
            if abs(float(value) - default_value) < 1e-12:
                continue
            cfg = dict(stage2_arch)
            cfg.update(regentad_defaults["weights"])
            cfg[param] = float(value)
            cfg["stage"] = "weight"
            cfg["vary_param"] = param
            configs.append(cfg)
    return configs


def stage2_architecture(
    df_stage1: pd.DataFrame,
    regentad_defaults: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if STAGE2_ARCH_SOURCE == "baseline":
        return default_architecture(regentad_defaults)

    if df_stage1.empty:
        raise ValueError("Stage 1 produced no rows, so no best architecture can be selected.")

    summary = (
        df_stage1.groupby(["d_model", "num_heads", "ff_dim", "lstm_units", "lr"], as_index=False)
        .agg(
            AUROC=("AUROC", "mean"),
            F1=("F1", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            FPR=("FPR", "mean"),
            Runtime=("Runtime", "mean"),
        )
        .sort_values(
            by=["AUROC", "F1", "Precision", "Recall", "FPR", "Runtime"],
            ascending=[False, False, False, False, True, True],
        )
    )
    best = summary.iloc[0]
    return {
        "d_model": int(best["d_model"]),
        "num_heads": int(best["num_heads"]),
        "ff_dim": int(best["ff_dim"]),
        "lstm_units": int(best["lstm_units"]),
        "lr": float(best["lr"]),
    }


def iterate_scenarios(fast: bool) -> Iterable[Dict[str, Any]]:
    for protocol_name in ACTIVE_PROTOCOLS:
        protocol = PROTOCOLS[protocol_name]
        dimensions = protocol.dimensions[:1] if fast else protocol.dimensions
        sample_sizes = protocol.sample_sizes[:2] if fast else protocol.sample_sizes
        anomalies = protocol.anomalies[:2] if fast else protocol.anomalies
        contamination_rates = protocol.contamination_rates[:2] if fast else protocol.contamination_rates
        n_iter = min(protocol.n_iter, 2) if fast else protocol.n_iter

        for dim in dimensions:
            for n_normal, n_shock in sample_sizes:
                for anomaly in anomalies:
                    for contam_rate in contamination_rates:
                        for iteration in range(n_iter):
                            yield {
                                "protocol": protocol_name,
                                "dim": int(dim),
                                "n_normal": int(n_normal),
                                "n_shock": int(n_shock),
                                "anomaly": anomaly,
                                "contam_rate": float(contam_rate),
                                "iteration": int(iteration),
                                "seed": BASE_SEED + int(iteration),
                            }


def _weights_for_config(
    cfg: Mapping[str, Any],
    regentad_defaults: Mapping[str, Dict[str, Any]],
) -> Dict[str, float]:
    return {
        key: float(cfg.get(key, regentad_defaults["weights"][key]))
        for key in regentad_defaults["weights"]
    }


def _weights_override_or_none(
    cfg: Mapping[str, Any],
    regentad_defaults: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    weights = _weights_for_config(cfg, regentad_defaults)
    default_weights = regentad_defaults["weights"]
    if all(abs(weights[k] - float(default_weights[k])) < 1e-12 for k in default_weights):
        return None
    return weights


def _log_class_balance_once(
    protocol_name: str,
    eval_scheme: str,
    dim: int,
    n_normal: int,
    n_shock: int,
    y_true_eval: np.ndarray,
) -> None:
    key = (protocol_name, eval_scheme, dim, n_normal, n_shock)
    if key in _CLASS_BALANCE_LOGGED:
        return
    _CLASS_BALANCE_LOGGED.add(key)
    normal_count = int((y_true_eval == 0).sum())
    anomaly_count = int((y_true_eval == 1).sum())
    print(
        f"[sanity] class balance | protocol={protocol_name} | scheme={EVAL_DISPLAY[eval_scheme]} | "
        f"dim={dim} | samples=({n_normal},{n_shock}) | normal={normal_count} | anomaly={anomaly_count}"
    )


def run_one_configuration(
    cfg: Mapping[str, Any],
    scenario: Mapping[str, Any],
    regentad_defaults: Mapping[str, Dict[str, Any]],
    fast: bool,
) -> Optional[Dict[str, Any]]:
    protocol = PROTOCOLS[str(scenario["protocol"])]
    seed = int(scenario["seed"])
    rng = np.random.default_rng(seed)

    set_all_seeds(seed)
    tf.keras.backend.clear_session()
    gc.collect()

    x, y = generate_synthetic_series(
        n_normal=int(scenario["n_normal"]),
        n_shock=int(scenario["n_shock"]),
        p=int(scenario["dim"]),
        anomaly_type=str(scenario["anomaly"]),
        seed=seed,
    )
    xp, yf, y_eval = make_windows(x, y, PAST_LEN, HORIZON)

    shock_positions = np.where(y_eval == 1)[0]
    if len(shock_positions) == 0:
        return None

    shock_start = int(shock_positions[0])
    xp_tr = xp[:shock_start]
    yf_tr = yf[:shock_start]
    if len(xp_tr) < 20:
        return None

    xp_tr_contam, yf_tr_contam = contaminate_training_data(
        xp_tr,
        yf_tr,
        float(scenario["contam_rate"]),
        rng,
    )

    resolved_scheme = resolve_evaluation_scheme(EVALUATION_SCHEME, protocol)
    eval_idx = get_evaluation_indices(resolved_scheme, y_eval, shock_start)
    y_true_eval = y_eval[eval_idx]
    _log_class_balance_once(
        protocol.name,
        resolved_scheme,
        int(scenario["dim"]),
        int(scenario["n_normal"]),
        int(scenario["n_shock"]),
        y_true_eval,
    )

    detector_kwargs: Dict[str, Any] = {
        "past_len": PAST_LEN,
        "horizon": HORIZON,
        "n_features": int(scenario["dim"]),
        "d_model": int(cfg["d_model"]),
        "num_heads": int(cfg["num_heads"]),
        "ff_dim": int(cfg["ff_dim"]),
        "lstm_units": int(cfg["lstm_units"]),
        "lr": float(cfg["lr"]),
        "random_state": seed,
    }
    weights_override = _weights_override_or_none(cfg, regentad_defaults)
    if weights_override is not None:
        detector_kwargs["weights"] = weights_override

    detector = ReGENTAD(**detector_kwargs)

    fit_kwargs: Dict[str, Any] = {"verbose": 0}
    if fast:
        fit_kwargs.update({"epochs": 5, "purify_epochs": 3})

    started = time.perf_counter()
    detector.fit(xp_tr_contam, yf_tr_contam, **fit_kwargs)
    pred, scores, parts, meta = detector.predict(
        xp,
        yf,
        return_scores=True,
        return_parts=True,
        return_metadata=True,
    )
    runtime = time.perf_counter() - started

    precision, recall, f1, fpr, auroc = eval_metrics(
        y_true_eval,
        pred[eval_idx],
        scores[eval_idx],
    )

    weights = _weights_for_config(cfg, regentad_defaults)
    return {
        "stage": str(cfg["stage"]),
        "vary_param": str(cfg.get("vary_param", "")),
        "ScenarioFamily": protocol.name,
        "Anomaly": str(scenario["anomaly"]),
        "Iteration": int(scenario["iteration"]),
        "Seed": seed,
        "Dim": int(scenario["dim"]),
        "N_Normal": int(scenario["n_normal"]),
        "N_Shock": int(scenario["n_shock"]),
        "ContamRate": float(scenario["contam_rate"]),
        "EvaluationScheme": EVAL_DISPLAY[resolved_scheme],
        "ResolvedEvaluationScheme": resolved_scheme,
        "NotebookDefaultEvaluationScheme": EVAL_DISPLAY[protocol.notebook_default_eval],
        "EvalNormalWindows": int((y_true_eval == 0).sum()),
        "EvalAnomalyWindows": int((y_true_eval == 1).sum()),
        "d_model": int(cfg["d_model"]),
        "num_heads": int(cfg["num_heads"]),
        "ff_dim": int(cfg["ff_dim"]),
        "lstm_units": int(cfg["lstm_units"]),
        "lr": float(cfg["lr"]),
        "err": float(weights["err"]),
        "recon": float(weights["recon"]),
        "knn": float(weights["knn"]),
        "dyn": float(weights["dyn"]),
        "regime": float(weights["regime"]),
        "vol": float(weights["vol"]),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "AUROC": auroc,
        "Runtime": runtime,
        "DecisionRule": str(detector.decision_rule),
        "PredictMetaRule": str(meta.get("decision_rule", detector.decision_rule)) if isinstance(meta, dict) else "",
        "ScoreParts": ",".join(sorted(parts.keys())) if isinstance(parts, dict) else "",
    }


def checkpoint_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row["stage"]),
        str(row["ScenarioFamily"]),
        str(row["ResolvedEvaluationScheme"]),
        str(row["Anomaly"]),
        int(row["Iteration"]),
        int(row["Seed"]),
        int(row["Dim"]),
        int(row["N_Normal"]),
        int(row["N_Shock"]),
        round(float(row["ContamRate"]), 6),
        int(row["d_model"]),
        int(row["num_heads"]),
        int(row["ff_dim"]),
        int(row["lstm_units"]),
        round(float(row["lr"]), 10),
        round(float(row["err"]), 10),
        round(float(row["recon"]), 10),
        round(float(row["knn"]), 10),
        round(float(row["dyn"]), 10),
        round(float(row["regime"]), 10),
        round(float(row["vol"]), 10),
    )


def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_search(
    configs: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    regentad_defaults: Mapping[str, Dict[str, Any]],
    fast: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    completed: set[Tuple[Any, ...]] = set()

    if checkpoint_path.exists():
        try:
            ckpt = pd.read_csv(checkpoint_path)
            rows = ckpt.to_dict("records")
            completed = {checkpoint_key(row) for row in rows}
            print(f"[resume] loaded {len(rows)} rows from {checkpoint_path.name}")
        except Exception as exc:
            rows = []
            completed = set()
            print(f"[resume] could not read {checkpoint_path.name}: {exc}")

    scenarios = list(iterate_scenarios(fast=fast))
    total = len(configs) * len(scenarios)
    counter = 0
    start = time.perf_counter()

    for cfg_idx, cfg in enumerate(configs, start=1):
        for scenario in scenarios:
            counter += 1

            fake_weights = _weights_for_config(cfg, regentad_defaults)
            fake_row = {
                "stage": cfg["stage"],
                "ScenarioFamily": scenario["protocol"],
                "ResolvedEvaluationScheme": resolve_evaluation_scheme(EVALUATION_SCHEME, PROTOCOLS[scenario["protocol"]]),
                "Anomaly": scenario["anomaly"],
                "Iteration": scenario["iteration"],
                "Seed": scenario["seed"],
                "Dim": scenario["dim"],
                "N_Normal": scenario["n_normal"],
                "N_Shock": scenario["n_shock"],
                "ContamRate": scenario["contam_rate"],
                "d_model": cfg["d_model"],
                "num_heads": cfg["num_heads"],
                "ff_dim": cfg["ff_dim"],
                "lstm_units": cfg["lstm_units"],
                "lr": cfg["lr"],
                "err": fake_weights["err"],
                "recon": fake_weights["recon"],
                "knn": fake_weights["knn"],
                "dyn": fake_weights["dyn"],
                "regime": fake_weights["regime"],
                "vol": fake_weights["vol"],
            }
            if checkpoint_key(fake_row) in completed:
                continue

            elapsed = time.perf_counter() - start
            print(
                f"[run] {counter}/{total} | stage={cfg['stage']} | cfg={cfg_idx}/{len(configs)} | "
                f"protocol={scenario['protocol']} | anomaly={scenario['anomaly']} | "
                f"samples=({scenario['n_normal']},{scenario['n_shock']}) | "
                f"contam={scenario['contam_rate']:.2f} | iter={scenario['iteration']} | "
                f"elapsed={elapsed:.1f}s"
            )

            try:
                row = run_one_configuration(
                    cfg=cfg,
                    scenario=scenario,
                    regentad_defaults=regentad_defaults,
                    fast=fast,
                )
                if row is None:
                    continue
                rows.append(row)
                completed.add(checkpoint_key(row))
                save_dataframe_atomic(pd.DataFrame(rows), checkpoint_path)
            except Exception as exc:
                print(f"[run] ERROR: {exc}")

    return pd.DataFrame(rows)


def build_sensitivity_summary(
    df_stage1: pd.DataFrame,
    df_stage2: pd.DataFrame,
    regentad_defaults: Mapping[str, Dict[str, Any]],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    arch_labels = {
        "d_model": "d_model",
        "num_heads": "num_heads",
        "ff_dim": "ff_dim",
        "lstm_units": "lstm_units",
        "lr": "lr",
    }
    weight_labels = {
        "err": "err",
        "recon": "recon",
        "knn": "knn",
        "dyn": "dyn",
        "regime": "regime",
        "vol": "vol",
    }

    for param, component in arch_labels.items():
        for level in sorted(df_stage1[param].dropna().unique()):
            subset = df_stage1[df_stage1[param] == level]
            if subset.empty:
                continue
            row = {"Component": component, "Level": level}
            for metric in METRIC_COLS:
                row[metric] = float(subset[metric].mean())
            records.append(row)

    for param, component in weight_labels.items():
        default_level = float(regentad_defaults["weights"][param])
        for level in sorted(WEIGHT_GRID[param]):
            if abs(float(level) - default_level) < 1e-12:
                subset = df_stage1
            else:
                subset = df_stage2[
                    (df_stage2["vary_param"] == param)
                    & (df_stage2[param].round(10) == round(float(level), 10))
                ]
            if subset.empty:
                continue
            row = {"Component": component, "Level": level}
            for metric in METRIC_COLS:
                row[metric] = float(subset[metric].mean())
            records.append(row)

    summary = pd.DataFrame(records)
    if not summary.empty:
        summary = summary[
            ["Component", "Level", "Precision", "Recall", "F1", "FPR", "AUROC", "Runtime"]
        ]
    return summary


def build_top_configs_table(df_all: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df_all.groupby(CONFIG_GROUP_COLS, as_index=False)
        .agg(
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            FPR=("FPR", "mean"),
            AUROC=("AUROC", "mean"),
            Runtime=("Runtime", "mean"),
        )
        .sort_values(
            by=["AUROC", "F1", "Precision", "Recall", "FPR", "Runtime"],
            ascending=[False, False, False, False, True, True],
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    grouped.insert(0, "Rank", np.arange(1, len(grouped) + 1))
    return grouped.rename(
        columns={
            "d_model": "d",
            "num_heads": "Heads",
            "ff_dim": "FF",
            "lstm_units": "LSTM",
            "lr": "LR",
            "err": "Err",
            "recon": "Recon",
            "knn": "kNN",
            "dyn": "Dyn",
            "regime": "Regime",
            "vol": "Vol",
        }
    )[
        [
            "Rank",
            "d",
            "Heads",
            "FF",
            "LSTM",
            "LR",
            "Err",
            "Recon",
            "kNN",
            "Dyn",
            "Regime",
            "Vol",
            "F1",
            "AUROC",
            "Precision",
            "Recall",
            "FPR",
            "Runtime",
        ]
    ]


def print_detected_defaults(regentad_defaults: Mapping[str, Dict[str, Any]]) -> None:
    init_defaults = regentad_defaults["init"]
    fit_defaults = regentad_defaults["fit"]
    predict_defaults = regentad_defaults["predict"]

    print("[sanity] ReGENTAD defaults detected from source:")
    print(
        "  init: "
        f"d_model={init_defaults['d_model']}, num_heads={init_defaults['num_heads']}, "
        f"ff_dim={init_defaults['ff_dim']}, lstm_units={init_defaults['lstm_units']}, "
        f"dropout={init_defaults['dropout']}, lr={init_defaults['lr']}, "
        f"decision_rule={init_defaults['decision_rule']}"
    )
    print(
        "  fit: "
        f"validation_split={fit_defaults['validation_split']}, epochs={fit_defaults['epochs']}, "
        f"batch_size={fit_defaults['batch_size']}, purify={fit_defaults['purify']}, "
        f"purify_q={fit_defaults['purify_q']}, purify_max_remove={fit_defaults['purify_max_remove']}, "
        f"purify_epochs={fit_defaults['purify_epochs']}, purify_iters={fit_defaults['purify_iters']}"
    )
    print(
        "  predict: "
        f"alpha={predict_defaults['alpha']}, min_duration={predict_defaults['min_duration']}, "
        f"dilate={predict_defaults['dilate']}, window={predict_defaults['window']}, "
        f"robust={predict_defaults['robust']}, min_history={predict_defaults['min_history']}, "
        f"quantile_buffer={predict_defaults['quantile_buffer']}"
    )
    print(f"  weights: {regentad_defaults['weights']}")
    print(
        "[sanity] Simulation window settings:"
        f" past_len={PAST_LEN}, horizon={HORIZON}, active_protocols={list(ACTIVE_PROTOCOLS)}"
    )
    if normalize_evaluation_scheme(EVALUATION_SCHEME) == "notebook_default":
        print(
            "[sanity] Selected evaluation scheme: notebook_default "
            "(structural -> Anomalous Segment, stocks -> Mixed Test Segment)"
        )
    else:
        resolved = EVAL_DISPLAY[normalize_evaluation_scheme(EVALUATION_SCHEME)]
        print(f"[sanity] Selected evaluation scheme: {resolved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ReGENTAD hyperparameter sensitivity analysis")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke-test mode with smaller scenario subsets and shorter fit overrides.",
    )
    args = parser.parse_args()

    regentad_defaults = extract_regentad_defaults(PROJECT_ROOT / "ReGENTAD.py")
    print_detected_defaults(regentad_defaults)
    # validate_notebook_alignment(regentad_defaults)

    out_dir = results_dir()
    stage1_ckpt = out_dir / "checkpoint_stage1_arch.csv"
    stage2_ckpt = out_dir / "checkpoint_stage2_weights.csv"
    raw_stage1_path = out_dir / "raw_stage1_arch.csv"
    raw_stage2_path = out_dir / "raw_stage2_weights.csv"
    summary_path = out_dir / "sensitivity_summary.csv"
    top_path = out_dir / "top_configs_by_auroc.csv"

    stage1_configs = build_arch_configs(regentad_defaults)
    df_stage1 = run_search(
        configs=stage1_configs,
        checkpoint_path=stage1_ckpt,
        regentad_defaults=regentad_defaults,
        fast=args.fast,
    )
    save_dataframe_atomic(df_stage1, raw_stage1_path)

    chosen_arch = stage2_architecture(df_stage1, regentad_defaults)
    print(f"[stage2] architecture source={STAGE2_ARCH_SOURCE} | selected={chosen_arch}")

    stage2_configs = build_weight_configs(chosen_arch, regentad_defaults)
    df_stage2 = run_search(
        configs=stage2_configs,
        checkpoint_path=stage2_ckpt,
        regentad_defaults=regentad_defaults,
        fast=args.fast,
    )
    save_dataframe_atomic(df_stage2, raw_stage2_path)

    summary = build_sensitivity_summary(df_stage1, df_stage2, regentad_defaults)
    save_dataframe_atomic(summary, summary_path)

    df_all = pd.concat([df_stage1, df_stage2], ignore_index=True, sort=False)
    top_configs = build_top_configs_table(df_all)
    save_dataframe_atomic(top_configs, top_path)

    print(f"[done] wrote {raw_stage1_path.name}, {raw_stage2_path.name}, {summary_path.name}, {top_path.name}")


if __name__ == "__main__":
    main()
