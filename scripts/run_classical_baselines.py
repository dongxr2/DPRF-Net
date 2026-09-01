"""Run frozen traditional-ML baselines on the DPRF 15-condition benchmark."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dprf import benchmark as base  # noqa: E402
from dprf import data_protocol as signal_base  # noqa: E402


FEATURE_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE = FEATURE_DIR / "clean_reextract.csv.gz"
OUTPUT = PROJECT_ROOT / "results" / "runs" / "classical_reproduction"
LOADS = [0, 111, 222, 333, 444, 555]
SEEDS = [301, 302, 303, 304, 305]
STATE_PROBABILITIES = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
REPEATS = 2
CONDITIONS = [
    "clean_reextract",
    "vibration_missing",
    "electrical_missing",
    "vibration_gain_075",
    "vibration_gain_050",
    "vibration_gain_025",
    "electrical_gain_075",
    "electrical_gain_050",
    "electrical_gain_025",
    "vibration_drift_025rms",
    "vibration_drift_050rms",
    "vibration_drift_100rms",
    "electrical_drift_025rms",
    "electrical_drift_050rms",
    "electrical_drift_100rms",
]
TRAIN_STATE_NAMES = [
    "clean_reextract",
    "vibration_missing",
    "electrical_missing",
    "vibration_gain_050",
    "electrical_gain_050",
    "vibration_drift_050rms",
    "electrical_drift_050rms",
]
MODEL_KEYS = ["lr", "svm_rbf", "knn", "gnb", "lda", "cart", "rf", "extra_trees", "hist_gb", "xgboost"]


def make_model(key: str, seed: int):
    if key == "lr":
        return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)
    if key == "svm_rbf":
        return SVC(C=10.0, gamma="scale", class_weight="balanced", cache_size=2048)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance", p=2, n_jobs=1)
    if key == "gnb":
        return GaussianNB(var_smoothing=1e-9)
    if key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if key == "cart":
        return DecisionTreeClassifier(min_samples_leaf=2, class_weight="balanced", random_state=seed)
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if key == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=seed,
        )
    if key == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is unavailable")
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
    raise ValueError(key)


def align(table: pd.DataFrame, keys: pd.DataFrame, test_load: int, train: bool) -> pd.DataFrame:
    mask = table["load"].astype(int) != int(test_load)
    selected = table[mask].copy() if train else table[~mask].copy()
    return keys.merge(selected, on=signal_base.KEYS, how="left", validate="one_to_one")


def make_missing(clean_v: np.ndarray, clean_e: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "vibration_missing": (np.zeros_like(clean_v), clean_e.copy()),
        "electrical_missing": (clean_v.copy(), np.zeros_like(clean_e)),
    }


def fixed_mixture(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_parts, y_parts = [], []
    n = len(y)
    for _ in range(REPEATS):
        states = rng.choice(len(TRAIN_STATE_NAMES), size=n, p=STATE_PROBABILITIES)
        vib = np.empty_like(arrays["clean_reextract"][0])
        elec = np.empty_like(arrays["clean_reextract"][1])
        for state_id, condition in enumerate(TRAIN_STATE_NAMES):
            selected = states == state_id
            source_v, source_e = arrays[condition]
            vib[selected] = source_v[selected]
            elec[selected] = source_e[selected]
        x_parts.append(np.concatenate([vib, elec], axis=1))
        y_parts.append(y.copy())
    return np.concatenate(x_parts), np.concatenate(y_parts)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    clean = pd.read_csv(REFERENCE)
    all_tables = signal_base.load_condition_tables(FEATURE_DIR, clean)
    vibration_columns, electrical_columns = base.split_feature_columns(clean)
    feature_columns = vibration_columns + electrical_columns
    rows = []
    started_all = time.perf_counter()

    for test_load in LOADS:
        train_clean = clean[clean["load"].astype(int) != int(test_load)].copy()
        test_clean = clean[clean["load"].astype(int) == int(test_load)].copy()
        train_keys = train_clean[signal_base.KEYS].reset_index(drop=True)
        test_keys = test_clean[signal_base.KEYS].reset_index(drop=True)
        scaler_v = StandardScaler().fit(train_clean[vibration_columns])
        scaler_e = StandardScaler().fit(train_clean[electrical_columns])

        train_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        needed = ["clean_reextract", "vibration_gain_050", "electrical_gain_050", "vibration_drift_050rms", "electrical_drift_050rms"]
        for condition in needed:
            frame = align(all_tables[condition], train_keys, test_load, True)
            if frame[feature_columns].isna().any().any():
                raise ValueError(f"Training alignment failure: load={test_load}, condition={condition}")
            train_arrays[condition] = (
                scaler_v.transform(frame[vibration_columns]).astype(np.float32),
                scaler_e.transform(frame[electrical_columns]).astype(np.float32),
            )
        train_arrays.update(make_missing(*train_arrays["clean_reextract"]))

        test_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for condition in CONDITIONS:
            if condition in {"vibration_missing", "electrical_missing"}:
                continue
            frame = align(all_tables[condition], test_keys, test_load, False)
            if frame[feature_columns].isna().any().any():
                raise ValueError(f"Test alignment failure: load={test_load}, condition={condition}")
            test_arrays[condition] = (
                scaler_v.transform(frame[vibration_columns]).astype(np.float32),
                scaler_e.transform(frame[electrical_columns]).astype(np.float32),
            )
        test_arrays.update(make_missing(*test_arrays["clean_reextract"]))
        y_train = train_clean["label_id"].to_numpy(dtype=np.int64)
        y_test = test_clean["label_id"].to_numpy(dtype=np.int64)

        for seed in SEEDS:
            x_train, y_aug = fixed_mixture(train_arrays, y_train, seed)
            for model_key in MODEL_KEYS:
                print(f"[load={test_load:03d}] [seed={seed}] [model={model_key}]", flush=True)
                model = make_model(model_key, seed)
                started = time.perf_counter()
                model.fit(x_train, y_aug)
                fit_s = time.perf_counter() - started
                for condition in CONDITIONS:
                    xv, xe = test_arrays[condition]
                    prediction = model.predict(np.concatenate([xv, xe], axis=1))
                    row = metric_row(y_test, prediction)
                    row.update(
                        {
                            "test_load": test_load,
                            "seed": seed,
                            "model": model_key,
                            "condition": condition,
                            "fit_time_s": fit_s,
                        }
                    )
                    rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_long.csv", index=False, encoding="utf-8-sig")
    degraded = metrics[metrics["condition"] != "clean_reextract"].copy()
    degraded["family"] = np.select(
        [
            degraded["condition"].str.contains("missing"),
            degraded["condition"].str.contains("gain"),
            degraded["condition"].str.contains("drift"),
        ],
        ["missing", "gain", "drift"],
        default="unknown",
    )
    summary = degraded.groupby("model", as_index=False).agg(
        all_degraded_macro_f1=("macro_f1", "mean"),
        worst_condition_mean=("macro_f1", lambda s: np.nan),
    )
    clean_summary = metrics[metrics["condition"] == "clean_reextract"].groupby("model")["macro_f1"].mean()
    condition_means = degraded.groupby(["model", "condition"])["macro_f1"].mean()
    family = degraded.groupby(["model", "family"])["macro_f1"].mean().unstack()
    summary["clean_macro_f1"] = summary["model"].map(clean_summary)
    summary["worst_condition_mean"] = summary["model"].map(condition_means.groupby("model").min())
    summary = summary.merge(family, left_on="model", right_index=True, how="left")
    summary = summary.sort_values("all_degraded_macro_f1", ascending=False)
    summary.to_csv(OUTPUT / "model_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": str(PROJECT_ROOT / "docs" / "CLASSICAL_BASELINE_PROTOCOL.md"),
        "loads": LOADS,
        "seeds": SEEDS,
        "models": MODEL_KEYS,
        "conditions": CONDITIONS,
        "state_probabilities": STATE_PROBABILITIES.tolist(),
        "augmentation_repeats": REPEATS,
        "rows": len(metrics),
        "elapsed_s": time.perf_counter() - started_all,
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
