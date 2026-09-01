"""Evaluate all benchmark models on features re-extracted from perturbed signals."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from dprf import benchmark as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = ROOT / "data" / "processed"
DEFAULT_REFERENCE = DEFAULT_FEATURE_DIR / "clean_reextract.csv.gz"
DEFAULT_CONFIG = ROOT / "configs" / "locked.json"
KEYS = ["source_file", "window_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--models", nargs="+", choices=base.ALL_MODELS)
    parser.add_argument("--loads", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_condition_tables(
    feature_dir: Path, reference: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    tables: dict[str, pd.DataFrame] = {}
    expected_keys = reference[KEYS].sort_values(KEYS).reset_index(drop=True)
    for condition in manifest["conditions"]:
        csv_path = feature_dir / f"{condition}.csv"
        gzip_path = feature_dir / f"{condition}.csv.gz"
        table_path = gzip_path if gzip_path.exists() else csv_path
        table = pd.read_csv(table_path)
        observed_keys = table[KEYS].sort_values(KEYS).reset_index(drop=True)
        if not expected_keys.equals(observed_keys):
            raise ValueError(f"Sample keys do not match reference for {condition}")
        tables[condition] = table
    return tables


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["model", "condition"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
            train_time_s_mean=("train_time_s", "mean"),
            n_fold_seed=("macro_f1", "size"),
        )
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with args.config.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    models = args.models or raw_config["models"]
    loads = args.loads or raw_config["loads"]
    seeds = args.seeds or raw_config["seeds"]
    epochs = args.epochs or raw_config["epochs"]
    if args.smoke:
        loads, seeds, epochs = [555], [41], min(epochs, 2)
    config = base.TrainConfig(
        epochs=epochs,
        batch_size=raw_config["batch_size"],
        learning_rate=raw_config["learning_rate"],
        weight_decay=raw_config["weight_decay"],
        hidden_dim=raw_config["hidden_dim"],
        neuron_dropout=raw_config["neuron_dropout"],
        gate_loss_weight=raw_config["gate_loss_weight"],
        switch_loss_weight=raw_config["switch_loss_weight"],
    )
    run_name = args.run_name or (
        ("signal_smoke_" if args.smoke else "signal_full_")
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = ROOT / "results" / run_name
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    clean = pd.read_csv(args.reference_csv)
    condition_tables = load_condition_tables(args.feature_dir, clean)
    conditions = list(condition_tables)
    vibration_columns, electrical_columns = base.split_feature_columns(clean)
    neural_models = {"vib_mlp", "elec_mlp", "early_mlp", "mean_md", "gate_md", "rcarf"}
    rows: list[dict] = []
    diagnostics: list[dict] = []
    confusion: dict[str, list[list[int]]] = {}
    started_all = time.perf_counter()

    for test_load in loads:
        train = clean[clean["load"].astype(int) != int(test_load)].copy()
        clean_test = clean[clean["load"].astype(int) == int(test_load)].copy()
        scaler_v = StandardScaler().fit(train[vibration_columns])
        scaler_e = StandardScaler().fit(train[electrical_columns])
        xv_train = scaler_v.transform(train[vibration_columns]).astype(np.float32)
        xe_train = scaler_e.transform(train[electrical_columns]).astype(np.float32)
        y_train = train["label_id"].to_numpy(dtype=np.int64)
        y_test = clean_test["label_id"].to_numpy(dtype=np.int64)
        test_keys = clean_test[KEYS].reset_index(drop=True)
        prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for condition, table in condition_tables.items():
            condition_test = table[table["load"].astype(int) == int(test_load)].copy()
            condition_test = test_keys.merge(condition_test, on=KEYS, how="left", validate="one_to_one")
            if condition_test[vibration_columns + electrical_columns].isna().any().any():
                raise ValueError(f"Missing aligned samples: load={test_load}, condition={condition}")
            prepared[condition] = (
                scaler_v.transform(condition_test[vibration_columns]).astype(np.float32),
                scaler_e.transform(condition_test[electrical_columns]).astype(np.float32),
            )

        for seed in seeds:
            for model_key in models:
                print(
                    f"[load={test_load}] [seed={seed}] [model={model_key}] training",
                    flush=True,
                )
                started = time.perf_counter()
                if model_key in neural_models:
                    model = base.train_neural(
                        model_key, xv_train, xe_train, y_train, seed, config
                    )
                else:
                    model = base.make_classical(model_key, seed, args.smoke)
                    model.fit(np.concatenate([xv_train, xe_train], axis=1), y_train)
                train_time = time.perf_counter() - started
                model_complexity = base.complexity(model_key, model)

                for condition in conditions:
                    xv_test, xe_test = prepared[condition]
                    if model_key in neural_models:
                        prediction, diag = base.predict_neural(
                            model_key, model, xv_test, xe_test
                        )
                    else:
                        prediction = model.predict(
                            np.concatenate([xv_test, xe_test], axis=1)
                        )
                        diag = {}
                    row = base.metric_row(y_test, prediction)
                    row.update(
                        {
                            "test_load": int(test_load),
                            "seed": int(seed),
                            "model": model_key,
                            "condition": condition,
                            "n_test": len(y_test),
                            "train_time_s": train_time,
                            "complexity": model_complexity,
                        }
                    )
                    rows.append(row)
                    if diag:
                        diagnostics.append(
                            {
                                "test_load": int(test_load),
                                "seed": int(seed),
                                "model": model_key,
                                "condition": condition,
                                **diag,
                            }
                        )
                    key = f"{test_load}|{seed}|{model_key}|{condition}"
                    confusion[key] = confusion_matrix(
                        y_test, prediction, labels=list(range(6))
                    ).tolist()
                del model
        pd.DataFrame(rows).to_csv(out_dir / "metrics_long.partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    summarize(metrics).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "fusion_diagnostics.csv", index=False)
    (out_dir / "confusion_matrices.json").write_text(
        json.dumps(confusion, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_dir": str(args.feature_dir.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
        "models": models,
        "loads": loads,
        "seeds": seeds,
        "conditions": conditions,
        "train_config": asdict(config),
        "elapsed_s": time.perf_counter() - started_all,
        "rows": len(metrics),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    partial = out_dir / "metrics_long.partial.csv"
    if partial.exists():
        partial.unlink()
    print(json.dumps({"output": str(out_dir), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
