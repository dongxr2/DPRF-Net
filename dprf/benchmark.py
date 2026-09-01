"""Fair leave-one-load-out benchmark for the expanded motor-fault study.

All models use the same fold, train-only standardization, random seeds and test
perturbations. Neural fusion models with modality dropout use mutually
exclusive categorical states, so the probability of both modalities missing
is exactly zero.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "locked.json"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "clean_reextract.csv.gz"
ALL_MODELS = [
    "vib_mlp",
    "elec_mlp",
    "lr",
    "svm_rbf",
    "rf",
    "xgboost",
    "early_mlp",
    "mean_md",
    "gate_md",
    "rcarf",
]
CASES = [
    "clean",
    "vibration_missing",
    "electrical_missing",
    "current_missing",
    "voltage_missing",
    "noise_20db",
    "noise_10db",
    "noise_5db",
    "noise_0db",
]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    neuron_dropout: float
    gate_loss_weight: float
    switch_loss_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS)
    parser.add_argument("--loads", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    metadata = {"label", "label_id", "load", "freq", "source_file", "window_id"}
    features = [column for column in df.columns if column not in metadata]
    vib = [
        column
        for column in features
        if column.startswith("Vibration") or column.startswith("VibVector")
    ]
    elec = [column for column in features if column not in vib]
    if len(vib) != 41 or len(elec) != 55:
        raise ValueError(f"Unexpected feature split: vibration={len(vib)}, electrical={len(elec)}")
    return vib, elec


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float, classes: int = 6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FusionNet(nn.Module):
    def __init__(
        self,
        vib_dim: int,
        elec_dim: int,
        hidden: int,
        dropout: float,
        mode: str,
    ):
        super().__init__()
        self.mode = mode
        self.vib = nn.Sequential(nn.Linear(vib_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.elec = nn.Sequential(nn.Linear(elec_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        if mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden * 4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )

    def encode(
        self, xv: torch.Tensor, xe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hv, he = self.vib(xv), self.elec(xe)
        if self.mode == "mean":
            return 0.5 * (hv + he), None
        z = torch.cat([hv, he, torch.abs(hv - he), hv * he], dim=1)
        gate = torch.softmax(self.gate(z), dim=1)
        return gate[:, :1] * hv + gate[:, 1:] * he, gate

    def forward(self, xv: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
        fused, _ = self.encode(xv, xe)
        return self.head(fused)


class RCARF(nn.Module):
    def __init__(self, vib_dim: int, elec_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.vib = nn.Sequential(nn.Linear(vib_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.elec = nn.Sequential(nn.Linear(elec_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        interaction_dim = hidden * 4
        self.reliability_gate = nn.Sequential(
            nn.Linear(interaction_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.switch_strength = nn.Sequential(
            nn.Linear(interaction_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )

    def encode(
        self, xv: torch.Tensor, xe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hv, he = self.vib(xv), self.elec(xe)
        z = torch.cat([hv, he, torch.abs(hv - he), hv * he], dim=1)
        gate = torch.softmax(self.reliability_gate(z), dim=1)
        adaptive = gate[:, :1] * hv + gate[:, 1:] * he
        stable = 0.5 * (hv + he)
        alpha = torch.sigmoid(self.switch_strength(z))
        return stable + alpha * (adaptive - stable), gate, alpha

    def forward(self, xv: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.encode(xv, xe)
        return self.head(fused)


def categorical_modality_mask(
    batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return keep masks and state: 0=complete, 1=vibration missing, 2=electrical missing."""
    draw = torch.rand(batch_size, 1, device=device)
    state = torch.zeros(batch_size, dtype=torch.long, device=device)
    state[(draw[:, 0] >= 0.50) & (draw[:, 0] < 0.75)] = 1
    state[draw[:, 0] >= 0.75] = 2
    keep_v = (state != 1).float().unsqueeze(1)
    keep_e = (state != 2).float().unsqueeze(1)
    return keep_v, keep_e, state


def make_neural_model(
    key: str, vib_dim: int, elec_dim: int, cfg: TrainConfig
) -> nn.Module:
    if key == "vib_mlp":
        return MLP(vib_dim, cfg.hidden_dim, cfg.neuron_dropout)
    if key == "elec_mlp":
        return MLP(elec_dim, cfg.hidden_dim, cfg.neuron_dropout)
    if key == "early_mlp":
        return MLP(vib_dim + elec_dim, cfg.hidden_dim, cfg.neuron_dropout)
    if key == "mean_md":
        return FusionNet(vib_dim, elec_dim, cfg.hidden_dim, cfg.neuron_dropout, "mean")
    if key == "gate_md":
        return FusionNet(vib_dim, elec_dim, cfg.hidden_dim, cfg.neuron_dropout, "gate")
    if key == "rcarf":
        return RCARF(vib_dim, elec_dim, cfg.hidden_dim, cfg.neuron_dropout)
    raise ValueError(key)


def train_neural(
    key: str,
    xv: np.ndarray,
    xe: np.ndarray,
    y: np.ndarray,
    seed: int,
    cfg: TrainConfig,
) -> nn.Module:
    set_seed(seed)
    model = make_neural_model(key, xv.shape[1], xe.shape[1], cfg)
    dataset = TensorDataset(
        torch.tensor(xv, dtype=torch.float32),
        torch.tensor(xe, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    ce = nn.CrossEntropyLoss()
    for _ in range(cfg.epochs):
        model.train()
        for xv_b, xe_b, y_b in loader:
            optimizer.zero_grad(set_to_none=True)
            if key == "vib_mlp":
                logits = model(xv_b)
                loss = ce(logits, y_b)
            elif key == "elec_mlp":
                logits = model(xe_b)
                loss = ce(logits, y_b)
            elif key == "early_mlp":
                logits = model(torch.cat([xv_b, xe_b], dim=1))
                loss = ce(logits, y_b)
            else:
                keep_v, keep_e, state = categorical_modality_mask(len(y_b), xv_b.device)
                xv_masked, xe_masked = xv_b * keep_v, xe_b * keep_e
                if key == "rcarf":
                    fused, gate, alpha = model.encode(xv_masked, xe_masked)
                    loss = ce(model.head(fused), y_b)
                    missing = state != 0
                    if bool(missing.any()):
                        gate_target = (state[missing] == 1).long()
                        gate_loss = nn.functional.nll_loss(
                            torch.log(gate[missing].clamp_min(1e-8)), gate_target
                        )
                        loss = loss + cfg.gate_loss_weight * gate_loss
                    switch_target = missing.float().unsqueeze(1)
                    loss = loss + cfg.switch_loss_weight * nn.functional.binary_cross_entropy(
                        alpha, switch_target
                    )
                else:
                    loss = ce(model(xv_masked, xe_masked), y_b)
            loss.backward()
            optimizer.step()
    return model.eval()


def make_classical(key: str, seed: int, smoke: bool):
    if key == "lr":
        return LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=1
        )
    if key == "svm_rbf":
        return SVC(C=10.0, gamma="scale", class_weight="balanced")
    if key == "rf":
        return RandomForestClassifier(
            n_estimators=30 if smoke else 300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if key == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is unavailable")
        return XGBClassifier(
            n_estimators=20 if smoke else 300,
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


def perturb(
    xv: np.ndarray,
    xe: np.ndarray,
    case: str,
    current_indices: np.ndarray,
    voltage_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    xv_out, xe_out = xv.copy(), xe.copy()
    if case == "clean":
        return xv_out, xe_out
    if case == "vibration_missing":
        xv_out.fill(0.0)
        return xv_out, xe_out
    if case == "electrical_missing":
        xe_out.fill(0.0)
        return xv_out, xe_out
    if case == "current_missing":
        xe_out[:, current_indices] = 0.0
        return xv_out, xe_out
    if case == "voltage_missing":
        xe_out[:, voltage_indices] = 0.0
        return xv_out, xe_out
    if case.startswith("noise_") and case.endswith("db"):
        snr_db = float(case.removeprefix("noise_").removesuffix("db"))
        rng = np.random.default_rng(seed)
        for array in (xv_out, xe_out):
            signal_power = max(float(np.mean(array**2)), 1e-12)
            noise_sd = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
            array += rng.normal(0.0, noise_sd, size=array.shape)
        return xv_out, xe_out
    raise ValueError(case)


def predict_neural(
    key: str, model: nn.Module, xv: np.ndarray, xe: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    xv_t = torch.tensor(xv, dtype=torch.float32)
    xe_t = torch.tensor(xe, dtype=torch.float32)
    diagnostics: dict[str, float] = {}
    with torch.inference_mode():
        if key == "vib_mlp":
            logits = model(xv_t)
        elif key == "elec_mlp":
            logits = model(xe_t)
        elif key == "early_mlp":
            logits = model(torch.cat([xv_t, xe_t], dim=1))
        elif key == "rcarf":
            fused, gate, alpha = model.encode(xv_t, xe_t)
            logits = model.head(fused)
            diagnostics = {
                "vibration_weight_mean": float(gate[:, 0].mean()),
                "electrical_weight_mean": float(gate[:, 1].mean()),
                "switch_strength_mean": float(alpha.mean()),
            }
        elif key == "gate_md":
            fused, gate = model.encode(xv_t, xe_t)
            logits = model.head(fused)
            diagnostics = {
                "vibration_weight_mean": float(gate[:, 0].mean()),
                "electrical_weight_mean": float(gate[:, 1].mean()),
            }
        else:
            logits = model(xv_t, xe_t)
    return logits.argmax(1).cpu().numpy(), diagnostics


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def complexity(key: str, model) -> int:
    if isinstance(model, nn.Module):
        return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    if key == "lr":
        return int(model.coef_.size + model.intercept_.size)
    if key == "svm_rbf":
        return int(model.support_vectors_.size + model.dual_coef_.size)
    if key == "rf":
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    if key == "xgboost":
        return int(model.get_booster().num_boosted_rounds())
    return 0


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, case), group in long_df.groupby(["model", "case"], sort=False):
        values = group["macro_f1"].to_numpy()
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(
            {
                "model": model,
                "case": case,
                "n_fold_seed": len(values),
                "macro_f1_mean": mean,
                "macro_f1_sd": sd,
                "macro_f1_ci95_low": max(0.0, mean - half),
                "macro_f1_ci95_high": min(1.0, mean + half),
                "accuracy_mean": float(group["accuracy"].mean()),
                "weighted_f1_mean": float(group["weighted_f1"].mean()),
                "train_time_s_mean": float(group["train_time_s"].mean()),
                "complexity_mean": float(group["complexity"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
    cfg = TrainConfig(
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
        ("smoke_" if args.smoke else "full_") + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = ROOT / "results" / run_name
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out_dir}")
    out_dir.mkdir(parents=True)

    df = pd.read_csv(args.feature_csv)
    vib_cols, elec_cols = split_feature_columns(df)
    all_labels = sorted(df["label_id"].unique().tolist())
    if all_labels != list(range(6)):
        raise ValueError(f"Expected label IDs 0..5, got {all_labels}")
    current_indices = np.array(
        [
            i
            for i, name in enumerate(elec_cols)
            if name.startswith("Current") or name in {"Voltage_rms_to_current_rms"}
        ],
        dtype=int,
    )
    voltage_indices = np.array(
        [
            i
            for i, name in enumerate(elec_cols)
            if name.startswith("Voltage") or name in {"Voltage_rms_to_current_rms"}
        ],
        dtype=int,
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    confusion: dict[str, list[list[int]]] = {}
    started_all = time.perf_counter()
    neural_keys = {"vib_mlp", "elec_mlp", "early_mlp", "mean_md", "gate_md", "rcarf"}

    for test_load in loads:
        train = df[df["load"].astype(int) != int(test_load)].copy()
        test = df[df["load"].astype(int) == int(test_load)].copy()
        if test.empty:
            raise ValueError(f"No samples for test load {test_load}")
        scaler_v = StandardScaler().fit(train[vib_cols])
        scaler_e = StandardScaler().fit(train[elec_cols])
        xv_train = scaler_v.transform(train[vib_cols]).astype(np.float32)
        xe_train = scaler_e.transform(train[elec_cols]).astype(np.float32)
        xv_test = scaler_v.transform(test[vib_cols]).astype(np.float32)
        xe_test = scaler_e.transform(test[elec_cols]).astype(np.float32)
        y_train = train["label_id"].to_numpy(dtype=np.int64)
        y_test = test["label_id"].to_numpy(dtype=np.int64)

        for seed in seeds:
            for key in models:
                print(
                    f"[fold load={test_load}] [seed={seed}] [model={key}] training",
                    flush=True,
                )
                started = time.perf_counter()
                if key in neural_keys:
                    model = train_neural(key, xv_train, xe_train, y_train, seed, cfg)
                else:
                    model = make_classical(key, seed, args.smoke)
                    model.fit(np.concatenate([xv_train, xe_train], axis=1), y_train)
                train_time = time.perf_counter() - started
                model_complexity = complexity(key, model)

                for case in CASES:
                    xv_case, xe_case = perturb(
                        xv_test,
                        xe_test,
                        case,
                        current_indices,
                        voltage_indices,
                        seed=100_000 + int(test_load),
                    )
                    if key in neural_keys:
                        prediction, diag = predict_neural(key, model, xv_case, xe_case)
                    else:
                        prediction = model.predict(np.concatenate([xv_case, xe_case], axis=1))
                        diag = {}
                    row = metric_row(y_test, prediction)
                    row.update(
                        {
                            "test_load": int(test_load),
                            "seed": int(seed),
                            "model": key,
                            "case": case,
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
                                "model": key,
                                "case": case,
                                **diag,
                            }
                        )
                    cm_key = f"{test_load}|{seed}|{key}|{case}"
                    confusion[cm_key] = confusion_matrix(
                        y_test, prediction, labels=list(range(6))
                    ).tolist()

                del model

        pd.DataFrame(rows).to_csv(out_dir / "metrics_long.partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    summarize(metrics).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "fusion_diagnostics.csv", index=False)
    with (out_dir / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion, handle, ensure_ascii=False)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "feature_csv": str(args.feature_csv.resolve()),
        "config_file": str(args.config.resolve()),
        "models": models,
        "loads": loads,
        "seeds": seeds,
        "train_config": asdict(cfg),
        "cases": CASES,
        "categorical_modality_states": {
            "complete": 0.50,
            "vibration_missing": 0.25,
            "electrical_missing": 0.25,
            "both_missing": 0.0,
        },
        "current_feature_indices": current_indices.tolist(),
        "voltage_feature_indices": voltage_indices.tolist(),
        "elapsed_s": time.perf_counter() - started_all,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": None if XGBClassifier is None else __import__("xgboost").__version__,
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    partial = out_dir / "metrics_long.partial.csv"
    if partial.exists():
        partial.unlink()
    print(json.dumps({"output": str(out_dir), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
