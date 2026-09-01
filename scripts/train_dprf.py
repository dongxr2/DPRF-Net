"""Run the locked DPRF-Net experiment under 15 paired input conditions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dprf import benchmark as base  # noqa: E402
from dprf import data_protocol as signal_base  # noqa: E402


TRAIN_CONDITIONS = {
    3: "vibration_gain_050",
    4: "electrical_gain_050",
    5: "vibration_drift_050rms",
    6: "electrical_drift_050rms",
}
PRIMARY_CONDITIONS = [
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
MODEL_KEYS = [
    "mean_da",
    "gate_da",
    "rcarf_hard",
    "dprf_no_rel_loss",
    "dprf_no_switch_loss",
    "dprf_no_semantic",
    "dprf_full",
]


@dataclass(frozen=True)
class Config:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    reliability_loss_weight: float
    switch_loss_weight: float
    state_probabilities: tuple[float, ...]


class DecoupledReliabilityFusion(nn.Module):
    def __init__(
        self,
        vib_dim: int,
        elec_dim: int,
        hidden: int,
        dropout: float,
        use_semantic: bool = True,
    ):
        super().__init__()
        self.use_semantic = use_semantic
        self.vib = nn.Sequential(
            nn.Linear(vib_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.elec = nn.Sequential(
            nn.Linear(elec_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        interaction_dim = hidden * 4
        self.semantic_gate = nn.Sequential(
            nn.Linear(interaction_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(interaction_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        self.switch_head = nn.Sequential(
            nn.Linear(interaction_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6)
        )

    def encode(self, xv: torch.Tensor, xe: torch.Tensor):
        hv, he = self.vib(xv), self.elec(xe)
        z = torch.cat([hv, he, torch.abs(hv - he), hv * he], dim=1)
        if self.use_semantic:
            semantic = torch.softmax(self.semantic_gate(z), dim=1)
        else:
            semantic = torch.full(
                (len(z), 2), 0.5, dtype=z.dtype, device=z.device
            )
        reliability = torch.sigmoid(self.reliability_head(z))
        calibrated = semantic * reliability.clamp_min(1e-6)
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(1e-6)
        adaptive = calibrated[:, :1] * hv + calibrated[:, 1:] * he
        stable = 0.5 * (hv + he)
        alpha = torch.sigmoid(self.switch_head(z))
        fused = stable + alpha * (adaptive - stable)
        return fused, semantic, reliability, calibrated, alpha

    def forward(self, xv: torch.Tensor, xe: torch.Tensor):
        fused, *_ = self.encode(xv, xe)
        return self.head(fused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "locked.json")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "clean_reextract.csv.gz",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--loads", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def aligned_subset(
    table: pd.DataFrame, keys: pd.DataFrame, test_load: int, train: bool
) -> pd.DataFrame:
    mask = table["load"].astype(int) != int(test_load)
    selected = table[mask].copy() if train else table[~mask].copy()
    return keys.merge(selected, on=signal_base.KEYS, how="left", validate="one_to_one")


def sample_states(batch_size: int, probabilities: tuple[float, ...]) -> torch.Tensor:
    probs = torch.tensor(probabilities, dtype=torch.float32)
    return torch.multinomial(probs, batch_size, replacement=True)


def soft_targets(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quality = torch.ones((len(state), 2), dtype=torch.float32)
    quality[state == 1, 0] = 0.0
    quality[state == 2, 1] = 0.0
    quality[(state == 3) | (state == 5), 0] = 0.5
    quality[(state == 4) | (state == 6), 1] = 0.5
    switch = (1.0 - quality.min(dim=1).values).unsqueeze(1)
    return quality, switch


def make_model(key: str, vib_dim: int, elec_dim: int, cfg: Config):
    if key == "mean_da":
        return base.FusionNet(vib_dim, elec_dim, cfg.hidden_dim, cfg.dropout, "mean")
    if key == "gate_da":
        return base.FusionNet(vib_dim, elec_dim, cfg.hidden_dim, cfg.dropout, "gate")
    if key == "rcarf_hard":
        return base.RCARF(vib_dim, elec_dim, cfg.hidden_dim, cfg.dropout)
    if key.startswith("dprf_"):
        return DecoupledReliabilityFusion(
            vib_dim,
            elec_dim,
            cfg.hidden_dim,
            cfg.dropout,
            use_semantic=(key != "dprf_no_semantic"),
        )
    raise ValueError(key)


def train_model(
    key: str,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    seed: int,
    cfg: Config,
):
    base.set_seed(seed)
    tensors = {
        condition: (
            torch.tensor(vibration, dtype=torch.float32),
            torch.tensor(electrical, dtype=torch.float32),
        )
        for condition, (vibration, electrical) in arrays.items()
    }
    clean_v, clean_e = arrays["clean_reextract"]
    model = make_model(key, clean_v.shape[1], clean_e.shape[1], cfg)
    dataset = TensorDataset(
        torch.arange(len(y), dtype=torch.long),
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
    bce = nn.BCELoss()
    for _ in range(cfg.epochs):
        model.train()
        for sample_indices, labels in loader:
            state = sample_states(len(sample_indices), cfg.state_probabilities)
            xv = tensors["clean_reextract"][0][sample_indices].clone()
            xe = tensors["clean_reextract"][1][sample_indices].clone()
            xv[state == 1] = 0.0
            xe[state == 2] = 0.0
            for state_id, condition in TRAIN_CONDITIONS.items():
                selected = state == state_id
                if bool(selected.any()):
                    xv[selected] = tensors[condition][0][sample_indices[selected]]
                    xe[selected] = tensors[condition][1][sample_indices[selected]]
            quality_target, switch_target = soft_targets(state)
            optimizer.zero_grad(set_to_none=True)
            if key.startswith("dprf_"):
                fused, _, reliability, _, alpha = model.encode(xv, xe)
                loss = ce(model.head(fused), labels)
                if key != "dprf_no_rel_loss":
                    loss = loss + cfg.reliability_loss_weight * bce(
                        reliability, quality_target
                    )
                if key != "dprf_no_switch_loss":
                    loss = loss + cfg.switch_loss_weight * bce(alpha, switch_target)
            elif key == "rcarf_hard":
                fused, gate, alpha = model.encode(xv, xe)
                loss = ce(model.head(fused), labels)
                degraded = state != 0
                target_electrical = torch.isin(
                    state[degraded], torch.tensor([1, 3, 5])
                ).long()
                loss = loss + cfg.reliability_loss_weight * nn.functional.nll_loss(
                    torch.log(gate[degraded].clamp_min(1e-8)), target_electrical
                )
                loss = loss + cfg.switch_loss_weight * bce(
                    alpha, degraded.float().unsqueeze(1)
                )
            else:
                loss = ce(model(xv, xe), labels)
            loss.backward()
            optimizer.step()
    return model.eval()


def quality_target_for_condition(condition: str) -> tuple[float, float]:
    if condition == "clean_reextract":
        return 1.0, 1.0
    if condition == "vibration_missing":
        return 0.0, 1.0
    if condition == "electrical_missing":
        return 1.0, 0.0
    if condition.startswith("vibration_gain_"):
        return float(condition.rsplit("_", 1)[1]) / 100.0, 1.0
    if condition.startswith("electrical_gain_"):
        return 1.0, float(condition.rsplit("_", 1)[1]) / 100.0
    drift_quality = {
        "025rms": 0.75,
        "050rms": 0.50,
        "100rms": 0.00,
    }
    if condition.startswith("vibration_drift_"):
        return drift_quality[condition.rsplit("_", 1)[1]], 1.0
    if condition.startswith("electrical_drift_"):
        return 1.0, drift_quality[condition.rsplit("_", 1)[1]]
    raise ValueError(condition)


def predict(key: str, model, xv: np.ndarray, xe: np.ndarray):
    xv_t = torch.tensor(xv, dtype=torch.float32)
    xe_t = torch.tensor(xe, dtype=torch.float32)
    diagnostics = {}
    with torch.inference_mode():
        if key == "mean_da":
            logits = model(xv_t, xe_t)
        elif key == "gate_da":
            fused, gate = model.encode(xv_t, xe_t)
            logits = model.head(fused)
            diagnostics = {
                "semantic_vibration_mean": float(gate[:, 0].mean()),
                "semantic_electrical_mean": float(gate[:, 1].mean()),
            }
        elif key == "rcarf_hard":
            fused, gate, alpha = model.encode(xv_t, xe_t)
            logits = model.head(fused)
            diagnostics = {
                "calibrated_vibration_mean": float(gate[:, 0].mean()),
                "calibrated_electrical_mean": float(gate[:, 1].mean()),
                "switch_mean": float(alpha.mean()),
            }
        else:
            fused, semantic, reliability, calibrated, alpha = model.encode(xv_t, xe_t)
            logits = model.head(fused)
            diagnostics = {
                "semantic_vibration_mean": float(semantic[:, 0].mean()),
                "semantic_electrical_mean": float(semantic[:, 1].mean()),
                "reliability_vibration_mean": float(reliability[:, 0].mean()),
                "reliability_electrical_mean": float(reliability[:, 1].mean()),
                "calibrated_vibration_mean": float(calibrated[:, 0].mean()),
                "calibrated_electrical_mean": float(calibrated[:, 1].mean()),
                "switch_mean": float(alpha.mean()),
            }
    return logits.argmax(1).cpu().numpy(), diagnostics


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    loads = args.loads or raw["loads"]
    seeds = args.seeds or raw["seeds"]
    epochs = args.epochs or raw["epochs"]
    models = args.models or raw["models"]
    if args.smoke:
        loads, seeds, epochs = [555], [201], min(2, epochs)
    cfg = Config(
        epochs=epochs,
        batch_size=raw["batch_size"],
        learning_rate=raw["learning_rate"],
        weight_decay=raw["weight_decay"],
        hidden_dim=raw["hidden_dim"],
        dropout=raw["dropout"],
        reliability_loss_weight=raw["reliability_loss_weight"],
        switch_loss_weight=raw["switch_loss_weight"],
        state_probabilities=tuple(raw["state_probabilities"]),
    )
    out_dir = PROJECT_ROOT / "results" / "runs" / args.run_name
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    clean = pd.read_csv(args.reference_csv)
    all_tables = signal_base.load_condition_tables(args.feature_dir, clean)
    tables = {
        key: value
        for key, value in all_tables.items()
        if key in PRIMARY_CONDITIONS
    }
    vibration_columns, electrical_columns = base.split_feature_columns(clean)
    feature_columns = vibration_columns + electrical_columns
    rows, diagnostics, confusion = [], [], {}
    started_all = time.perf_counter()
    for test_load in loads:
        train_clean = clean[clean["load"].astype(int) != int(test_load)].copy()
        test_clean = clean[clean["load"].astype(int) == int(test_load)].copy()
        train_keys = train_clean[signal_base.KEYS].reset_index(drop=True)
        test_keys = test_clean[signal_base.KEYS].reset_index(drop=True)
        scaler_v = StandardScaler().fit(train_clean[vibration_columns])
        scaler_e = StandardScaler().fit(train_clean[electrical_columns])
        train_arrays = {}
        for condition in ["clean_reextract", *TRAIN_CONDITIONS.values()]:
            frame = aligned_subset(all_tables[condition], train_keys, test_load, True)
            if frame[feature_columns].isna().any().any():
                raise ValueError(f"Missing training alignment: {test_load}, {condition}")
            train_arrays[condition] = (
                scaler_v.transform(frame[vibration_columns]).astype(np.float32),
                scaler_e.transform(frame[electrical_columns]).astype(np.float32),
            )
        test_arrays = {}
        for condition, table in tables.items():
            frame = aligned_subset(table, test_keys, test_load, False)
            test_arrays[condition] = (
                scaler_v.transform(frame[vibration_columns]).astype(np.float32),
                scaler_e.transform(frame[electrical_columns]).astype(np.float32),
            )
        clean_v, clean_e = test_arrays["clean_reextract"]
        test_arrays["vibration_missing"] = (np.zeros_like(clean_v), clean_e.copy())
        test_arrays["electrical_missing"] = (clean_v.copy(), np.zeros_like(clean_e))
        y_train = train_clean["label_id"].to_numpy(dtype=np.int64)
        y_test = test_clean["label_id"].to_numpy(dtype=np.int64)
        for seed in seeds:
            for model_key in models:
                print(
                    f"[load={test_load}] [seed={seed}] [model={model_key}] training",
                    flush=True,
                )
                started = time.perf_counter()
                model = train_model(model_key, train_arrays, y_train, seed, cfg)
                train_time = time.perf_counter() - started
                for condition in PRIMARY_CONDITIONS:
                    xv, xe = test_arrays[condition]
                    prediction, diag = predict(model_key, model, xv, xe)
                    row = base.metric_row(y_test, prediction)
                    row.update(
                        {
                            "test_load": int(test_load),
                            "seed": int(seed),
                            "model": model_key,
                            "condition": condition,
                            "n_test": len(y_test),
                            "train_time_s": train_time,
                            "complexity": base.complexity(model_key, model),
                        }
                    )
                    rows.append(row)
                    if diag:
                        qv, qe = quality_target_for_condition(condition)
                        diagnostics.append(
                            {
                                "test_load": int(test_load),
                                "seed": int(seed),
                                "model": model_key,
                                "condition": condition,
                                "quality_target_vibration": qv,
                                "quality_target_electrical": qe,
                                "switch_target": 1.0 - min(qv, qe),
                                **diag,
                            }
                        )
                    confusion[
                        f"{test_load}|{seed}|{model_key}|{condition}"
                    ] = confusion_matrix(
                        y_test, prediction, labels=list(range(6))
                    ).tolist()
                del model
        pd.DataFrame(rows).to_csv(out_dir / "metrics_long.partial.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "diagnostics.csv", index=False)
    (out_dir / "confusion_matrices.json").write_text(
        json.dumps(confusion, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "locked_confirmation",
        "exploratory": False,
        "noise_in_training": False,
        "noise_in_evaluation": False,
        "loads": loads,
        "seeds": seeds,
        "models": models,
        "conditions": PRIMARY_CONDITIONS,
        "config": raw,
        "effective_epochs": epochs,
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

