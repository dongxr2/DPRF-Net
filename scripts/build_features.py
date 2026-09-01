"""Build deterministic raw-signal perturbation feature tables.

The original CSV files are read-only. Each condition is written as a separate
derived feature table under the isolated expansion workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHANNELS = [
    "VibrationX",
    "VibrationY",
    "VibrationZ",
    "Current1",
    "Current2",
    "Current3",
    "Voltage",
]
LABEL_NAMES = ["N", "BB", "BR", "RB3", "RB5", "SW"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_NAMES)}
FILE_PATTERN = re.compile(
    r"(?P<label>[A-Z0-9]+)_(?P<load>\d{3})_(?P<freq>\d+(?:\.\d+)?)\.csv$"
)
CONDITIONS = [
    "clean_reextract",
    "raw_noise_20db",
    "raw_noise_10db",
    "raw_noise_5db",
    "raw_noise_0db",
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


def safe_skew_kurt(x: np.ndarray, mean: float, std: float) -> tuple[float, float]:
    if std < 1e-12:
        return 0.0, 0.0
    z = (x - mean) / std
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def dominant_frequency_index(x: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(x - np.mean(x)))
    if len(spectrum) <= 1:
        return 0.0
    spectrum[0] = 0.0
    return float(np.argmax(spectrum))


def band_energy_ratio(x: np.ndarray, start: float, end: float) -> float:
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    total = float(np.sum(power)) + 1e-12
    n = len(power)
    low = max(0, int(start * n))
    high = min(n, int(end * n))
    return float(np.sum(power[low:high]) / total)


def extract_features(window: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    for index, name in enumerate(CHANNELS):
        signal = window[:, index]
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        abs_mean = float(np.mean(np.abs(signal))) + 1e-12
        rms = float(np.sqrt(np.mean(signal**2)))
        peak = float(np.max(np.abs(signal)))
        skew, kurtosis = safe_skew_kurt(signal, mean, std)
        features[f"{name}_mean"] = mean
        features[f"{name}_std"] = std
        features[f"{name}_rms"] = rms
        features[f"{name}_peak"] = peak
        features[f"{name}_ptp"] = float(np.ptp(signal))
        features[f"{name}_skew"] = skew
        features[f"{name}_kurt"] = kurtosis
        features[f"{name}_crest"] = peak / (rms + 1e-12)
        features[f"{name}_shape"] = rms / abs_mean
        features[f"{name}_domidx"] = dominant_frequency_index(signal)
        features[f"{name}_band_low"] = band_energy_ratio(signal, 0.00, 0.10)
        features[f"{name}_band_mid"] = band_energy_ratio(signal, 0.10, 0.30)
        features[f"{name}_band_high"] = band_energy_ratio(signal, 0.30, 1.00)

    vibration = window[:, 0:3]
    current = window[:, 3:6]
    vibration_norm = np.sqrt(np.sum(vibration**2, axis=1))
    current_rms = np.sqrt(np.mean(current**2, axis=0))
    features["VibVector_rms"] = float(np.sqrt(np.mean(vibration_norm**2)))
    features["VibVector_std"] = float(np.std(vibration_norm))
    features["Current_rms_mean"] = float(np.mean(current_rms))
    features["Current_rms_unbalance"] = float(
        np.std(current_rms) / (np.mean(current_rms) + 1e-12)
    )
    features["Voltage_rms_to_current_rms"] = (
        features["Voltage_rms"] / (features["Current_rms_mean"] + 1e-12)
    )
    return features


def stable_rng(source_file: str, window_id: int, condition: str) -> np.random.Generator:
    payload = f"{source_file}|{window_id}|{condition}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def add_exact_awgn(
    window: np.ndarray, snr_db: float, source_file: str, window_id: int, condition: str
) -> np.ndarray:
    output = window.copy()
    rng = stable_rng(source_file, window_id, condition)
    ratio = np.sqrt(10.0 ** (snr_db / 10.0))
    for channel in range(output.shape[1]):
        signal = output[:, channel]
        signal_rms = max(float(np.sqrt(np.mean(signal**2))), 1e-12)
        noise = rng.standard_normal(len(signal))
        noise -= np.mean(noise)
        noise_rms = max(float(np.sqrt(np.mean(noise**2))), 1e-12)
        output[:, channel] = signal + noise * (signal_rms / ratio / noise_rms)
    return output


def apply_condition(
    window: np.ndarray, condition: str, source_file: str, window_id: int
) -> np.ndarray:
    if condition == "clean_reextract":
        return window
    if condition.startswith("raw_noise_"):
        snr = float(condition.removeprefix("raw_noise_").removesuffix("db"))
        return add_exact_awgn(window, snr, source_file, window_id, condition)
    output = window.copy()
    if "_gain_" in condition:
        gain = int(condition.rsplit("_", 1)[1]) / 100.0
        channels = slice(0, 3) if condition.startswith("vibration") else slice(3, 7)
        output[:, channels] *= gain
        return output
    if "_drift_" in condition:
        magnitude = int(condition.rsplit("_", 1)[1].removesuffix("rms")) / 100.0
        channel_indices = range(0, 3) if condition.startswith("vibration") else range(3, 7)
        ramp = np.linspace(-0.5, 0.5, len(output), dtype=np.float64)
        for channel in channel_indices:
            rms = max(float(np.sqrt(np.mean(output[:, channel] ** 2))), 1e-12)
            output[:, channel] += ramp * magnitude * rms
        return output
    raise ValueError(condition)


def compare_clean(reference_csv: Path, clean: pd.DataFrame) -> dict:
    reference = pd.read_csv(reference_csv)
    keys = ["source_file", "window_id"]
    metadata = {"label", "label_id", "load", "freq", *keys}
    feature_columns = [column for column in reference.columns if column not in metadata]
    merged = reference.merge(clean, on=keys, suffixes=("_reference", "_reextract"))
    differences = []
    for column in feature_columns:
        left = merged[f"{column}_reference"].to_numpy(dtype=np.float64)
        right = merged[f"{column}_reextract"].to_numpy(dtype=np.float64)
        differences.append(np.abs(left - right))
    all_differences = np.concatenate(differences)
    return {
        "reference_rows": len(reference),
        "matched_rows": len(merged),
        "max_abs_difference": float(np.max(all_differences)),
        "mean_abs_difference": float(np.mean(all_differences)),
        "allclose_rtol_1e-10_atol_1e-12": bool(
            np.allclose(all_differences, 0.0, rtol=1e-10, atol=1e-12)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "With_Driver_Dataset")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "clean_reextract.csv.gz",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--windows-per-file", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    files = sorted(args.raw_dir.rglob("*.csv"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(args.raw_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = {condition: [] for condition in CONDITIONS}
    started = time.perf_counter()
    nrows = args.window_size * args.windows_per_file

    for file_index, path in enumerate(files, start=1):
        match = FILE_PATTERN.match(path.name)
        if match is None or match.group("label") not in LABEL_TO_ID:
            raise ValueError(f"Unexpected file name: {path.name}")
        print(f"[{file_index:03d}/{len(files):03d}] {path.name}", flush=True)
        raw = pd.read_csv(path, usecols=CHANNELS, nrows=nrows).to_numpy(
            dtype=np.float64, copy=False
        )
        if len(raw) < nrows:
            raise ValueError(f"Insufficient rows in {path}: {len(raw)} < {nrows}")
        for window_id in range(args.windows_per_file):
            start = window_id * args.window_size
            clean_window = raw[start : start + args.window_size]
            for condition in CONDITIONS:
                perturbed = apply_condition(clean_window, condition, path.name, window_id)
                features = extract_features(perturbed)
                features.update(
                    {
                        "label": match.group("label"),
                        "label_id": LABEL_TO_ID[match.group("label")],
                        "load": int(match.group("load")),
                        "freq": float(match.group("freq")),
                        "source_file": path.name,
                        "window_id": window_id,
                    }
                )
                rows[condition].append(features)

    outputs = {}
    for condition in CONDITIONS:
        frame = pd.DataFrame(rows[condition])
        output = args.out_dir / f"{condition}.csv"
        frame.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
        outputs[condition] = {"path": str(output), "rows": len(frame), "columns": len(frame.columns)}

    clean_frame = pd.DataFrame(rows["clean_reextract"])
    clean_check = (
        compare_clean(args.reference_csv, clean_frame)
        if args.max_files is None
        else {"skipped": "max-files smoke run"}
    )
    manifest = {
        "raw_dir": str(args.raw_dir.resolve()),
        "reference_csv": str(args.reference_csv.resolve()),
        "files": len(files),
        "window_size": args.window_size,
        "windows_per_file": args.windows_per_file,
        "conditions": CONDITIONS,
        "outputs": outputs,
        "clean_reextract_check": clean_check,
        "elapsed_s": time.perf_counter() - started,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
