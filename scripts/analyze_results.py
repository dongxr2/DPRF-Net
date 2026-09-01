"""Locked load-level statistical analysis for DPRF-Net confirmation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


DEGRADED = [
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
FAMILIES = {
    "missing": ["vibration_missing", "electrical_missing"],
    "gain": [item for item in DEGRADED if "_gain_" in item],
    "drift": [item for item in DEGRADED if "_drift_" in item],
}
COMPARATORS = [
    "gate_da",
    "mean_da",
    "rcarf_hard",
    "dprf_no_rel_loss",
    "dprf_no_switch_loss",
    "dprf_no_semantic",
]
CLASS_NAMES = ["N", "BB", "BR", "RB3", "RB5", "SW"]
CURVES = {
    "vibration_gain": (
        ["vibration_gain_075", "vibration_gain_050", "vibration_gain_025"],
        "quality_target_vibration",
        "reliability_vibration_mean",
    ),
    "electrical_gain": (
        ["electrical_gain_075", "electrical_gain_050", "electrical_gain_025"],
        "quality_target_electrical",
        "reliability_electrical_mean",
    ),
    "vibration_drift": (
        ["vibration_drift_025rms", "vibration_drift_050rms", "vibration_drift_100rms"],
        "quality_target_vibration",
        "reliability_vibration_mean",
    ),
    "electrical_drift": (
        ["electrical_drift_025rms", "electrical_drift_050rms", "electrical_drift_100rms"],
        "quality_target_electrical",
        "reliability_electrical_mean",
    ),
}


def bootstrap_ci(values: np.ndarray, seed: int = 20260730) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def paired_statistics(load_scores: pd.DataFrame) -> pd.DataFrame:
    pivot = load_scores.pivot(index="test_load", columns="model", values="macro_f1")
    rows = []
    for comparator in COMPARATORS:
        difference = (pivot["dprf_full"] - pivot[comparator]).to_numpy()
        if np.allclose(difference, 0):
            p_value = 1.0
        else:
            p_value = float(
                wilcoxon(
                    pivot["dprf_full"],
                    pivot[comparator],
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        ci_low, ci_high = bootstrap_ci(difference)
        sd = float(np.std(difference, ddof=1))
        rows.append(
            {
                "comparison": f"dprf_full vs {comparator}",
                "dprf_mean": float(pivot["dprf_full"].mean()),
                "comparator_mean": float(pivot[comparator].mean()),
                "mean_difference": float(difference.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "cohens_dz": float(difference.mean() / sd) if sd > 0 else np.inf,
                "wins": int((difference > 1e-12).sum()),
                "ties": int((np.abs(difference) <= 1e-12).sum()),
                "losses": int((difference < -1e-12).sum()),
                "wilcoxon_p_raw": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_raw"].tolist())
    return result


def mechanism_statistics(diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dprf = diagnostics[diagnostics["model"].str.startswith("dprf_")].copy()
    dprf["reliability_mae"] = 0.5 * (
        (dprf["reliability_vibration_mean"] - dprf["quality_target_vibration"]).abs()
        + (
            dprf["reliability_electrical_mean"]
            - dprf["quality_target_electrical"]
        ).abs()
    )
    dprf["switch_mae"] = (dprf["switch_mean"] - dprf["switch_target"]).abs()
    load_mechanisms = (
        dprf.groupby(["test_load", "model"], as_index=False)[
            ["reliability_mae", "switch_mae"]
        ]
        .mean()
        .sort_values(["model", "test_load"])
    )
    pivot_rel = load_mechanisms.pivot(
        index="test_load", columns="model", values="reliability_mae"
    )
    pivot_switch = load_mechanisms.pivot(
        index="test_load", columns="model", values="switch_mae"
    )
    rows = []
    for metric, pivot, comparator in [
        ("reliability_mae", pivot_rel, "dprf_no_rel_loss"),
        ("switch_mae", pivot_switch, "dprf_no_switch_loss"),
    ]:
        difference = (pivot["dprf_full"] - pivot[comparator]).to_numpy()
        ci_low, ci_high = bootstrap_ci(difference, seed=20260731)
        rows.append(
            {
                "metric": metric,
                "comparison": f"dprf_full vs {comparator}",
                "dprf_mean": float(pivot["dprf_full"].mean()),
                "comparator_mean": float(pivot[comparator].mean()),
                "mean_difference": float(difference.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wins_lower_is_better": int((difference < 0).sum()),
                "losses": int((difference > 0).sum()),
                "wilcoxon_p_raw": float(
                    wilcoxon(
                        pivot["dprf_full"],
                        pivot[comparator],
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                ),
            }
        )
    return load_mechanisms, pd.DataFrame(rows)


def monotonicity_by_load(diagnostics: pd.DataFrame) -> pd.DataFrame:
    full = diagnostics[diagnostics["model"] == "dprf_full"]
    rows = []
    for test_load, frame in full.groupby("test_load"):
        averaged = frame.groupby("condition", as_index=True).mean(numeric_only=True)
        for curve, (conditions, target_column, estimate_column) in CURVES.items():
            targets = averaged.loc[conditions, target_column].to_numpy()
            estimates = averaged.loc[conditions, estimate_column].to_numpy()
            rho = float(spearmanr(targets, estimates).statistic)
            rows.append(
                {
                    "test_load": int(test_load),
                    "curve": curve,
                    "spearman_rho": rho,
                    "direction_correct": bool(rho > 0),
                    "estimates": json.dumps(estimates.tolist()),
                }
            )
    return pd.DataFrame(rows)


def confusion_audit(
    matrices: dict[str, list[list[int]]], metadata: dict
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    expected = (
        len(metadata["loads"])
        * len(metadata["seeds"])
        * len(metadata["models"])
        * len(metadata["conditions"])
    )
    rows = []
    invalid_shape = 0
    for key, raw_matrix in matrices.items():
        matrix = np.asarray(raw_matrix, dtype=float)
        if matrix.shape != (6, 6):
            invalid_shape += 1
            continue
        test_load, seed, model, condition = key.split("|", 3)
        denominators = matrix.sum(axis=1)
        recall = np.divide(
            np.diag(matrix),
            denominators,
            out=np.full(6, np.nan),
            where=denominators > 0,
        )
        for class_name, value in zip(CLASS_NAMES, recall):
            rows.append(
                {
                    "test_load": int(test_load),
                    "seed": int(seed),
                    "model": model,
                    "condition": condition,
                    "class_name": class_name,
                    "recall": float(value),
                    "collapsed_below_0_1": bool(value < 0.1),
                }
            )
    recalls = pd.DataFrame(rows)
    collapsed = recalls[
        recalls["collapsed_below_0_1"]
        & recalls["condition"].isin(DEGRADED)
    ].copy()
    audit = {
        "matrices": len(matrices),
        "expected_matrices": expected,
        "invalid_shape": invalid_shape,
        "finite_recalls": bool(np.isfinite(recalls["recall"]).all()),
        "collapsed_cells_all_models": int(len(collapsed)),
        "collapsed_cells_dprf_full": int(
            (collapsed["model"] == "dprf_full").sum()
        ),
        "passed": bool(
            len(matrices) == expected
            and invalid_shape == 0
            and np.isfinite(recalls["recall"]).all()
        ),
    }
    class_summary = (
        recalls[recalls["condition"].isin(DEGRADED)]
        .groupby(["model", "class_name"], as_index=False)
        .agg(mean_recall=("recall", "mean"), min_recall=("recall", "min"))
    )
    return audit, class_summary, collapsed


def write_summary(
    run_dir: Path,
    model_summary: pd.DataFrame,
    paired: pd.DataFrame,
    mechanism: pd.DataFrame,
    monotonicity: pd.DataFrame,
    confusion: dict,
) -> None:
    score = model_summary.set_index("model")
    lines = [
        "# DPRF-Net 锁定确认实验结果摘要",
        "",
        "## 完整性",
        "",
        f"- 主指标记录：3150/3150；重复键 0；数值均有限。",
        f"- 混淆矩阵：{confusion['matrices']}/{confusion['expected_matrices']}，"
        f"形状异常 {confusion['invalid_shape']}。",
        f"- DPRF-Net 在退化工况中的类别召回率低于 0.1 的单元数："
        f"{confusion['collapsed_cells_dprf_full']}。",
        "",
        "## 主要结果",
        "",
        f"- DPRF-Net：{score.loc['dprf_full', 'all_degraded_macro_f1']:.4f}；"
        f"正常工况 {score.loc['dprf_full', 'clean_macro_f1']:.4f}。",
        f"- RCARF-hard：{score.loc['rcarf_hard', 'all_degraded_macro_f1']:.4f}；"
        f"Gate-DA：{score.loc['gate_da', 'all_degraded_macro_f1']:.4f}；"
        f"Mean-DA：{score.loc['mean_da', 'all_degraded_macro_f1']:.4f}。",
        f"- DPRF-Net 最差工况均值："
        f"{score.loc['dprf_full', 'worst_condition_mean']:.4f}。",
        "",
        "## 负载级比较",
        "",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: 差值 {row.mean_difference:+.4f}，"
            f"95% CI [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}]，"
            f"胜/平/负 {row.wins}/{row.ties}/{row.losses}，"
            f"Holm 校正 p={row.wilcoxon_p_holm:.4f}。"
        )
    lines.extend(["", "## 机制指标", ""])
    for row in mechanism.itertuples(index=False):
        lines.append(
            f"- {row.metric}: DPRF-Net {row.dprf_mean:.4f}，"
            f"对照 {row.comparator_mean:.4f}，差值 {row.mean_difference:+.4f}。"
        )
    direction_count = int(monotonicity["direction_correct"].sum())
    lines.extend(
        [
            f"- 负载级可靠性曲线方向正确：{direction_count}/{len(monotonicity)}。",
            "",
            "## 结论边界",
            "",
            "- 结果支持在本数据源留一负载条件下的内部确认，不构成跨设备外部验证。",
            "- 分类优势、可靠性校准与残差开关作用分开陈述，不把机制可解释性等同于准确率提升。",
        ]
    )
    (run_dir / "RESULT_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text("utf-8"))
    metrics = pd.read_csv(run_dir / "metrics_long.csv")
    diagnostics = pd.read_csv(run_dir / "diagnostics.csv")
    matrices = json.loads((run_dir / "confusion_matrices.json").read_text("utf-8"))

    expected_rows = (
        len(metadata["loads"])
        * len(metadata["seeds"])
        * len(metadata["models"])
        * len(metadata["conditions"])
    )
    validation = {
        "rows": int(len(metrics)),
        "expected_rows": int(expected_rows),
        "duplicates": int(
            metrics.duplicated(["test_load", "seed", "model", "condition"]).sum()
        ),
        "finite": bool(
            np.isfinite(metrics[["accuracy", "macro_f1", "weighted_f1"]]).all().all()
        ),
        "noise_absent": bool(
            not metrics["condition"].str.contains("noise", case=False).any()
            and not metadata["noise_in_training"]
            and not metadata["noise_in_evaluation"]
        ),
    }
    validation["passed"] = bool(
        validation["rows"] == validation["expected_rows"]
        and validation["duplicates"] == 0
        and validation["finite"]
        and validation["noise_absent"]
    )
    if not validation["passed"]:
        raise RuntimeError(validation)

    degraded = metrics[metrics["condition"].isin(DEGRADED)]
    load_scores = (
        degraded.groupby(["test_load", "model"], as_index=False)["macro_f1"].mean()
    )
    model_summary = (
        degraded.groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "all_degraded_macro_f1"})
    )
    clean = (
        metrics[metrics["condition"] == "clean_reextract"]
        .groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "clean_macro_f1"})
    )
    worst = (
        degraded.groupby(["model", "condition"], as_index=False)["macro_f1"]
        .mean()
        .groupby("model", as_index=False)["macro_f1"]
        .min()
        .rename(columns={"macro_f1": "worst_condition_mean"})
    )
    model_summary = (
        model_summary.merge(clean, on="model")
        .merge(worst, on="model")
        .sort_values("all_degraded_macro_f1", ascending=False)
    )
    family_rows = []
    for family, conditions in FAMILIES.items():
        subset = metrics[metrics["condition"].isin(conditions)]
        grouped = subset.groupby("model", as_index=False)["macro_f1"].mean()
        grouped["family"] = family
        family_rows.append(grouped)
    family_summary = pd.concat(family_rows, ignore_index=True)

    paired = paired_statistics(load_scores)
    load_mechanisms, mechanism = mechanism_statistics(diagnostics)
    monotonicity = monotonicity_by_load(diagnostics)
    confusion, class_summary, collapsed = confusion_audit(matrices, metadata)

    load_scores.to_csv(run_dir / "load_level_scores.csv", index=False)
    model_summary.to_csv(run_dir / "confirmation_model_summary.csv", index=False)
    family_summary.to_csv(run_dir / "family_summary.csv", index=False)
    paired.to_csv(run_dir / "paired_comparisons.csv", index=False)
    load_mechanisms.to_csv(run_dir / "load_level_mechanisms.csv", index=False)
    mechanism.to_csv(run_dir / "mechanism_comparisons.csv", index=False)
    monotonicity.to_csv(run_dir / "load_level_monotonicity.csv", index=False)
    class_summary.to_csv(run_dir / "class_recall_summary.csv", index=False)
    collapsed.to_csv(run_dir / "collapsed_class_cells.csv", index=False)

    report = {
        "validation": validation,
        "confusion_audit": confusion,
        "model_summary": model_summary.to_dict(orient="records"),
        "paired_comparisons": paired.to_dict(orient="records"),
        "mechanism_comparisons": mechanism.to_dict(orient="records"),
        "monotonicity_direction_correct": int(
            monotonicity["direction_correct"].sum()
        ),
        "monotonicity_total": int(len(monotonicity)),
    }
    (run_dir / "confirmation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(
        run_dir, model_summary, paired, mechanism, monotonicity, confusion
    )
    print(json.dumps({"validation": validation, "confusion": confusion}, indent=2))
    print("\nModel summary:")
    print(model_summary.round(6).to_string(index=False))
    print("\nPaired comparisons:")
    print(paired.round(6).to_string(index=False))
    print("\nMechanism comparisons:")
    print(mechanism.round(6).to_string(index=False))
    print(
        f"\nMonotonicity directions: "
        f"{int(monotonicity['direction_correct'].sum())}/{len(monotonicity)}"
    )


if __name__ == "__main__":
    main()
