from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired CLA attention-ablation analysis")
    parser.add_argument("--scenario", choices=["1", "2", "3"], default="3")
    parser.add_argument("--reference", default="CLA")
    parser.add_argument("--ablation", default="CLA_NoAttention")
    parser.add_argument("--split", default="iid")
    parser.add_argument("--station-order", default="original")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs_linux_robustness"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def hierarchical_bootstrap_ci(
    frame: pd.DataFrame,
    value_column: str,
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    frame = frame[np.isfinite(frame[value_column])]
    seeds = np.sort(frame["Seed"].unique())
    if len(seeds) == 0:
        return np.nan, np.nan, np.nan
    values = {
        current_seed: frame.loc[frame["Seed"] == current_seed, value_column].to_numpy(float)
        for current_seed in seeds
    }
    estimate = float(np.mean([np.mean(values[current_seed]) for current_seed in seeds]))
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            current = values[sampled_seed]
            sampled = rng.choice(current, size=len(current), replace=True)
            seed_means.append(np.mean(sampled))
        boot[index] = np.mean(seed_means)
    return estimate, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def load_manifest(run_dir: Path) -> dict:
    with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_pair(reference_dir: Path, ablation_dir: Path) -> tuple[dict, dict]:
    reference = load_manifest(reference_dir)
    ablation = load_manifest(ablation_dir)
    checks = [
        (reference["scenario"] == ablation["scenario"], "scenario"),
        (reference["seed"] == ablation["seed"], "training seed"),
        (reference["split_seed"] == ablation["split_seed"], "split seed"),
        (reference["data"]["sha256"] == ablation["data"]["sha256"], "dataset hash"),
        (reference["station_order"] == ablation["station_order"], "station order"),
        (reference["model_parameters"] == ablation["model_parameters"], "model/training parameters"),
    ]
    for training_key in [
        "learning_rate_used",
        "batch_size_used",
        "epochs_requested",
        "patience",
        "input_transform",
        "test_fraction",
        "validation_fraction",
        "calibration_fraction",
    ]:
        checks.append(
            (
                reference["training"].get(training_key) == ablation["training"].get(training_key),
                f"training setting {training_key}",
            )
        )
    for valid, name in checks:
        if not valid:
            raise ValueError(f"Attention-ablation pairing failed: different {name}.")
    reference_split = np.load(reference_dir / "split_indices.npz")
    ablation_split = np.load(ablation_dir / "split_indices.npz")
    for partition in ["train", "validation", "calibration", "test", "station_permutation"]:
        if not np.array_equal(reference_split[partition], ablation_split[partition]):
            raise ValueError(f"Attention-ablation pairing failed: different {partition} indices.")
    return reference, ablation


def main() -> None:
    args = parse_args()
    base = (
        args.output_root.resolve()
        / f"scenario_{args.scenario}"
        / args.split
        / f"order_{args.station_order}"
    )
    reference_base = base / args.reference
    ablation_base = base / args.ablation
    reference_seeds = {int(path.name.split("_")[-1]) for path in reference_base.glob("seed_*") if (path / "manifest.json").exists()}
    ablation_seeds = {int(path.name.split("_")[-1]) for path in ablation_base.glob("seed_*") if (path / "manifest.json").exists()}
    seeds = sorted(reference_seeds & ablation_seeds)
    if len(seeds) < 3:
        raise ValueError(
            f"At least three paired seeds are required; found {seeds}. "
            f"Reference seeds={sorted(reference_seeds)}, ablation seeds={sorted(ablation_seeds)}."
        )

    manifests = []
    metric_frames = []
    robustness_frames = []
    clean_test_rows = []
    for seed in seeds:
        reference_dir = reference_base / f"seed_{seed}"
        ablation_dir = ablation_base / f"seed_{seed}"
        reference_manifest, ablation_manifest = validate_pair(reference_dir, ablation_dir)
        manifests.append((reference_manifest, ablation_manifest))

        for model, run_dir in [(args.reference, reference_dir), (args.ablation, ablation_dir)]:
            metrics = pd.read_csv(run_dir / "metrics.csv")
            metrics.insert(0, "Seed", seed)
            metrics.insert(0, "Model", model)
            metric_frames.append(metrics)

            robust = pd.read_csv(run_dir / "robustness_replicates.csv")
            robust.insert(0, "Seed", seed)
            robust.insert(0, "Model", model)
            robustness_frames.append(robust)

        reference_predictions = pd.read_csv(reference_dir / "raw_predictions.csv")
        ablation_predictions = pd.read_csv(ablation_dir / "raw_predictions.csv")
        error_columns = sorted(
            column
            for column in set(reference_predictions.columns) & set(ablation_predictions.columns)
            if column.startswith("AE_") or column.startswith("LOGAE_") or column == "XY_DISTANCE_km"
        )
        paired = reference_predictions[["sample_index"] + error_columns].merge(
            ablation_predictions[["sample_index"] + error_columns],
            on="sample_index",
            suffixes=("_CLA", "_NoAttention"),
            validate="one_to_one",
        )
        for metric in error_columns:
            reference_values = paired[f"{metric}_CLA"].to_numpy(float)
            ablation_values = paired[f"{metric}_NoAttention"].to_numpy(float)
            difference = ablation_values - reference_values
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(
                    ablation_values,
                    reference_values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            clean_test_rows.append(
                {
                    "Seed": seed,
                    "Sample_metric": metric,
                    "N": len(paired),
                    "CLA_mean_error": float(np.mean(reference_values)),
                    "NoAttention_mean_error": float(np.mean(ablation_values)),
                    "Mean_advantage_CLA": float(np.mean(difference)),
                    "Median_advantage_CLA": float(np.median(difference)),
                    "CLA_sample_win_fraction": float(np.mean(reference_values < ablation_values)),
                    "Wilcoxon_statistic": float(statistic),
                    "P_value_raw": float(p_value),
                }
            )

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    numeric_columns = all_metrics.select_dtypes(include=[np.number]).columns.difference(["Seed"])
    clean_summary = (
        all_metrics.groupby(["Model", "Parameter"])[numeric_columns]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    clean_summary.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else column
        for column in clean_summary.columns
    ]
    clean_summary.to_csv(base / "attention_ablation_clean_summary.csv", index=False, encoding="utf-8-sig")

    clean_tests = pd.DataFrame(clean_test_rows)
    clean_tests["P_value_Holm"] = np.nan
    for seed, indices in clean_tests.groupby("Seed").groups.items():
        clean_tests.loc[indices, "P_value_Holm"] = holm_adjust(clean_tests.loc[indices, "P_value_raw"].tolist())
    clean_tests["CLA_better"] = clean_tests["Mean_advantage_CLA"] > 0
    clean_tests.to_csv(base / "attention_ablation_clean_paired_tests.csv", index=False, encoding="utf-8-sig")

    robustness = pd.concat(robustness_frames, ignore_index=True)
    metric_columns = [name for name in ["MAE", "RMSE", "Log10_MAE"] if name in robustness.columns]
    long = robustness.melt(
        id_vars=["Model", "Seed", "Dropout_rate", "Lognormal_sigma", "Repeat", "Parameter"],
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])
    keys = ["Seed", "Dropout_rate", "Lognormal_sigma", "Repeat", "Parameter", "Metric"]
    reference_long = long[long["Model"] == args.reference][keys + ["Value"]].rename(columns={"Value": "CLA_value"})
    ablation_long = long[long["Model"] == args.ablation][keys + ["Value"]].rename(columns={"Value": "NoAttention_value"})
    paired_robustness = reference_long.merge(ablation_long, on=keys, validate="one_to_one")

    clean_reference = (
        paired_robustness[(paired_robustness["Dropout_rate"] == 0) & (paired_robustness["Lognormal_sigma"] == 0)]
        .groupby(["Seed", "Parameter", "Metric"], as_index=False)["CLA_value"]
        .mean()
        .rename(columns={"CLA_value": "CLA_clean"})
    )
    clean_ablation = (
        paired_robustness[(paired_robustness["Dropout_rate"] == 0) & (paired_robustness["Lognormal_sigma"] == 0)]
        .groupby(["Seed", "Parameter", "Metric"], as_index=False)["NoAttention_value"]
        .mean()
        .rename(columns={"NoAttention_value": "NoAttention_clean"})
    )
    paired_robustness = paired_robustness.merge(clean_reference, on=["Seed", "Parameter", "Metric"], validate="many_to_one")
    paired_robustness = paired_robustness.merge(clean_ablation, on=["Seed", "Parameter", "Metric"], validate="many_to_one")
    paired_robustness["CLA_relative_degradation_percent"] = 100.0 * (
        paired_robustness["CLA_value"] - paired_robustness["CLA_clean"]
    ) / paired_robustness["CLA_clean"].clip(lower=1e-30)
    paired_robustness["NoAttention_relative_degradation_percent"] = 100.0 * (
        paired_robustness["NoAttention_value"] - paired_robustness["NoAttention_clean"]
    ) / paired_robustness["NoAttention_clean"].clip(lower=1e-30)
    paired_robustness["Absolute_advantage_CLA"] = (
        paired_robustness["NoAttention_value"] - paired_robustness["CLA_value"]
    )
    paired_robustness["Relative_degradation_advantage_CLA"] = (
        paired_robustness["NoAttention_relative_degradation_percent"]
        - paired_robustness["CLA_relative_degradation_percent"]
    )
    paired_robustness.to_csv(base / "attention_ablation_robustness_paired.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    group_columns = ["Dropout_rate", "Lognormal_sigma", "Parameter", "Metric"]
    for key, subset in paired_robustness.groupby(group_columns):
        row = dict(zip(group_columns, key))
        for column, prefix, local_seed in [
            ("Absolute_advantage_CLA", "Absolute", 20260910),
            ("Relative_degradation_advantage_CLA", "Relative_degradation", 20260911),
        ]:
            estimate, low, high = hierarchical_bootstrap_ci(
                subset,
                column,
                args.bootstrap_repeats,
                seed=local_seed,
            )
            row[f"{prefix}_advantage_CLA"] = estimate
            row[f"{prefix}_CI95_low"] = low
            row[f"{prefix}_CI95_high"] = high
            row[f"{prefix}_CLA_better"] = bool(low > 0)
            row[f"{prefix}_NoAttention_better"] = bool(high < 0)
        row["Independent_training_seeds"] = int(subset["Seed"].nunique())
        row["Paired_seed_mask_observations"] = len(subset)
        summary_rows.append(row)
    robust_summary = pd.DataFrame(summary_rows)
    robust_summary.to_csv(base / "attention_ablation_robustness_summary.csv", index=False, encoding="utf-8-sig")

    dropout = long[long["Lognormal_sigma"] == 0].copy()
    curve = (
        dropout.groupby(["Model", "Seed", "Parameter", "Metric", "Dropout_rate"], as_index=False)["Value"]
        .mean()
        .sort_values("Dropout_rate")
    )
    aurc_rows = []
    for key, subset in curve.groupby(["Model", "Seed", "Parameter", "Metric"]):
        x = subset["Dropout_rate"].to_numpy(float)
        y = subset["Value"].to_numpy(float)
        if len(x) >= 2 and np.max(x) > 0:
            aurc_rows.append(
                {
                    **dict(zip(["Model", "Seed", "Parameter", "Metric"], key)),
                    "Normalized_AURC": float(np.trapz(y, x) / np.max(x)),
                }
            )
    pd.DataFrame(aurc_rows).to_csv(base / "attention_ablation_dropout_AURC.csv", index=False, encoding="utf-8-sig")

    significant_clean = clean_tests[clean_tests["P_value_Holm"] < 0.05]
    perturbed = robust_summary[
        ~((robust_summary["Dropout_rate"] == 0) & (robust_summary["Lognormal_sigma"] == 0))
    ]
    report_lines = [
        "# CLA注意力消融实验自动摘要",
        "",
        f"- 配对训练随机种子：{', '.join(str(seed) for seed in seeds)}",
        f"- CLA参数量：{manifests[0][0]['total_parameters']}",
        f"- CLA_NoAttention参数量：{manifests[0][1]['total_parameters']}",
        "- 两模型使用相同数据哈希、训练/验证/校准/测试划分、超参数和扰动随机数。",
        "",
        "## 无扰动精度",
        "",
        f"- Holm校正后CLA显著更优：{int(((significant_clean['CLA_better']) == True).sum())}项。",
        f"- Holm校正后无Attention显著更优：{int(((significant_clean['CLA_better']) == False).sum())}项。",
        f"- 未达到显著差异：{int((clean_tests['P_value_Holm'] >= 0.05).sum())}项。",
        "",
        "## 扰动实验",
        "",
        f"- 扰动后绝对误差的95%CI支持CLA更优：{int(perturbed['Absolute_CLA_better'].sum())}项。",
        f"- 扰动后绝对误差的95%CI支持无Attention更优：{int(perturbed['Absolute_NoAttention_better'].sum())}项。",
        f"- 相对退化率的95%CI支持CLA更优：{int(perturbed['Relative_degradation_CLA_better'].sum())}项。",
        f"- 相对退化率的95%CI支持无Attention更优：{int(perturbed['Relative_degradation_NoAttention_better'].sum())}项。",
        "",
        "正的advantage表示CLA误差更低或相对退化更小。由于只有3个独立训练种子，结论应同时报告效应量和置信区间。",
    ]
    (base / "attention_ablation_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Attention-ablation outputs written under {base}")


if __name__ == "__main__":
    main()
