from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


PRIMARY_MODELS = {"CLA", "LSTM_Attention", "CLA_NoStationMix"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate scenario-3 IID, spatial-block, edge and station-order experiments."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def condition_from_manifest(manifest: dict) -> dict:
    split = manifest["split"]["split"]
    edge_label = manifest["split"].get("edge_label", "")
    return {
        "Split": split,
        "Edge_label": edge_label,
        "Station_order": manifest.get("station_order", "original"),
        "Station_order_seed": manifest.get("station_order_seed", manifest.get("split_seed", 42)),
        "Split_seed": manifest.get("split_seed", 42),
    }


def sample_metrics(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    rate_columns = [f"LOGAE_Rate_{i}" for i in range(1, 5) if f"LOGAE_Rate_{i}" in raw]
    if rate_columns:
        result["Rate_macro_Log10MAE"] = raw[rate_columns].to_numpy(float).mean(axis=1)
    if "XY_DISTANCE_km" in raw:
        result["XY_distance_km"] = raw["XY_DISTANCE_km"].to_numpy(float)
    if "LOGAE_Z" in raw:
        result["Z_Log10MAE"] = raw["LOGAE_Z"].to_numpy(float)
    return result


def run_estimates(values: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for metric, vector in values.items():
        rows.append({"Metric": metric, "Statistic": "mean", "Estimate": float(np.mean(vector))})
        if metric == "XY_distance_km":
            rows.extend(
                [
                    {"Metric": metric, "Statistic": "median", "Estimate": float(np.median(vector))},
                    {"Metric": metric, "Statistic": "p95", "Estimate": float(np.quantile(vector, 0.95))},
                ]
            )
    return rows


def paired_bootstrap_ci(difference: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(difference)
    bootstrap = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, n, size=n)
        bootstrap[index] = np.mean(difference[sample])
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def main() -> None:
    args = parse_args()
    scenario_root = args.output_root.resolve() / "scenario_3"
    manifests = sorted(scenario_root.glob("**/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No completed scenario-3 manifests under {scenario_root}")

    run_rows: list[dict] = []
    samples_by_condition: dict[tuple, dict[str, pd.DataFrame]] = {}
    trace_rows: list[dict] = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("model") not in PRIMARY_MODELS:
            continue
        raw_path = manifest_path.with_name("raw_predictions.csv")
        if not raw_path.exists():
            continue
        raw = pd.read_csv(raw_path)
        condition = condition_from_manifest(manifest)
        model = manifest["model"]
        seed = int(manifest["seed"])
        values = sample_metrics(raw)
        for estimate in run_estimates(values):
            run_rows.append(
                {
                    **condition,
                    "Model": model,
                    "Seed": seed,
                    "N_test": len(raw),
                    **estimate,
                }
            )
        key = (
            condition["Split"],
            condition["Edge_label"],
            condition["Station_order"],
            condition["Station_order_seed"],
            condition["Split_seed"],
            seed,
        )
        samples_by_condition.setdefault(key, {})[model] = raw
        trace_rows.append(
            {
                **condition,
                "Model": model,
                "Seed": seed,
                "Manifest": str(manifest_path),
                "Predictions": str(raw_path),
                "Parameter_status": manifest.get("parameter_provenance", {}).get("status", "unknown"),
                "Parameter_file_sha256": manifest.get("data", {}).get("parameter_file_sha256", ""),
            }
        )

    run_frame = pd.DataFrame(run_rows)
    if run_frame.empty:
        raise RuntimeError("No eligible completed runs contained raw predictions.")

    grouping = [
        "Split",
        "Edge_label",
        "Station_order",
        "Station_order_seed",
        "Split_seed",
        "Model",
        "Metric",
        "Statistic",
    ]
    summary = (
        run_frame.groupby(grouping, dropna=False)["Estimate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "Seed_mean", "std": "Seed_SD", "count": "Seed_N"})
    )
    summary["Seed_SE"] = summary["Seed_SD"] / np.sqrt(summary["Seed_N"])
    summary["CI95_low"] = np.nan
    summary["CI95_high"] = np.nan
    valid = summary["Seed_N"] >= 2
    critical = summary.loc[valid, "Seed_N"].map(lambda n: student_t.ppf(0.975, int(n) - 1))
    half_width = critical.to_numpy() * summary.loc[valid, "Seed_SE"].to_numpy()
    summary.loc[valid, "CI95_low"] = summary.loc[valid, "Seed_mean"].to_numpy() - half_width
    summary.loc[valid, "CI95_high"] = summary.loc[valid, "Seed_mean"].to_numpy() + half_width

    comparison_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    for key, model_frames in sorted(samples_by_condition.items()):
        split, edge_label, station_order, order_seed, split_seed, training_seed = key
        if "CLA" not in model_frames:
            continue
        for comparator in ["LSTM_Attention", "CLA_NoStationMix"]:
            if comparator not in model_frames:
                continue
            left = model_frames["CLA"]
            right = model_frames[comparator]
            merged = left.merge(right, on="sample_index", suffixes=("_CLA", "_Comparator"), validate="one_to_one")
            left_metrics = sample_metrics(
                merged.rename(columns={c: c.removesuffix("_CLA") for c in merged if c.endswith("_CLA")})
            )
            right_metrics = sample_metrics(
                merged.rename(columns={c: c.removesuffix("_Comparator") for c in merged if c.endswith("_Comparator")})
            )
            for metric in sorted(set(left_metrics) & set(right_metrics)):
                cla_values = left_metrics[metric]
                comparator_values = right_metrics[metric]
                difference = cla_values - comparator_values
                if np.allclose(difference, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
                ci_low, ci_high = paired_bootstrap_ci(difference, args.bootstrap_repeats, rng)
                comparison_rows.append(
                    {
                        "Split": split,
                        "Edge_label": edge_label,
                        "Station_order": station_order,
                        "Station_order_seed": order_seed,
                        "Split_seed": split_seed,
                        "Seed": training_seed,
                        "Comparator": comparator,
                        "Metric": metric,
                        "N": len(difference),
                        "CLA_mean": float(np.mean(cla_values)),
                        "Comparator_mean": float(np.mean(comparator_values)),
                        "Mean_difference_CLA_minus_comparator": float(np.mean(difference)),
                        "Difference_CI95_low": ci_low,
                        "Difference_CI95_high": ci_high,
                        "Wilcoxon_statistic": float(statistic),
                        "P_value_raw": float(p_value),
                    }
                )

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["P_value_Holm"] = np.nan
        correction_group = [
            "Split",
            "Edge_label",
            "Station_order",
            "Station_order_seed",
            "Split_seed",
            "Seed",
        ]
        for _, indices in comparisons.groupby(correction_group, dropna=False).groups.items():
            comparisons.loc[indices, "P_value_Holm"] = holm_adjust(
                comparisons.loc[indices, "P_value_raw"].tolist()
            )
        comparisons["CLA_better"] = comparisons["Mean_difference_CLA_minus_comparator"] < 0

    destination = scenario_root / "generalization_summary"
    destination.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(destination / "run_level_primary_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(destination / "seed_summary_primary_metrics.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(destination / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(trace_rows).to_csv(destination / "result_traceability.csv", index=False, encoding="utf-8-sig")
    print(f"Generalization summaries written to {destination}")


if __name__ == "__main__":
    main()
