from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated-seed experiments and paired tests")
    parser.add_argument("--scenario", choices=["1", "2", "3"], required=True)
    parser.add_argument("--split", default="iid")
    parser.add_argument("--station-order", default="original")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs"))
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    base = (
        args.output_root.resolve()
        / f"scenario_{args.scenario}"
        / args.split
        / f"order_{args.station_order}"
    )
    metric_files = sorted(base.glob("*/seed_*/metrics.csv"))
    prediction_files = sorted(base.glob("*/seed_*/raw_predictions.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No completed runs under {base}")

    metrics_frames = []
    for path in metric_files:
        model = path.parent.parent.name
        seed = int(path.parent.name.split("_")[-1])
        frame = pd.read_csv(path)
        frame.insert(0, "Seed", seed)
        frame.insert(0, "Model", model)
        metrics_frames.append(frame)
    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_metrics.to_csv(base / "all_seed_metrics.csv", index=False, encoding="utf-8-sig")

    numeric_cols = all_metrics.select_dtypes(include=[np.number]).columns.difference(["Seed"])
    summary = (
        all_metrics.groupby(["Model", "Parameter"])[numeric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in column if str(part)]) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary.to_csv(base / "model_seed_summary.csv", index=False, encoding="utf-8-sig")

    records = []
    by_seed: dict[int, dict[str, pd.DataFrame]] = {}
    for path in prediction_files:
        model = path.parent.parent.name
        seed = int(path.parent.name.split("_")[-1])
        by_seed.setdefault(seed, {})[model] = pd.read_csv(path)

    for seed, model_frames in sorted(by_seed.items()):
        for model_a, model_b in itertools.combinations(sorted(model_frames), 2):
            a = model_frames[model_a]
            b = model_frames[model_b]
            error_cols = sorted(set(c for c in a.columns if c.startswith("AE_") or c.startswith("LOGAE_") or c == "XY_DISTANCE_km") & set(b.columns))
            merged = a[["sample_index"] + error_cols].merge(
                b[["sample_index"] + error_cols], on="sample_index", suffixes=("_A", "_B"), validate="one_to_one"
            )
            for metric in error_cols:
                value_a = merged[f"{metric}_A"].to_numpy(float)
                value_b = merged[f"{metric}_B"].to_numpy(float)
                difference = value_a - value_b
                if np.allclose(difference, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(value_a, value_b, alternative="two-sided", zero_method="wilcox")
                records.append(
                    {
                        "Seed": seed,
                        "Model_A": model_a,
                        "Model_B": model_b,
                        "Sample_metric": metric,
                        "N": len(merged),
                        "Median_A": float(np.median(value_a)),
                        "Median_B": float(np.median(value_b)),
                        "Median_paired_difference_A_minus_B": float(np.median(difference)),
                        "Wilcoxon_statistic": float(statistic),
                        "P_value_raw": float(p_value),
                    }
                )

    tests = pd.DataFrame(records)
    if not tests.empty:
        tests["P_value_Holm"] = np.nan
        for (_, metric), index in tests.groupby(["Seed", "Sample_metric"]).groups.items():
            p_values = tests.loc[index, "P_value_raw"].tolist()
            tests.loc[index, "P_value_Holm"] = holm_adjust(p_values)
        tests["Better_model"] = np.where(
            tests["Median_paired_difference_A_minus_B"] < 0,
            tests["Model_A"],
            np.where(tests["Median_paired_difference_A_minus_B"] > 0, tests["Model_B"], "Tie"),
        )
    tests.to_csv(base / "paired_wilcoxon_holm.csv", index=False, encoding="utf-8-sig")
    print(f"Aggregated outputs written under {base}")


if __name__ == "__main__":
    main()
