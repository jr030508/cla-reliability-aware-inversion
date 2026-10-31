from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare robustness degradation using paired perturbations")
    parser.add_argument("--scenario", choices=["1", "2", "3"], default="3")
    parser.add_argument("--models", nargs="+", default=["CLA", "CNN_LSTM", "LSTM_Attention", "2D_CNN_Attention"])
    parser.add_argument("--reference", default="CLA")
    parser.add_argument("--split", default="iid")
    parser.add_argument("--station-order", default="original")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs_robustness_focus"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def hierarchical_bootstrap_mean_ci(
    frame: pd.DataFrame, value_column: str, repeats: int, seed: int
) -> tuple[float, float, float]:
    frame = frame[np.isfinite(frame[value_column])]
    seeds = frame["Seed"].unique()
    if not len(seeds):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for i in range(repeats):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            values = frame.loc[frame["Seed"] == sampled_seed, value_column].to_numpy(float)
            sampled_values = values[rng.integers(0, len(values), size=len(values))]
            seed_means.append(np.mean(sampled_values))
        boot[i] = np.mean(seed_means)
    estimate = float(frame.groupby("Seed")[value_column].mean().mean())
    return estimate, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    args = parse_args()
    base = (
        args.output_root.resolve()
        / f"scenario_{args.scenario}"
        / args.split
        / f"order_{args.station_order}"
    )
    frames = []
    for model in args.models:
        for path in sorted((base / model).glob("seed_*/robustness_replicates.csv")):
            frame = pd.read_csv(path)
            frame.insert(0, "Seed", int(path.parent.name.split("_")[-1]))
            frame.insert(0, "Model", model)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No robustness results under {base}")
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(base / "all_robustness_replicates.csv", index=False, encoding="utf-8-sig")

    metric_names = [name for name in ["MAE", "RMSE", "Log10_MAE"] if name in data.columns]
    long = data.melt(
        id_vars=["Model", "Seed", "Dropout_rate", "Lognormal_sigma", "Repeat", "Parameter"],
        value_vars=metric_names,
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=["Value"])
    clean = (
        long[(long["Dropout_rate"] == 0) & (long["Lognormal_sigma"] == 0)]
        .groupby(["Model", "Seed", "Parameter", "Metric"], as_index=False)["Value"]
        .mean()
        .rename(columns={"Value": "Clean_value"})
    )
    long = long.merge(clean, on=["Model", "Seed", "Parameter", "Metric"], validate="many_to_one")
    long["Relative_degradation_percent"] = 100.0 * (long["Value"] - long["Clean_value"]) / long["Clean_value"].clip(lower=1e-30)
    long.to_csv(base / "robustness_degradation_long.csv", index=False, encoding="utf-8-sig")

    condition_keys = ["Seed", "Dropout_rate", "Lognormal_sigma", "Repeat", "Parameter", "Metric"]
    reference = long[long["Model"] == args.reference][condition_keys + ["Relative_degradation_percent"]].rename(
        columns={"Relative_degradation_percent": "Reference_degradation"}
    )
    comparisons = []
    for model in args.models:
        if model == args.reference:
            continue
        candidate = long[long["Model"] == model][condition_keys + ["Relative_degradation_percent"]].rename(
            columns={"Relative_degradation_percent": "Baseline_degradation"}
        )
        paired = reference.merge(candidate, on=condition_keys, validate="one_to_one")
        paired["Robustness_advantage_reference"] = paired["Baseline_degradation"] - paired["Reference_degradation"]
        paired["Baseline"] = model
        comparisons.append(paired)
    paired_all = pd.concat(comparisons, ignore_index=True)
    paired_all.to_csv(base / "paired_robustness_differences.csv", index=False, encoding="utf-8-sig")

    rows = []
    group_keys = ["Baseline", "Dropout_rate", "Lognormal_sigma", "Parameter", "Metric"]
    for key, subset in paired_all.groupby(group_keys):
        estimate, low, high = hierarchical_bootstrap_mean_ci(
            subset, "Robustness_advantage_reference", args.bootstrap_repeats, seed=20260905
        )
        rows.append(
            {
                **dict(zip(group_keys, key)),
                "Mean_advantage_percentage_points": estimate,
                "CI95_low": low,
                "CI95_high": high,
                "Paired_seed_mask_observations": len(subset),
                "Independent_training_seeds": int(subset["Seed"].nunique()),
                "Reference_better_if_positive": True,
            }
        )
    pd.DataFrame(rows).to_csv(base / "robustness_advantage_ci.csv", index=False, encoding="utf-8-sig")

    dropout_only = long[long["Lognormal_sigma"] == 0]
    curve = (
        dropout_only.groupby(["Model", "Seed", "Parameter", "Metric", "Dropout_rate"], as_index=False)["Value"]
        .mean()
        .sort_values("Dropout_rate")
    )
    aurc_rows = []
    for key, subset in curve.groupby(["Model", "Seed", "Parameter", "Metric"]):
        x = subset["Dropout_rate"].to_numpy(float)
        y = subset["Value"].to_numpy(float)
        if len(x) >= 2 and np.max(x) > 0:
            # np.trapezoid is not available in NumPy 1.19.
            aurc = float(np.trapz(y, x) / np.max(x))
            aurc_rows.append({**dict(zip(["Model", "Seed", "Parameter", "Metric"], key)), "Normalized_AURC": aurc})
    pd.DataFrame(aurc_rows).to_csv(base / "dropout_AURC.csv", index=False, encoding="utf-8-sig")
    print(f"Robustness comparison written under {base}")


if __name__ == "__main__":
    main()
