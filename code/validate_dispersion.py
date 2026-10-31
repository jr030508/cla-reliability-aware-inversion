from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dispersion predictions against tracer observations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--observed-column", default="observed")
    parser.add_argument("--predicted-column", default="predicted")
    parser.add_argument("--group-column", default="experiment_id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    valid = np.isfinite(observed) & np.isfinite(predicted) & (observed >= 0) & (predicted >= 0)
    observed = observed[valid].astype(float)
    predicted = predicted[valid].astype(float)
    if len(observed) < 3:
        raise ValueError("At least three valid observation-prediction pairs are required.")
    mean_o, mean_p = float(np.mean(observed)), float(np.mean(predicted))
    if mean_o <= 0 or mean_p <= 0:
        raise ValueError("Mean observed and predicted concentrations must be positive.")
    ratio_valid = (observed > 0) & (predicted > 0)
    ratio = predicted[ratio_valid] / observed[ratio_valid]
    log_difference = np.log(observed[ratio_valid]) - np.log(predicted[ratio_valid])
    correlation = (
        np.corrcoef(observed, predicted)[0, 1]
        if np.std(observed) > 0 and np.std(predicted) > 0
        else np.nan
    )
    return {
        "N": int(len(observed)),
        "Mean_observed": mean_o,
        "Mean_predicted": mean_p,
        "FAC2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if len(ratio) else np.nan,
        "FB": float(2.0 * (mean_p - mean_o) / (mean_p + mean_o)),
        "NMSE": float(np.mean((predicted - observed) ** 2) / (mean_p * mean_o)),
        "Pearson_r": float(correlation),
        "MG": float(np.exp(np.mean(log_difference))) if len(log_difference) else np.nan,
        "VG": float(np.exp(np.mean(log_difference**2))) if len(log_difference) else np.nan,
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {args.observed_column, args.predicted_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input must contain {sorted(required)}.")
    if frame.empty:
        raise ValueError(
            "The tracer template is empty. Fill observed and predicted concentrations with matching station/time pairs."
        )

    rows = [{"Group": "ALL", **calculate_metrics(frame[args.observed_column].to_numpy(), frame[args.predicted_column].to_numpy())}]
    if args.group_column in frame.columns:
        for group, subset in frame.groupby(args.group_column, dropna=False):
            rows.append(
                {
                    "Group": str(group),
                    **calculate_metrics(subset[args.observed_column].to_numpy(), subset[args.predicted_column].to_numpy()),
                }
            )
    result = pd.DataFrame(rows)
    output = args.output or args.input.with_name(f"{args.input.stem}_validation_metrics.csv")
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"Written to {output.resolve()}")


if __name__ == "__main__":
    main()
