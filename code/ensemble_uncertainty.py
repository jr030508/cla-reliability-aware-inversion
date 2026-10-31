from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_LABELS = {"Rate_1", "Rate_2", "Rate_3", "Rate_4", "Z"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-ensemble split-conformal uncertainty intervals")
    parser.add_argument("--scenario", choices=["1", "2", "3"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="iid")
    parser.add_argument("--station-order", default="original")
    parser.add_argument("--coverage", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs"))
    return parser.parse_args()


def load_aligned(paths: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    frames = [pd.read_csv(path).set_index("sample_index").sort_index() for path in paths]
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    if len(common) == 0:
        raise ValueError("No common sample indices across ensemble members.")
    # str.removeprefix is unavailable on Python 3.8, which is common with TF 2.6.
    labels = [column[len("True_") :] for column in frames[0].columns if column.startswith("True_")]
    truths = {}
    predictions = {}
    for label in labels:
        truth_stack = np.stack([frame.loc[common, f"True_{label}"].to_numpy(float) for frame in frames])
        if not np.allclose(truth_stack, truth_stack[0], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Truth values differ across ensemble members for {label}.")
        truths[label] = truth_stack[0]
        predictions[label] = np.stack(
            [frame.loc[common, f"Pred_{label}"].to_numpy(float) for frame in frames], axis=0
        )
    return common.to_numpy(), truths, predictions, labels


def transform(label: str, value: np.ndarray) -> np.ndarray:
    if label in POSITIVE_LABELS:
        if np.any(value <= 0):
            raise ValueError(f"{label} contains non-positive values; log-domain uncertainty is invalid.")
        return np.log10(value)
    return value


def inverse(label: str, value: np.ndarray) -> np.ndarray:
    return np.power(10.0, value) if label in POSITIVE_LABELS else value


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between 0 and 1.")
    n = len(residuals)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    try:
        return float(np.quantile(residuals, level, method="higher"))
    except TypeError:
        # NumPy 1.19 uses the former keyword name.
        return float(np.quantile(residuals, level, interpolation="higher"))


def main() -> None:
    args = parse_args()
    model_dir = (
        args.output_root.resolve()
        / f"scenario_{args.scenario}"
        / args.split
        / f"order_{args.station_order}"
        / args.model
    )
    calibration_paths = sorted(model_dir.glob("seed_*/calibration_predictions.csv"))
    test_paths = sorted(model_dir.glob("seed_*/raw_predictions.csv"))
    if len(calibration_paths) < 3 or len(test_paths) != len(calibration_paths):
        raise ValueError("At least three completed seeds with calibration and test predictions are required.")

    cal_index, cal_truth, cal_pred, labels = load_aligned(calibration_paths)
    test_index, test_truth, test_pred, test_labels = load_aligned(test_paths)
    if labels != test_labels:
        raise ValueError("Calibration and test labels differ.")

    prediction_table = pd.DataFrame({"sample_index": test_index})
    metric_rows = []
    for label in labels:
        cal_truth_t = transform(label, cal_truth[label])
        cal_member_t = transform(label, cal_pred[label])
        test_truth_t = transform(label, test_truth[label])
        test_member_t = transform(label, test_pred[label])
        cal_center = np.mean(cal_member_t, axis=0)
        test_center = np.mean(test_member_t, axis=0)
        test_std = np.std(test_member_t, axis=0, ddof=1)
        residuals = np.abs(cal_truth_t - cal_center)

        prediction_table[f"True_{label}"] = test_truth[label]
        prediction_table[f"EnsembleMean_{label}"] = inverse(label, test_center)
        prediction_table[f"EnsembleSD_transformed_{label}"] = test_std

        for coverage in args.coverage:
            q = conformal_quantile(residuals, coverage)
            lower_t, upper_t = test_center - q, test_center + q
            lower, upper = inverse(label, lower_t), inverse(label, upper_t)
            key = int(round(coverage * 100))
            prediction_table[f"Lower{key}_{label}"] = lower
            prediction_table[f"Upper{key}_{label}"] = upper
            empirical = np.mean((test_truth[label] >= lower) & (test_truth[label] <= upper))
            metric_rows.append(
                {
                    "Parameter": label,
                    "Nominal_coverage": coverage,
                    "Empirical_coverage": float(empirical),
                    "Mean_interval_width": float(np.mean(upper - lower)),
                    "Median_interval_width": float(np.median(upper - lower)),
                    "Conformal_quantile_transformed": q,
                    "Calibration_samples": len(cal_index),
                    "Test_samples": len(test_index),
                    "Ensemble_members": len(test_paths),
                    "Transform": "log10" if label in POSITIVE_LABELS else "identity",
                }
            )

    prediction_table.to_csv(model_dir / "ensemble_conformal_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(model_dir / "uncertainty_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"Uncertainty outputs written under {model_dir}")


if __name__ == "__main__":
    main()
