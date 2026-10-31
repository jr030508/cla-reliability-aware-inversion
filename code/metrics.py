from __future__ import annotations

import numpy as np
import pandas as pd
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), np.finfo(np.float64).eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator) * 100.0)


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(np.abs(y_pred - y_true)) / np.maximum(np.sum(np.abs(y_true)), 1e-30) * 100.0)


def parameter_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    error = y_pred - y_true
    abs_error = np.abs(error)
    ape = abs_error / np.maximum(np.abs(y_true), np.finfo(np.float64).eps)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_physical = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else np.nan
    row = {
        "Parameter": label,
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": r2_physical,
        "R2_physical": r2_physical,
        "R2_log10": np.nan,
        "MAPE_percent": _safe_mape(y_true, y_pred),
        "sMAPE_percent": _smape(y_true, y_pred),
        "WAPE_percent": _wape(y_true, y_pred),
        "Median_APE_percent": float(np.median(ape) * 100.0),
        "P95_APE_percent": float(np.quantile(ape, 0.95) * 100.0),
        "Bias": float(np.mean(error)),
        "P95_absolute_error": float(np.quantile(abs_error, 0.95)),
        "Negative_prediction_count": int(np.sum(y_pred < 0)),
    }
    if label.startswith("Rate_") or label == "Z":
        if np.all(y_true > 0) and np.all(y_pred > 0):
            true_log = np.log10(y_true)
            pred_log = np.log10(y_pred)
            log_error = pred_log - true_log
            log_denominator = np.sum((true_log - np.mean(true_log)) ** 2)
            row["Log10_MAE"] = float(np.mean(np.abs(log_error)))
            row["Log10_RMSE"] = float(
                np.sqrt(np.mean(log_error**2))
            )
            row["R2_log10"] = (
                float(1.0 - np.sum(log_error**2) / log_denominator)
                if log_denominator > 0
                else np.nan
            )
        else:
            row["Log10_MAE"] = np.nan
            row["Log10_RMSE"] = np.nan
    return row


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    rows = [parameter_metrics(y_true[:, i], y_pred[:, i], label) for i, label in enumerate(labels)]

    location_error = None
    if "X" in labels and "Y" in labels:
        ix, iy = labels.index("X"), labels.index("Y")
        location_error = np.sqrt((y_pred[:, ix] - y_true[:, ix]) ** 2 + (y_pred[:, iy] - y_true[:, iy]) ** 2)
        rows.append(
            {
                "Parameter": "XY_DISTANCE_km",
                "MAE": float(np.mean(location_error)),
                "RMSE": float(np.sqrt(np.mean(location_error**2))),
                "R2": np.nan,
                "R2_physical": np.nan,
                "R2_log10": np.nan,
                "MAPE_percent": np.nan,
                "sMAPE_percent": np.nan,
                "WAPE_percent": np.nan,
                "Median_APE_percent": np.nan,
                "P95_APE_percent": np.nan,
                "Bias": np.nan,
                "P95_absolute_error": float(np.quantile(location_error, 0.95)),
                "Negative_prediction_count": 0,
                "Median_distance_km": float(np.median(location_error)),
                "Max_distance_km": float(np.max(location_error)),
            }
        )
    metrics = pd.DataFrame(rows)

    intervals: list[dict] = []
    if bootstrap_repeats > 0:
        rng = np.random.default_rng(seed + 9000)
        n = len(y_true)
        for i, label in enumerate(labels):
            values = np.abs(y_pred[:, i] - y_true[:, i])
            boot = np.empty(bootstrap_repeats, dtype=np.float64)
            for b in range(bootstrap_repeats):
                sample = rng.integers(0, n, size=n)
                boot[b] = np.mean(values[sample])
            intervals.append(
                {
                    "Parameter": label,
                    "Metric": "MAE",
                    "Estimate": float(np.mean(values)),
                    "CI95_low": float(np.quantile(boot, 0.025)),
                    "CI95_high": float(np.quantile(boot, 0.975)),
                    "Bootstrap_repeats": int(bootstrap_repeats),
                }
            )
            if (label.startswith("Rate_") or label == "Z") and np.all(y_pred[:, i] > 0):
                log_values = np.abs(np.log10(y_pred[:, i]) - np.log10(y_true[:, i]))
                for b in range(bootstrap_repeats):
                    sample = rng.integers(0, n, size=n)
                    boot[b] = np.mean(log_values[sample])
                intervals.append(
                    {
                        "Parameter": label,
                        "Metric": "Log10_MAE",
                        "Estimate": float(np.mean(log_values)),
                        "CI95_low": float(np.quantile(boot, 0.025)),
                        "CI95_high": float(np.quantile(boot, 0.975)),
                        "Bootstrap_repeats": int(bootstrap_repeats),
                    }
                )
        if location_error is not None:
            for metric_name, statistic in [
                ("Mean_distance_km", np.mean),
                ("Median_distance_km", np.median),
                ("P95_distance_km", lambda value: np.quantile(value, 0.95)),
            ]:
                boot = np.empty(bootstrap_repeats, dtype=np.float64)
                for b in range(bootstrap_repeats):
                    sample = rng.integers(0, n, size=n)
                    boot[b] = statistic(location_error[sample])
                intervals.append(
                    {
                        "Parameter": "XY_DISTANCE_km",
                        "Metric": metric_name,
                        "Estimate": float(statistic(location_error)),
                        "CI95_low": float(np.quantile(boot, 0.025)),
                        "CI95_high": float(np.quantile(boot, 0.975)),
                        "Bootstrap_repeats": int(bootstrap_repeats),
                    }
                )

    return metrics, pd.DataFrame(intervals)


def sample_error_table(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    result = pd.DataFrame()
    for i, label in enumerate(labels):
        result[f"AE_{label}"] = np.abs(y_pred[:, i] - y_true[:, i])
        if (label.startswith("Rate_") or label == "Z") and np.all(y_pred[:, i] > 0):
            result[f"LOGAE_{label}"] = np.abs(np.log10(y_pred[:, i]) - np.log10(y_true[:, i]))
    if "X" in labels and "Y" in labels:
        ix, iy = labels.index("X"), labels.index("Y")
        result["XY_DISTANCE_km"] = np.sqrt(
            (y_pred[:, ix] - y_true[:, ix]) ** 2 + (y_pred[:, iy] - y_true[:, iy]) ** 2
        )
    return result
