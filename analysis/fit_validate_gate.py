from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("release_rate", "horizontal_location", "release_height")
FEATURES = (
    "missing_fraction",
    "weak_signal_fraction",
    "log_latent_ood",
    "log_ensemble_uncertainty",
    "log_conformal_half_width",
    "log_local_sensitivity",
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Fit and independently validate the T-ADRG reliability gate.")
    parser.add_argument("--indicator-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=here / "config_tadrg.json")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs" / "gate")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def prepare_features(frame):
    transformed = pd.DataFrame(index=frame.index)
    transformed["missing_fraction"] = 1.0 - frame["availability_fraction"].astype(float)
    transformed["weak_signal_fraction"] = 1.0 - frame["signal_support_fraction"].astype(float)
    transformed["log_latent_ood"] = np.log1p(np.maximum(frame["latent_ood"].astype(float), 0.0))
    transformed["log_ensemble_uncertainty"] = np.log1p(
        np.maximum(frame["ensemble_uncertainty"].astype(float), 0.0)
    )
    transformed["log_conformal_half_width"] = np.log1p(
        np.maximum(frame["conformal_half_width"].astype(float), 0.0)
    )
    transformed["log_local_sensitivity"] = np.log1p(
        np.maximum(frame["local_sensitivity"].astype(float), 0.0)
    )
    return transformed.loc[:, FEATURES].to_numpy(dtype=np.float64)


def fit_monotone_logistic(X, y, gate_config):
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = (X - mean) / scale
    weights = np.zeros(Z.shape[1], dtype=np.float64)
    prevalence = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
    intercept = float(np.log(prevalence / (1.0 - prevalence)))
    learning_rate = float(gate_config["learning_rate"])
    l2 = float(gate_config["l2"])
    for step in range(int(gate_config["iterations"])):
        probability = sigmoid(intercept + Z.dot(weights))
        residual = probability - y
        gradient_w = Z.T.dot(residual) / len(y) + l2 * weights
        gradient_b = float(np.mean(residual))
        rate = learning_rate / np.sqrt(1.0 + step / 500.0)
        weights = np.maximum(weights - rate * gradient_w, 0.0)
        intercept -= rate * gradient_b
    return {"mean": mean, "scale": scale, "weights": weights, "intercept": intercept}


def predict_risk(X, model):
    Z = (X - model["mean"]) / model["scale"]
    return sigmoid(model["intercept"] + Z.dot(model["weights"]))


def wilson_upper(failures, total, z):
    if total <= 0:
        return float("nan")
    p = float(failures) / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center + radius) / denominator)


def select_threshold(risk, failures, gate_config):
    target = float(gate_config["target_failure_rate"])
    z = float(gate_config["confidence_z"])
    minimum = int(gate_config["minimum_accepted"])
    candidates = np.unique(risk)
    best = None
    diagnostics = []
    for threshold in candidates:
        accepted = risk <= threshold
        total = int(np.sum(accepted))
        if total < minimum:
            continue
        count = int(np.sum(failures[accepted]))
        upper = wilson_upper(count, total, z)
        item = {
            "threshold": float(threshold), "accepted": total, "failures": count,
            "failure_rate": float(count / total), "wilson_upper": upper,
        }
        diagnostics.append(item)
        if upper <= target and (best is None or total > best["accepted"]):
            best = item
    if best is None:
        fallback = min(diagnostics, key=lambda item: item["wilson_upper"]) if diagnostics else None
        return None, fallback
    return best, min(diagnostics, key=lambda item: item["wilson_upper"])


def binary_auc(y, score):
    y = np.asarray(y, dtype=int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y, score):
    y = np.asarray(y, dtype=int)
    positives = int(np.sum(y))
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    ordered = y[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / positives)


def risk_coverage_curve(frame, z):
    ordered = frame.sort_values("predicted_risk", kind="mergesort").reset_index(drop=True)
    grid = np.unique(np.linspace(1, len(ordered), min(101, len(ordered)), dtype=int))
    rows = []
    for count in grid:
        accepted = ordered.iloc[:count]
        failures = int(accepted["failure"].sum())
        rows.append({
            "coverage": float(count / len(ordered)),
            "accepted": int(count),
            "failure_rate": float(failures / count),
            "wilson_upper": wilson_upper(failures, count, z),
            "mean_error": float(accepted["error"].mean()),
            "risk_threshold": float(accepted["predicted_risk"].max()),
        })
    return pd.DataFrame(rows)


def summarize_validation(frame, z):
    rows = []
    for condition, part in frame.groupby("condition", sort=False):
        accepted = part[part["accepted"]]
        failures = int(accepted["failure"].sum())
        rows.append({
            "condition": condition,
            "samples": int(len(part)),
            "accepted": int(len(accepted)),
            "acceptance_rate": float(len(accepted) / len(part)),
            "failure_rate_all": float(part["failure"].mean()),
            "failure_rate_accepted": float(accepted["failure"].mean()) if len(accepted) else np.nan,
            "wilson_upper_accepted": wilson_upper(failures, len(accepted), z) if len(accepted) else np.nan,
            "mean_error_all": float(part["error"].mean()),
            "mean_error_accepted": float(accepted["error"].mean()) if len(accepted) else np.nan,
        })
    return rows


def main():
    args = parse_args()
    config = load_json(args.config.resolve())
    frame = pd.read_csv(args.indicator_csv.resolve())
    required = {
        "role", "sample_index", "condition", "task", "error", "availability_fraction",
        "signal_support_fraction", "latent_ood", "ensemble_uncertainty",
        "conformal_half_width", "local_sensitivity",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Indicator CSV is missing columns: %s" % missing)
    if set(frame["role"].unique()) != {"calibration", "validation"}:
        raise ValueError("Both calibration and validation roles are required.")
    overlap = set(frame.loc[frame.role == "calibration", "sample_index"]) & set(
        frame.loc[frame.role == "validation", "sample_index"]
    )
    if overlap:
        raise ValueError("Calibration/validation leakage: %d sample indices overlap." % len(overlap))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_cfg = config["gate"]
    tolerance_cfg = config["tolerance"]
    rng = np.random.default_rng(int(config["random_seed"]) + 77)
    parameters = {
        "method": "T-ADRG v1", "features": list(FEATURES),
        "tolerance_mode": tolerance_cfg["mode"], "tasks": {},
    }
    validation_parts = []
    summary_rows = []
    coefficient_rows = []
    curve_parts = []

    for task in TASKS:
        cal = frame[(frame.role == "calibration") & (frame.task == task)].copy()
        val = frame[(frame.role == "validation") & (frame.task == task)].copy()
        sample_ids = np.asarray(sorted(cal["sample_index"].unique()))
        shuffled = rng.permutation(sample_ids)
        n_fit = int(round(len(shuffled) * float(gate_cfg["fit_fraction"])))
        n_fit = min(max(n_fit, 1), len(shuffled) - 1)
        fit_ids = set(shuffled[:n_fit].tolist())
        fit_part = cal[cal.sample_index.isin(fit_ids)].copy()
        threshold_part = cal[~cal.sample_index.isin(fit_ids)].copy()

        if tolerance_cfg["mode"] == "absolute":
            tolerance = tolerance_cfg["absolute"].get(task)
            if tolerance is None:
                raise ValueError("Absolute tolerance for %s is null." % task)
            tolerance = float(tolerance)
            tolerance_source = "user_defined_absolute"
        elif tolerance_cfg["mode"] == "calibration_quantile":
            clean_fit = fit_part[fit_part.condition == "clean"]
            if clean_fit.empty:
                raise ValueError("Clean calibration rows are required for exploratory tolerances.")
            tolerance = float(clean_fit["error"].quantile(float(tolerance_cfg["quantile"])))
            tolerance_source = "clean_gate_fit_quantile"
        else:
            raise ValueError("Unknown tolerance mode: %s" % tolerance_cfg["mode"])

        fit_part["failure"] = (fit_part["error"] > tolerance).astype(int)
        threshold_part["failure"] = (threshold_part["error"] > tolerance).astype(int)
        val["failure"] = (val["error"] > tolerance).astype(int)
        if fit_part["failure"].nunique() < 2:
            raise ValueError("Task %s has only one class in gate-fit data; change tolerance or conditions." % task)

        model = fit_monotone_logistic(prepare_features(fit_part), fit_part["failure"].to_numpy(), gate_cfg)
        threshold_risk = predict_risk(prepare_features(threshold_part), model)
        threshold_part["predicted_risk"] = threshold_risk
        selected, best_diagnostic = select_threshold(
            threshold_risk, threshold_part["failure"].to_numpy(dtype=bool), gate_cfg
        )
        feasible = selected is not None
        risk_threshold = float(selected["threshold"]) if feasible else None

        val["predicted_risk"] = predict_risk(prepare_features(val), model)
        val["accepted"] = val["predicted_risk"] <= risk_threshold if feasible else False
        val["tolerance"] = tolerance
        val["gate_feasible"] = feasible
        validation_parts.append(val)

        auc = binary_auc(val["failure"].to_numpy(), val["predicted_risk"].to_numpy())
        ap = average_precision(val["failure"].to_numpy(), val["predicted_risk"].to_numpy())
        task_rows = summarize_validation(val, float(gate_cfg["confidence_z"]))
        accepted = val[val.accepted]
        failures = int(accepted.failure.sum())
        overall = {
            "task": task,
            "condition": "ALL",
            "samples": int(len(val)),
            "accepted": int(len(accepted)),
            "acceptance_rate": float(len(accepted) / len(val)),
            "failure_rate_all": float(val.failure.mean()),
            "failure_rate_accepted": float(accepted.failure.mean()) if len(accepted) else np.nan,
            "wilson_upper_accepted": wilson_upper(failures, len(accepted), float(gate_cfg["confidence_z"]))
            if len(accepted) else np.nan,
            "mean_error_all": float(val.error.mean()),
            "mean_error_accepted": float(accepted.error.mean()) if len(accepted) else np.nan,
            "failure_detection_auroc": auc,
            "failure_detection_auprc": ap,
            "tolerance": tolerance,
            "gate_feasible_on_calibration": feasible,
        }
        summary_rows.append(overall)
        for item in task_rows:
            item.update({
                "task": task, "failure_detection_auroc": auc,
                "failure_detection_auprc": ap, "tolerance": tolerance,
                "gate_feasible_on_calibration": feasible,
            })
            summary_rows.append(item)

        curve = risk_coverage_curve(val, float(gate_cfg["confidence_z"]))
        curve.insert(0, "task", task)
        curve_parts.append(curve)
        for feature, mean, scale, weight in zip(FEATURES, model["mean"], model["scale"], model["weights"]):
            coefficient_rows.append({
                "task": task, "feature": feature, "mean": float(mean),
                "scale": float(scale), "nonnegative_weight": float(weight),
            })

        parameters["tasks"][task] = {
            "error_definition": {
                "release_rate": "mean absolute log10 error across Rate_1...Rate_4",
                "horizontal_location": "Euclidean error in km",
                "release_height": "absolute log10 error of Z",
            }[task],
            "tolerance": tolerance,
            "tolerance_source": tolerance_source,
            "fit_sample_ids": sorted(int(x) for x in fit_ids),
            "threshold_sample_ids": sorted(int(x) for x in set(sample_ids.tolist()) - fit_ids),
            "model": {
                "mean": model["mean"].tolist(), "scale": model["scale"].tolist(),
                "weights": model["weights"].tolist(), "intercept": float(model["intercept"]),
            },
            "risk_threshold": risk_threshold,
            "gate_feasible": feasible,
            "selected_threshold_diagnostics": selected,
            "best_available_diagnostics": best_diagnostic,
        }

    validation = pd.concat(validation_parts, ignore_index=True)
    validation.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "gate_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(curve_parts, ignore_index=True).to_csv(
        output_dir / "risk_coverage.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(coefficient_rows).to_csv(
        output_dir / "indicator_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "gate_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_summary = pd.DataFrame(summary_rows)
    headline = all_summary[all_summary.condition == "ALL"].copy()
    columns = [
        "task", "acceptance_rate", "failure_rate_all", "failure_rate_accepted",
        "wilson_upper_accepted", "failure_detection_auroc", "failure_detection_auprc",
        "tolerance", "gate_feasible_on_calibration",
    ]
    lines = [
        "# T-ADRG 独立验证摘要", "",
        "本文件由脚本自动生成。只有 `gate_feasible_on_calibration=True` 且独立验证结果满足预设失败率上限，才可声称门控有效。",
        "", "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in headline.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(("%.5g" % value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "注意：当前门控不含气象变量，不能用于声称跨气象事件泛化。"])
    (output_dir / "验证摘要.md").write_text("\n".join(lines), encoding="utf-8")
    print("Gate outputs written to %s" % output_dir)


if __name__ == "__main__":
    main()
