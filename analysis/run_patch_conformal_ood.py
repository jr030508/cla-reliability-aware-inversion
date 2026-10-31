from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("release_rate", "horizontal_location", "release_height")


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="No-retraining conformal and OOD patch experiments.")
    parser.add_argument("--run-root", type=Path, default=here.parent / "outputs_tadrg_full")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--local-k", type=int, default=150)
    return parser.parse_args()


def finite_quantile(values, coverage):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        raise ValueError("Cannot calibrate an empty stratum.")
    rank = min(max(int(np.ceil((len(values) + 1) * coverage)), 1), len(values))
    return float(values[rank - 1])


def assign_bins(values, edges):
    return np.digitize(np.asarray(values, dtype=float), np.asarray(edges[1:-1], dtype=float), right=False)


def mondrian_width(calibration, validation, conditioning, edges, denominator, coverage, min_bin=30):
    cal_bin = assign_bins(calibration[conditioning], edges)
    val_bin = assign_bins(validation[conditioning], edges)
    global_q = finite_quantile(calibration["score"], coverage)
    q_by_bin = {}
    n_by_bin = {}
    for bin_number in range(len(edges) - 1):
        scores = calibration.loc[cal_bin == bin_number, "score"].to_numpy()
        n_by_bin[bin_number] = int(len(scores))
        q_by_bin[bin_number] = finite_quantile(scores, coverage) if len(scores) >= min_bin else global_q
    q = np.asarray([q_by_bin[int(value)] for value in val_bin], dtype=float)
    return q * denominator, q_by_bin, n_by_bin


def local_width(calibration, validation, denominator, coverage, k):
    cal_risk = calibration["predicted_risk"].to_numpy(dtype=float)
    cal_score = calibration["score"].to_numpy(dtype=float)
    val_risk = validation["predicted_risk"].to_numpy(dtype=float)
    k = min(max(int(k), 30), len(calibration))
    widths = np.empty(len(validation), dtype=float)
    for i, risk in enumerate(val_risk):
        nearest = np.argpartition(np.abs(cal_risk - risk), k - 1)[:k]
        widths[i] = finite_quantile(cal_score[nearest], coverage) * denominator[i]
    return widths


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from fit_validate_gate import predict_risk, prepare_features

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    indicators = pd.read_csv(run_root / "indicators" / "tadrg_indicators.csv")
    calibration_all = indicators[indicators.role == "calibration"].copy()
    validation_all = pd.read_csv(run_root / "gate" / "validation_predictions.csv")
    parameters = json.loads((run_root / "gate" / "gate_parameters.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "indicators" / "indicator_manifest.json").read_text(encoding="utf-8"))

    coverage_rows = []
    parameter_rows = []
    rejection_rows = []
    patched_parts = []
    for task in TASKS:
        item = parameters["tasks"][task]
        model_data = item["model"]
        model = {
            "mean": np.asarray(model_data["mean"], dtype=float),
            "scale": np.asarray(model_data["scale"], dtype=float),
            "weights": np.asarray(model_data["weights"], dtype=float),
            "intercept": float(model_data["intercept"]),
        }
        calibration = calibration_all[calibration_all.task == task].copy()
        calibration["predicted_risk"] = predict_risk(prepare_features(calibration), model)
        threshold_ids = set(int(x) for x in item["threshold_sample_ids"])
        calibration = calibration[calibration.sample_index.isin(threshold_ids)].copy()
        validation = validation_all[validation_all.task == task].copy()
        floor = float(manifest["conformal"][task]["floor"])
        calibration["denominator"] = calibration.ensemble_uncertainty + floor
        calibration["score"] = calibration.error / calibration.denominator
        denominator = validation.ensemble_uncertainty.to_numpy(dtype=float) + floor

        global_q = finite_quantile(calibration.score, args.coverage)
        widths = {
            "existing_global": validation.conformal_half_width.to_numpy(dtype=float),
            "recalibrated_global": global_q * denominator,
        }

        missing_edges = [-np.inf, 0.01, 0.15, 0.40, np.inf]
        calibration["missing_fraction"] = 1.0 - calibration.availability_fraction
        validation["missing_fraction"] = 1.0 - validation.availability_fraction
        width_missing, q_missing, n_missing = mondrian_width(
            calibration, validation, "missing_fraction", missing_edges,
            denominator, args.coverage
        )
        widths["mondrian_missing"] = width_missing

        risk_quantiles = np.unique(calibration.predicted_risk.quantile([0.0, 0.25, 0.50, 0.75, 1.0]).to_numpy())
        if len(risk_quantiles) < 3:
            risk_quantiles = np.asarray([-np.inf, np.inf])
        else:
            risk_quantiles[0] = -np.inf
            risk_quantiles[-1] = np.inf
        width_risk, q_risk, n_risk = mondrian_width(
            calibration, validation, "predicted_risk", risk_quantiles,
            denominator, args.coverage
        )
        widths["mondrian_risk_q4"] = width_risk
        widths["local_risk_k%d" % args.local_k] = local_width(
            calibration, validation, denominator, args.coverage, args.local_k
        )

        for method, width in widths.items():
            patched = validation[["sample_index", "task", "condition", "error", "predicted_risk"]].copy()
            patched["method"] = method
            patched["half_width"] = width
            patched["covered"] = patched.error <= patched.half_width
            patched["width_ratio_to_existing"] = width / np.maximum(validation.conformal_half_width, 1e-12)
            patched_parts.append(patched)
            for condition, part in patched.groupby("condition", sort=False):
                coverage_rows.append({
                    "task": task, "method": method, "condition": condition,
                    "samples": int(len(part)), "coverage": float(part.covered.mean()),
                    "mean_half_width": float(part.half_width.mean()),
                    "median_width_ratio_to_existing": float(part.width_ratio_to_existing.median()),
                })

        parameter_rows.extend([
            {"task": task, "method": "recalibrated_global", "stratum": "all",
             "n": int(len(calibration)), "q": global_q},
            *({"task": task, "method": "mondrian_missing", "stratum": str(key),
               "n": n_missing[key], "q": q_missing[key]} for key in q_missing),
            *({"task": task, "method": "mondrian_risk_q4", "stratum": str(key),
               "n": n_risk[key], "q": q_risk[key]} for key in q_risk),
        ])

        # Learned OOD threshold: 90th percentile of independent calibration-threshold
        # rows under clean or mild bias. It is never tuned on validation data.
        reference = calibration[calibration.condition.isin(["clean", "bias_0.05"])]
        learned_threshold = float(reference.predicted_risk.quantile(0.90))
        validation["learned_reject"] = validation.predicted_risk > learned_threshold
        # Protocol gate uses known acquisition metadata. LOD > 1% or any explicit
        # station/time unavailability is outside the declared clean/light-bias scope.
        validation["protocol_reject"] = (
            (validation.availability_fraction < 1.0 - 1e-12)
            | (validation.detection_quantile > 0.01 + 1e-12)
        )
        validation["hybrid_reject"] = validation.learned_reject | validation.protocol_reject
        for condition, part in validation.groupby("condition", sort=False):
            rejection_rows.append({
                "task": task, "condition": condition, "samples": int(len(part)),
                "learned_risk_threshold": learned_threshold,
                "learned_rejection_rate": float(part.learned_reject.mean()),
                "protocol_rejection_rate": float(part.protocol_reject.mean()),
                "hybrid_rejection_rate": float(part.hybrid_reject.mean()),
            })

    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "conformal_condition_coverage.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "conformal_calibration_parameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rejection_rows).to_csv(
        output_dir / "ood_rejection_by_condition.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(patched_parts, ignore_index=True).to_csv(
        output_dir / "patched_conformal_predictions.csv", index=False, encoding="utf-8-sig"
    )
    # Engineering contrast: quantify how much coverage loss and detection-limit
    # truncation increase error relative to a 20% persistent multiplicative bias.
    rng = np.random.default_rng(20261028)
    contrast_rows = []
    for task, task_frame in validation_all.groupby("task", sort=False):
        reference = task_frame.loc[task_frame.condition == "bias_0.20", "error"].to_numpy(dtype=float)
        for condition in ("station_32", "time_3", "lod_q10"):
            exposed = task_frame.loc[task_frame.condition == condition, "error"].to_numpy(dtype=float)
            ratios = np.empty(3000, dtype=float)
            for repeat in range(len(ratios)):
                ratios[repeat] = (
                    rng.choice(exposed, len(exposed), replace=True).mean()
                    / rng.choice(reference, len(reference), replace=True).mean()
                )
            superiority = float(np.mean(exposed[:, None] > reference[None, :]))
            ties = float(np.mean(exposed[:, None] == reference[None, :]))
            contrast_rows.append({
                "task": task, "condition": condition, "reference": "bias_0.20",
                "mean_error_ratio": float(exposed.mean() / reference.mean()),
                "bootstrap_ratio_ci_low": float(np.quantile(ratios, 0.025)),
                "bootstrap_ratio_ci_high": float(np.quantile(ratios, 0.975)),
                "probability_exposed_error_greater": superiority + 0.5 * ties,
            })
    pd.DataFrame(contrast_rows).to_csv(
        output_dir / "coverage_vs_bias_error_contrast.csv", index=False, encoding="utf-8-sig"
    )
    print("Patch experiment outputs written to %s" % output_dir)


if __name__ == "__main__":
    main()
