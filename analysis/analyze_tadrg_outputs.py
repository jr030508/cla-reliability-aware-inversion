from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Audit completed T-ADRG outputs without refitting the gate.")
    parser.add_argument("--run-root", type=Path, default=here / "outputs_tadrg_full")
    parser.add_argument("--output-dir", type=Path, default=here / "analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from fit_validate_gate import average_precision, binary_auc

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indicators = pd.read_csv(run_root / "indicators" / "tadrg_indicators.csv")
    validation = pd.read_csv(run_root / "gate" / "validation_predictions.csv")
    summary = pd.read_csv(run_root / "gate" / "gate_validation_summary.csv")
    parameters = json.loads((run_root / "gate" / "gate_parameters.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "indicators" / "indicator_manifest.json").read_text(encoding="utf-8"))

    condition_coverage = (
        validation.assign(conformal_covered=validation.error <= validation.conformal_half_width)
        .groupby(["task", "condition"], as_index=False)
        .agg(
            conformal_coverage=("conformal_covered", "mean"),
            conformal_mean_width=("conformal_half_width", "mean"),
            median_predicted_risk=("predicted_risk", "median"),
        )
    )
    condition_audit = summary[summary.condition != "ALL"].merge(
        condition_coverage, on=["task", "condition"], how="left", validate="one_to_one"
    )
    condition_audit["accepted_failures"] = np.rint(
        condition_audit.accepted * condition_audit.failure_rate_accepted.fillna(0.0)
    ).astype(int)
    condition_audit["passes_10pct_wilson"] = condition_audit.wilson_upper_accepted <= 0.10
    condition_audit.to_csv(output_dir / "condition_audit.csv", index=False, encoding="utf-8-sig")

    scopes = {
        "all_conditions": None,
        "non_extreme": ["clean", "bias_0.10", "bias_0.20", "lod_q05", "lod_q10"],
        "clean_and_bias": ["clean", "bias_0.10", "bias_0.20"],
    }
    discrimination = []
    for task, task_frame in validation.groupby("task", sort=False):
        for scope, conditions in scopes.items():
            part = task_frame if conditions is None else task_frame[task_frame.condition.isin(conditions)]
            y = part.failure.to_numpy(dtype=int)
            score = part.predicted_risk.to_numpy(dtype=float)
            discrimination.append({
                "task": task,
                "scope": scope,
                "samples": int(len(part)),
                "failure_prevalence": float(np.mean(y)),
                "auroc": binary_auc(y, score),
                "auprc": average_precision(y, score),
            })
    pd.DataFrame(discrimination).to_csv(
        output_dir / "discrimination_scope_audit.csv", index=False, encoding="utf-8-sig"
    )

    gate_rows = []
    for task, item in parameters["tasks"].items():
        selected = item.get("selected_threshold_diagnostics") or {}
        best = item.get("best_available_diagnostics") or {}
        gate_rows.append({
            "task": task,
            "tolerance": item["tolerance"],
            "tolerance_source": item["tolerance_source"],
            "gate_feasible": item["gate_feasible"],
            "risk_threshold": item["risk_threshold"],
            "calibration_accepted": selected.get("accepted"),
            "calibration_failures": selected.get("failures"),
            "calibration_failure_rate": selected.get("failure_rate"),
            "calibration_wilson_upper": selected.get("wilson_upper"),
            "best_available_threshold": best.get("threshold"),
            "best_available_accepted": best.get("accepted"),
            "best_available_wilson_upper": best.get("wilson_upper"),
        })
    pd.DataFrame(gate_rows).to_csv(
        output_dir / "gate_calibration_audit.csv", index=False, encoding="utf-8-sig"
    )

    redundancy_rows = []
    for task, part in validation.groupby("task", sort=False):
        redundancy_rows.append({
            "task": task,
            "pearson_ensemble_uncertainty_vs_conformal_width": float(
                np.corrcoef(part.ensemble_uncertainty, part.conformal_half_width)[0, 1]
            )
        })
    pd.DataFrame(redundancy_rows).to_csv(
        output_dir / "feature_redundancy_audit.csv", index=False, encoding="utf-8-sig"
    )

    calibration_ids = set(indicators.loc[indicators.role == "calibration", "sample_index"].astype(int))
    validation_ids = set(indicators.loc[indicators.role == "validation", "sample_index"].astype(int))
    integrity = {
        "indicator_rows": int(len(indicators)),
        "validation_rows": int(len(validation)),
        "indicator_missing_cells": int(indicators.isna().sum().sum()),
        "validation_missing_cells": int(validation.isna().sum().sum()),
        "calibration_unique_samples": int(len(calibration_ids)),
        "validation_unique_samples": int(len(validation_ids)),
        "calibration_validation_overlap": int(len(calibration_ids & validation_ids)),
        "duplicate_sample_task_rows": int(
            indicators.duplicated(["role", "sample_index", "task"], keep=False).sum()
        ),
        "manifest": manifest,
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("T-ADRG audit written to %s" % output_dir)


if __name__ == "__main__":
    main()
