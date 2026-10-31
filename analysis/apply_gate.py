from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_validate_gate import FEATURES, predict_risk, prepare_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a locked T-ADRG gate to truth-free indicator rows."
    )
    parser.add_argument("--indicator-csv", type=Path, required=True)
    parser.add_argument("--gate-parameters", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.indicator_csv.resolve())
    if "task" not in frame:
        raise ValueError("Indicator CSV must contain a task column.")
    with args.gate_parameters.resolve().open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    if list(parameters["features"]) != list(FEATURES):
        raise ValueError("Feature order in gate parameters is incompatible with this script.")

    result_parts = []
    for task, part in frame.groupby("task", sort=False):
        if task not in parameters["tasks"]:
            raise ValueError("No locked gate parameters for task: %s" % task)
        task_parameters = parameters["tasks"][task]
        model_data = task_parameters["model"]
        model = {
            "mean": np.asarray(model_data["mean"], dtype=float),
            "scale": np.asarray(model_data["scale"], dtype=float),
            "weights": np.asarray(model_data["weights"], dtype=float),
            "intercept": float(model_data["intercept"]),
        }
        part = part.copy()
        part["predicted_failure_risk"] = predict_risk(prepare_features(part), model)
        threshold = task_parameters["risk_threshold"]
        part["gate_feasible"] = bool(task_parameters["gate_feasible"])
        part["risk_threshold"] = np.nan if threshold is None else float(threshold)
        part["accepted"] = (
            part["predicted_failure_risk"] <= float(threshold)
            if threshold is not None and task_parameters["gate_feasible"] else False
        )
        result_parts.append(part)
    output = pd.concat(result_parts, ignore_index=True)
    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv.resolve(), index=False, encoding="utf-8-sig")
    print("Wrote %d prospective gate decisions to %s" % (len(output), args.output_csv.resolve()))


if __name__ == "__main__":
    main()
