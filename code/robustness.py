from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import load_config, load_dataset, resolve_from_config
from metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated physically interpretable robustness tests")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout-rates", nargs="+", type=float)
    parser.add_argument("--lognormal-sigmas", nargs="+", type=float)
    parser.add_argument("--design", choices=["one_factor", "factorial"], default="one_factor")
    parser.add_argument("--max-test-samples", type=int, help="Pilot only; omit for the final full test set")
    return parser.parse_args()


def perturb(
    X: np.ndarray,
    dropout_rate: float,
    lognormal_sigma: float,
    detection_limit: float,
    background_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = X.astype(np.float64, copy=True)
    if lognormal_sigma > 0:
        multiplier = np.exp(rng.normal(0.0, lognormal_sigma, size=out.shape) - 0.5 * lognormal_sigma**2)
        out *= multiplier
    if background_sd > 0:
        out += rng.normal(0.0, background_sd, size=out.shape)
    out = np.maximum(out, 0.0)
    if detection_limit > 0:
        out[out < detection_limit] = 0.0
    if dropout_rate > 0:
        # Independent station failure mask for every sample; failure persists over all time steps.
        keep = rng.random((len(out), 1, out.shape[-1])) >= dropout_rate
        out *= keep
    return out.astype(np.float32)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    config_path = args.config.resolve()
    config = load_config(config_path)
    scenario = str(manifest["scenario"])
    scenario_cfg = config["scenarios"][scenario]
    robustness_cfg = config["robustness"]
    repeats = args.repeats or int(robustness_cfg["repeats"])

    manifest_data_path = manifest.get("data", {}).get("data_path")
    data_path = Path(manifest_data_path) if manifest_data_path else resolve_from_config(config_path, scenario_cfg["data"])
    if not data_path.exists():
        data_path = resolve_from_config(config_path, scenario_cfg["data"])
    bundle = load_dataset(
        data_path,
        int(config["time_steps"]),
        int(config["feature_dim"]),
        list(scenario_cfg["labels"]),
    )
    split = np.load(run_dir / "split_indices.npz")
    test_idx = split["test"]
    permutation = split["station_permutation"]
    X_test_raw = bundle.X[test_idx][:, :, permutation]
    y_true = bundle.y[test_idx]
    if args.max_test_samples:
        X_test_raw = X_test_raw[: args.max_test_samples]
        y_true = y_true[: args.max_test_samples]
    with (run_dir / "scalers.pkl").open("rb") as handle:
        transformers = pickle.load(handle)

    import tensorflow as tf
    from models import FixedGraphPropagation

    model_path = run_dir / "model.h5"
    if not model_path.exists():
        model_path = run_dir / "model.keras"
    model = tf.keras.models.load_model(
        str(model_path), custom_objects={"FixedGraphPropagation": FixedGraphPropagation}, compile=False
    )

    records = []
    base_seed = int(manifest["seed"])
    dropout_rates = args.dropout_rates or robustness_cfg["dropout_rates"]
    sigmas = args.lognormal_sigmas or robustness_cfg["lognormal_sigmas"]
    if args.design == "factorial":
        conditions = [(float(dropout), float(sigma)) for dropout in dropout_rates for sigma in sigmas]
    else:
        conditions = {(float(dropout), 0.0) for dropout in dropout_rates}
        conditions |= {(0.0, float(sigma)) for sigma in sigmas}
        conditions = sorted(conditions)
    for dropout_rate, sigma in conditions:
        current_repeats = 1 if dropout_rate == 0 and sigma == 0 else repeats
        for repeat in range(current_repeats):
            rng = np.random.default_rng(base_seed * 100000 + repeat * 100 + int(dropout_rate * 1000) + int(sigma * 10000))
            X_perturbed = perturb(
                X_test_raw,
                dropout_rate,
                sigma,
                float(robustness_cfg["detection_limit"]),
                float(robustness_cfg["background_sd"]),
                rng,
            )
            X_scaled = transformers["input"].transform(X_perturbed)
            pred_scaled = model.predict(X_scaled, batch_size=args.batch_size, verbose=0)
            y_pred = transformers["target"].inverse_transform(pred_scaled)
            metrics, _ = evaluate_predictions(y_true, y_pred, transformers["labels"], 0, base_seed + repeat)
            for row in metrics.to_dict("records"):
                records.append(
                    {
                        "Dropout_rate": dropout_rate,
                        "Lognormal_sigma": sigma,
                        "Repeat": repeat,
                        **row,
                    }
                )

    result = pd.DataFrame(records)
    result.to_csv(run_dir / "robustness_replicates.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in ["MAE", "RMSE", "R2", "MAPE_percent", "sMAPE_percent", "WAPE_percent", "P95_absolute_error", "Log10_MAE"] if c in result]
    summary = (
        result.groupby(["Dropout_rate", "Lognormal_sigma", "Parameter"])[numeric]
        .agg(["mean", "std", "median", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)]).replace("<lambda_0>", "q025").replace("<lambda_1>", "q975")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    summary.to_csv(run_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Robustness results written to {run_dir}")


if __name__ == "__main__":
    main()
