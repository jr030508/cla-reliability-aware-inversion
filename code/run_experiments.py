from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import (
    InputTransformer,
    TargetTransformer,
    build_normalized_adjacency,
    dataset_summary,
    load_config,
    load_dataset,
    load_station_coordinates,
    resolve_from_config,
    sha256_file,
    split_indices,
    station_permutation,
)
from metrics import evaluate_predictions, sample_error_table


VALID_MODELS = [
    "CLA",
    "CLA_NoAttention",
    "CLA_NoStationMix",
    "LSTM",
    "LSTM_Attention",
    "CNN_LSTM",
    "2D_CNN",
    "2D_CNN_Attention",
    "Graph_CLA",
]

# Must be set before TensorFlow is imported. TF 2.6 reads this environment flag early.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible CNN-LSTM-A experiments")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data-file", type=Path, help="Override the scenario CSV path from config.json")
    parser.add_argument("--scenario", choices=["1", "2", "3"], required=True)
    parser.add_argument("--models", nargs="+", choices=VALID_MODELS, default=["CLA"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--split", choices=["iid", "group_event", "spatial_block", "parameter_edge"], default="iid")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed split seed; --seeds only changes model training")
    parser.add_argument("--edge-label", help="Target used for parameter_edge, e.g. Rate_1 or Z")
    parser.add_argument("--station-order", choices=["original", "random", "spatial_xy"], default="original")
    parser.add_argument(
        "--station-order-seed",
        type=int,
        help="Independent seed for a random station permutation; defaults to --split-seed.",
    )
    parser.add_argument("--station-coordinates", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Override batch size; use 16 or 32 if GPU memory is limited")
    parser.add_argument("--bootstrap-repeats", type=int)
    parser.add_argument("--params-json", type=Path, help="Optional best-parameter JSON from tuning")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_model_parameters(
    base_parameters: dict,
    parameter_payload: dict | None,
    scenario: str,
    model_name: str,
) -> tuple[dict, dict]:
    """Resolve a frozen parameter set while retaining backward compatibility.

    A legacy flat JSON is still accepted.  The preferred schema has
    ``default_by_model`` and ``scenarios`` mappings plus a provenance block.
    """
    resolved = dict(base_parameters)
    provenance = {"source": "config.model", "status": "default"}
    if parameter_payload is None:
        return resolved, provenance

    if "scenarios" not in parameter_payload and "default_by_model" not in parameter_payload:
        resolved.update(parameter_payload)
        return resolved, {"source": "legacy_flat_params_json", "status": "machine_read"}

    model_defaults = parameter_payload.get("default_by_model", {}).get(model_name, {})
    scenario_parameters = (
        parameter_payload.get("scenarios", {}).get(str(scenario), {}).get(model_name, {})
    )
    resolved.update(model_defaults)
    resolved.update(scenario_parameters)
    provenance = dict(parameter_payload.get("provenance", {}))
    provenance.update(
        {
            "source": "structured_params_json",
            "schema_version": parameter_payload.get("schema_version", 1),
            "model_default_applied": bool(model_defaults),
            "scenario_override_applied": bool(scenario_parameters),
        }
    )
    return resolved, provenance


def experiment_subdirectory(args: argparse.Namespace) -> Path:
    split_path = Path(args.split)
    if args.split == "parameter_edge":
        split_path /= f"edge_{args.edge_label or 'auto'}"
    order_name = f"order_{args.station_order}"
    if args.station_order == "random":
        order_name += f"_seed_{args.station_order_seed if args.station_order_seed is not None else args.split_seed}"
    return split_path / order_name


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import tensorflow as tf

    # tf.keras.utils.set_random_seed was added after TensorFlow 2.6.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.split == "parameter_edge" and not args.edge_label:
        raise ValueError("--edge-label is required for parameter_edge so runs cannot overwrite each other.")
    config_path = args.config.resolve()
    config = load_config(config_path)
    scenario_cfg = config["scenarios"][args.scenario]
    training_cfg = config["training"]
    base_model_params = dict(config["model"])
    parameter_payload = None
    if args.params_json:
        with args.params_json.open("r", encoding="utf-8") as handle:
            parameter_payload = json.load(handle)

    csv_path = args.data_file.resolve() if args.data_file else resolve_from_config(config_path, scenario_cfg["data"])
    labels = list(scenario_cfg["labels"])
    time_steps = int(config["time_steps"])
    feature_dim = int(config["feature_dim"])
    seeds = args.seeds or list(training_cfg["seeds"])
    epochs = args.epochs or int(training_cfg["epochs"])
    bootstrap_repeats = (
        args.bootstrap_repeats
        if args.bootstrap_repeats is not None
        else int(training_cfg["bootstrap_repeats"])
    )

    print(f"Loading scenario {args.scenario}: {csv_path}")
    print(f"Required target columns for scenario {args.scenario}: {labels}")
    try:
        bundle = load_dataset(csv_path, time_steps, feature_dim, labels)
    except ValueError as exc:
        if "Missing required columns" in str(exc):
            raise ValueError(
                f"Scenario/data mismatch: scenario {args.scenario} requires target columns {labels}, "
                f"but they were not all found in {csv_path}. Check that scenario 1 uses data s1.csv, "
                "scenario 2 uses data s2.csv, and scenario 3 uses data s3.csv. "
                f"Original error: {exc}"
            ) from exc
        raise
    summary = dataset_summary(bundle, csv_path)
    summary["sha256"] = sha256_file(csv_path)
    summary["requested_split"] = args.split
    summary["requested_station_order"] = args.station_order
    summary["station_order_seed"] = (
        args.station_order_seed if args.station_order_seed is not None else args.split_seed
    )
    if args.params_json:
        summary["parameter_file"] = str(args.params_json.resolve())
        summary["parameter_file_sha256"] = sha256_file(args.params_json.resolve())

    coords = None
    if args.station_coordinates:
        coords = load_station_coordinates(args.station_coordinates.resolve(), feature_dim)
    if "Graph_CLA" in args.models and coords is None:
        raise ValueError("Graph_CLA requires --station-coordinates with 136 station coordinates.")

    # Perform the split before importing TensorFlow so dry-run works in data-only environments.
    preflight_split = split_indices(
        bundle,
        args.split,
        args.split_seed,
        float(training_cfg["test_fraction"]),
        float(training_cfg["validation_fraction"]),
        float(training_cfg["calibration_fraction"]),
        args.edge_label,
    )[4]
    summary["preflight_split"] = preflight_split
    preflight_suffix = args.split
    if args.split == "parameter_edge":
        preflight_suffix += f"_{args.edge_label}"
    if args.station_order == "random":
        preflight_suffix += f"_order_seed_{summary['station_order_seed']}"
    preflight_path = args.output_root.resolve() / f"preflight_scenario_{args.scenario}_{preflight_suffix}.json"
    write_json(preflight_path, summary)
    print(f"Preflight written to {preflight_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import tensorflow as tf
    from models import build_model

    # Avoid TensorFlow 2.6 reserving all GPU memory at startup.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"Warning: could not enable GPU memory growth for {gpu}: {exc}")

    for seed in seeds:
        train_idx, val_idx, calibration_idx, test_idx, split_details = split_indices(
            bundle,
            args.split,
            args.split_seed,
            float(training_cfg["test_fraction"]),
            float(training_cfg["validation_fraction"]),
            float(training_cfg["calibration_fraction"]),
            args.edge_label,
        )
        station_order_seed = (
            args.station_order_seed if args.station_order_seed is not None else args.split_seed
        )
        permutation = station_permutation(args.station_order, feature_dim, station_order_seed, coords)
        X_ordered = bundle.X[:, :, permutation]

        x_transformer = InputTransformer(training_cfg["input_transform"]).fit(X_ordered[train_idx])
        y_transformer = TargetTransformer(labels).fit(bundle.y[train_idx])
        X_train = x_transformer.transform(X_ordered[train_idx])
        X_val = x_transformer.transform(X_ordered[val_idx])
        X_calibration = x_transformer.transform(X_ordered[calibration_idx])
        X_test = x_transformer.transform(X_ordered[test_idx])
        y_train = y_transformer.transform(bundle.y[train_idx])
        y_val = y_transformer.transform(bundle.y[val_idx])

        for model_name in args.models:
            model_params, parameter_provenance = resolve_model_parameters(
                base_model_params, parameter_payload, args.scenario, model_name
            )
            run_dir = (
                args.output_root.resolve()
                / f"scenario_{args.scenario}"
                / experiment_subdirectory(args)
                / model_name
                / f"seed_{seed}"
            )
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists() and not args.overwrite:
                print(f"Skipping existing run: {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)

            set_reproducibility(seed)
            tf.keras.backend.clear_session()
            current_adjacency = None
            if coords is not None:
                current_adjacency = build_normalized_adjacency(coords, int(model_params["graph_k"]))
                if args.station_order != "original":
                    current_adjacency = current_adjacency[np.ix_(permutation, permutation)]
            model = build_model(
                model_name,
                (time_steps, feature_dim),
                len(labels),
                model_params,
                adjacency=current_adjacency,
            )
            learning_rate = float(model_params.get("learning_rate", training_cfg["learning_rate"]))
            batch_size = int(args.batch_size or model_params.get("batch_size", training_cfg["batch_size"]))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss="mae")

            summary_lines: list[str] = []
            model.summary(print_fn=summary_lines.append)
            (run_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(training_cfg["patience"]),
                    restore_best_weights=True,
                    mode="min",
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=max(5, int(training_cfg["patience"]) // 3), min_lr=1e-6
                ),
                tf.keras.callbacks.CSVLogger(str(run_dir / "epoch_log.csv")),
                tf.keras.callbacks.ModelCheckpoint(
                    str(run_dir / "best_model.h5"), monitor="val_loss", save_best_only=True, mode="min"
                ),
            ]
            started = datetime.now(timezone.utc)
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=callbacks,
                verbose=2,
            )
            finished = datetime.now(timezone.utc)
            model.save(str(run_dir / "model.h5"))

            prediction_scaled = model.predict(X_test, batch_size=batch_size, verbose=0)
            prediction = y_transformer.inverse_transform(prediction_scaled)
            calibration_scaled = model.predict(X_calibration, batch_size=batch_size, verbose=0)
            calibration_prediction = y_transformer.inverse_transform(calibration_scaled)
            truth = bundle.y[test_idx]
            metrics, intervals = evaluate_predictions(
                truth, prediction, labels, bootstrap_repeats=bootstrap_repeats, seed=seed
            )
            metrics.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
            intervals.to_csv(run_dir / "metric_intervals.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(history.history).assign(Epoch=lambda frame: np.arange(1, len(frame) + 1)).to_csv(
                run_dir / "history.csv", index=False, encoding="utf-8-sig"
            )

            raw = pd.DataFrame({"sample_index": test_idx})
            for i, label in enumerate(labels):
                raw[f"True_{label}"] = truth[:, i]
                raw[f"Pred_{label}"] = prediction[:, i]
            errors = sample_error_table(truth, prediction, labels)
            raw = pd.concat([raw, errors], axis=1)
            raw.to_csv(run_dir / "raw_predictions.csv", index=False, encoding="utf-8-sig")
            calibration_raw = pd.DataFrame({"sample_index": calibration_idx})
            for i, label in enumerate(labels):
                calibration_raw[f"True_{label}"] = bundle.y[calibration_idx, i]
                calibration_raw[f"Pred_{label}"] = calibration_prediction[:, i]
            calibration_raw.to_csv(run_dir / "calibration_predictions.csv", index=False, encoding="utf-8-sig")
            np.savez_compressed(
                run_dir / "split_indices.npz",
                train=train_idx,
                validation=val_idx,
                calibration=calibration_idx,
                test=test_idx,
                station_permutation=permutation,
            )
            with (run_dir / "scalers.pkl").open("wb") as handle:
                pickle.dump({"input": x_transformer, "target": y_transformer, "labels": labels}, handle)

            manifest = {
                "status": "complete",
                "scenario": args.scenario,
                "model": model_name,
                "seed": seed,
                "split_seed": args.split_seed,
                "split": split_details,
                "station_order": args.station_order,
                "station_order_seed": int(station_order_seed),
                "station_permutation": permutation.tolist(),
                "data": summary,
                "labels": labels,
                "model_parameters": model_params,
                "parameter_provenance": parameter_provenance,
                "total_parameters": int(model.count_params()),
                "training": {
                    **training_cfg,
                    "learning_rate_used": learning_rate,
                    "batch_size_used": batch_size,
                    "epochs_requested": epochs,
                    "epochs_completed": len(history.history["loss"]),
                    "best_validation_loss": float(np.min(history.history["val_loss"])),
                },
                "started_utc": started.isoformat(),
                "finished_utc": finished.isoformat(),
                "duration_seconds": (finished - started).total_seconds(),
                "software": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "tensorflow": tf.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            }
            write_json(manifest_path, manifest)
            print(f"Completed {model_name}, seed={seed}: {run_dir}")


if __name__ == "__main__":
    main()
