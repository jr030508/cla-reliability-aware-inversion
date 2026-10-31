from __future__ import annotations

import argparse
import json
import os
import pickle
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


MODELS = ["CLA", "LSTM", "LSTM_Attention", "CNN_LSTM", "2D_CNN", "2D_CNN_Attention", "Graph_CLA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Optuna tuning with a fixed trial budget")
    parser.add_argument("--scenario", choices=["1", "2", "3"], required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split", choices=["iid", "group_event", "spatial_block", "parameter_edge"], default="iid")
    parser.add_argument("--edge-label")
    parser.add_argument("--station-order", choices=["original", "random", "spatial_xy"], default="original")
    parser.add_argument("--station-coordinates", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("tuning"))
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def suggest_params(trial, model_name: str, base: dict) -> dict:
    params = dict(base)
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 2e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["dense_1_units"] = trial.suggest_categorical("dense_1_units", [32, 64, 96])
    params["dense_2_units"] = trial.suggest_categorical("dense_2_units", [16, 32, 48])
    params["dropout"] = trial.suggest_float("dropout", 0.10, 0.40, step=0.05)
    if model_name in {"CLA", "CNN_LSTM", "2D_CNN", "2D_CNN_Attention"}:
        params["filters_1"] = trial.suggest_categorical("filters_1", [16, 32, 48, 64])
        params["filters_2"] = trial.suggest_categorical("filters_2", [16, 32, 48, 64])
    if model_name in {"CLA", "LSTM", "LSTM_Attention", "CNN_LSTM", "Graph_CLA"}:
        params["lstm_1_units"] = trial.suggest_categorical("lstm_1_units", [64, 96, 128, 192])
    if model_name in {"CLA", "LSTM", "LSTM_Attention", "Graph_CLA"}:
        params["lstm_2_units"] = trial.suggest_categorical("lstm_2_units", [32, 64, 96, 128])
    if model_name == "Graph_CLA":
        params["graph_units"] = trial.suggest_categorical("graph_units", [8, 16, 32])
    return params


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    scenario_cfg = config["scenarios"][args.scenario]
    training_cfg = config["training"]
    csv_path = resolve_from_config(config_path, scenario_cfg["data"])
    labels = list(scenario_cfg["labels"])
    bundle = load_dataset(csv_path, int(config["time_steps"]), int(config["feature_dim"]), labels)
    train_idx, val_idx, _, _, split_details = split_indices(
        bundle,
        args.split,
        args.split_seed,
        float(training_cfg["test_fraction"]),
        float(training_cfg["validation_fraction"]),
        float(training_cfg["calibration_fraction"]),
        args.edge_label,
    )

    coords = None
    if args.station_coordinates:
        coords = load_station_coordinates(args.station_coordinates.resolve(), int(config["feature_dim"]))
    if args.model == "Graph_CLA" and coords is None:
        raise ValueError("Graph_CLA requires --station-coordinates.")
    permutation = station_permutation(args.station_order, int(config["feature_dim"]), args.split_seed, coords)
    X_ordered = bundle.X[:, :, permutation]
    x_transformer = InputTransformer(training_cfg["input_transform"]).fit(X_ordered[train_idx])
    y_transformer = TargetTransformer(labels).fit(bundle.y[train_idx])
    X_train = x_transformer.transform(X_ordered[train_idx])
    X_val = x_transformer.transform(X_ordered[val_idx])
    y_train = y_transformer.transform(bundle.y[train_idx])
    y_val = y_transformer.transform(bundle.y[val_idx])

    adjacency = None
    if coords is not None:
        adjacency = build_normalized_adjacency(coords, int(config["model"]["graph_k"]))
        adjacency = adjacency[np.ix_(permutation, permutation)]

    output_dir = (
        args.output_root.resolve()
        / f"scenario_{args.scenario}"
        / args.split
        / f"order_{args.station_order}"
        / args.model
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import optuna
    import tensorflow as tf
    from models import build_model

    tf.keras.utils.set_random_seed(args.seed)

    class OptunaPruningCallback(tf.keras.callbacks.Callback):
        def __init__(self, trial):
            super().__init__()
            self.trial = trial

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get("val_loss")
            if value is None:
                return
            self.trial.report(float(value), step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch + 1}")

    def objective(trial):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed + trial.number)
        params = suggest_params(trial, args.model, config["model"])
        model = build_model(
            args.model,
            (int(config["time_steps"]), int(config["feature_dim"])),
            len(labels),
            params,
            adjacency,
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="mae")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, mode="min"
            ),
            OptunaPruningCallback(trial),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=params["batch_size"],
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )
        trial.set_user_attr("epochs_completed", len(history.history["loss"]))
        trial.set_user_attr("parameter_count", int(model.count_params()))
        return float(np.min(history.history["val_loss"]))

    storage_path = (output_dir / "study.sqlite3").as_posix()
    study = optuna.create_study(
        study_name=f"scenario_{args.scenario}_{args.model}_{args.split}_{args.station_order}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, args.trials // 5)),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    best = {**config["model"], **study.best_trial.params}
    write_json(output_dir / "best_params.json", best)
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "tuning_scalers.pkl").open("wb") as handle:
        pickle.dump({"input": x_transformer, "target": y_transformer}, handle)
    manifest = {
        "scenario": args.scenario,
        "model": args.model,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "split": split_details,
        "station_order": args.station_order,
        "trials_requested_this_call": args.trials,
        "trials_total": len(study.trials),
        "best_validation_loss": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "data_sha256": sha256_file(csv_path),
        "data_summary": dataset_summary(bundle, csv_path),
        "test_set_used_for_objective": False
    }
    write_json(output_dir / "tuning_manifest.json", manifest)
    print(f"Best parameters written to {output_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
