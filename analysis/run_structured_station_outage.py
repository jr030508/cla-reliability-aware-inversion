from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
JER_ROOT = SCRIPT_PATH.parents[1]
EXPERIMENT_ROOT = SCRIPT_PATH.parents[2]

COLORS = {"CLA": "#0072B2", "LSTM_Attention": "#D55E00"}
MODEL_LABELS = {"CLA": "CLA", "LSTM_Attention": "LSTM-Attention"}
CONDITION_LABELS = {
    "radial_inner": "Inner\nrange",
    "radial_middle": "Middle\nrange",
    "radial_outer": "Outer\nrange",
    "azimuth_q1": "Azimuth\nQ1",
    "azimuth_q2": "Azimuth\nQ2",
    "azimuth_q3": "Azimuth\nQ3",
    "azimuth_q4": "Azimuth\nQ4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-only, coordinate-aware station outage experiment."
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=JER_ROOT / "source_data" / "receptor_coordinates_exact.csv",
    )
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=JER_ROOT / "structured_outage")
    parser.add_argument("--models", nargs="+", default=["CLA", "LSTM_Attention"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Generate and validate spatial masks without loading TensorFlow models.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Recreate the performance figure from the existing summary CSV without inference.",
    )
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=4,
        width=0.9,
        pad=4,
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(directory / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")


def load_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rename = {
        "Station_ID": "receptor_id",
        "X_km": "utm_x_km",
        "Y_km": "utm_y_km",
    }
    frame = frame.rename(columns=rename)
    required = {"receptor_id", "utm_x_km", "utm_y_km"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Coordinate file must include {sorted(required)}")
    frame = frame.sort_values("receptor_id").reset_index(drop=True)
    expected = np.arange(1, len(frame) + 1)
    if len(frame) != 136 or not np.array_equal(frame["receptor_id"].to_numpy(), expected):
        raise ValueError("Coordinates must contain receptor IDs 1..136 exactly once")
    return frame


def equal_count_groups(values: np.ndarray, groups: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    for group, indices in enumerate(np.array_split(order, groups)):
        labels[indices] = group
    return labels


def build_structured_masks(coords: pd.DataFrame) -> dict[str, np.ndarray]:
    source_x, source_y = 342.58, 6179.61
    dx = coords["utm_x_km"].to_numpy() - source_x
    dy = coords["utm_y_km"].to_numpy() - source_y
    distance = np.hypot(dx, dy)
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    radial_group = equal_count_groups(distance, 3)
    azimuth_group = equal_count_groups(bearing, 4)

    masks = {
        "radial_inner": radial_group == 0,
        "radial_middle": radial_group == 1,
        "radial_outer": radial_group == 2,
    }
    masks.update({f"azimuth_q{i + 1}": azimuth_group == i for i in range(4)})
    return masks


def export_masks(coords: pd.DataFrame, masks: dict[str, np.ndarray], output_dir: Path) -> None:
    rows = coords[["receptor_id", "utm_x_km", "utm_y_km"]].copy()
    for name, mask in masks.items():
        rows[name + "_removed"] = mask.astype(int)
    rows.to_csv(output_dir / "structured_outage_masks.csv", index=False, encoding="utf-8-sig")

    manifest = []
    for name, mask in masks.items():
        manifest.append(
            {
                "condition": name,
                "removed_count": int(mask.sum()),
                "retained_count": int(len(mask) - mask.sum()),
                "removed_receptor_ids": ",".join(map(str, coords.loc[mask, "receptor_id"].tolist())),
            }
        )
    pd.DataFrame(manifest).to_csv(
        output_dir / "structured_outage_mask_manifest.csv", index=False, encoding="utf-8-sig"
    )


def plot_masks(coords: pd.DataFrame, masks: dict[str, np.ndarray], output_dir: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 4.70))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.90, wspace=0.36, hspace=0.60)
    axes_flat = axes.flat
    for panel, (name, mask) in enumerate(masks.items()):
        ax = axes_flat[panel]
        ax.scatter(
            coords.loc[~mask, "utm_x_km"],
            coords.loc[~mask, "utm_y_km"],
            s=11,
            color="#BDBDBD",
            edgecolor="black",
            linewidth=0.2,
            label="Retained" if panel == 0 else None,
        )
        ax.scatter(
            coords.loc[mask, "utm_x_km"],
            coords.loc[mask, "utm_y_km"],
            s=15,
            color="#D55E00",
            edgecolor="black",
            linewidth=0.25,
            label="Removed" if panel == 0 else None,
        )
        ax.set_title(CONDITION_LABELS[name].replace("\n", " "), pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (km)")
        if panel % 4 == 0:
            ax.set_ylabel("Northing (km)")
        else:
            ax.set_yticklabels([])
        style_axes(ax)
        ax.text(
            -0.10,
            1.16,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
    axes_flat[7].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[7].legend(handles, labels, loc="center", frameon=False)
    save_figure(fig, output_dir, "FigS01_structured_outage_masks")
    plt.close(fig)


def load_model_and_data(run_dir: Path, data_file: Path, config_path: Path, max_samples: int | None):
    sys.path.insert(0, str(EXPERIMENT_ROOT))
    from data_pipeline import load_config, load_dataset

    with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    config = load_config(config_path)
    scenario = str(manifest["scenario"])
    labels = list(manifest.get("labels") or config["scenarios"][scenario]["labels"])
    bundle = load_dataset(data_file, int(config["time_steps"]), int(config["feature_dim"]), labels)
    split = np.load(run_dir / "split_indices.npz")
    test_idx = split["test"]
    permutation = split["station_permutation"]
    X_test = bundle.X[test_idx][:, :, permutation]
    y_true = bundle.y[test_idx]
    if max_samples:
        X_test = X_test[:max_samples]
        y_true = y_true[:max_samples]
    with (run_dir / "scalers.pkl").open("rb") as handle:
        transformers = pickle.load(handle)

    import tensorflow as tf

    model_path = run_dir / "model.h5"
    if not model_path.exists():
        model_path = run_dir / "best_model.h5"
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return manifest, labels, X_test, y_true, permutation, transformers, model


def predict_metrics(
    model,
    transformers,
    X: np.ndarray,
    y_true: pd.DataFrame,
    labels: list[str],
    batch_size: int,
) -> pd.DataFrame:
    from metrics import evaluate_predictions

    X_scaled = transformers["input"].transform(X)
    prediction_scaled = model.predict(X_scaled, batch_size=batch_size, verbose=0)
    prediction = transformers["target"].inverse_transform(prediction_scaled)
    metrics, _ = evaluate_predictions(y_true, prediction, labels, 0, 42)
    return metrics


def run_full(args: argparse.Namespace, coords: pd.DataFrame, masks: dict[str, np.ndarray]) -> None:
    if args.scenario_root is None or args.data_file is None:
        raise ValueError("--scenario-root and --data-file are required unless --mask-only is used")
    sys.path.insert(0, str(EXPERIMENT_ROOT))
    records = []
    for model_name in args.models:
        for seed in args.seeds:
            run_dir = args.scenario_root.resolve() / model_name / f"seed_{seed}"
            manifest, labels, X_test, y_true, permutation, transformers, model = load_model_and_data(
                run_dir,
                args.data_file.resolve(),
                args.config.resolve(),
                args.max_test_samples,
            )
            ordered_masks = {name: mask[np.asarray(permutation, dtype=int)] for name, mask in masks.items()}

            clean_metrics = predict_metrics(
                model, transformers, X_test, y_true, labels, args.batch_size
            )
            for row in clean_metrics.to_dict("records"):
                records.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "condition_type": "clean",
                        "condition": "clean",
                        "repeat": 0,
                        "removed_count": 0,
                        **row,
                    }
                )

            for condition_index, (condition, removed_mask) in enumerate(ordered_masks.items()):
                X_structured = np.asarray(X_test).copy()
                X_structured[:, :, removed_mask] = 0.0
                metrics = predict_metrics(
                    model, transformers, X_structured, y_true, labels, args.batch_size
                )
                for row in metrics.to_dict("records"):
                    records.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "condition_type": "structured",
                            "condition": condition,
                            "repeat": 0,
                            "removed_count": int(removed_mask.sum()),
                            **row,
                        }
                    )

                for repeat in range(args.random_repeats):
                    rng = np.random.default_rng(20261029 + seed * 1000 + condition_index * 100 + repeat)
                    random_removed = np.zeros(len(removed_mask), dtype=bool)
                    random_removed[
                        rng.choice(len(removed_mask), size=int(removed_mask.sum()), replace=False)
                    ] = True
                    X_random = np.asarray(X_test).copy()
                    X_random[:, :, random_removed] = 0.0
                    metrics = predict_metrics(
                        model, transformers, X_random, y_true, labels, args.batch_size
                    )
                    for row in metrics.to_dict("records"):
                        records.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "condition_type": "matched_random",
                                "condition": condition,
                                "repeat": repeat,
                                "removed_count": int(random_removed.sum()),
                                **row,
                            }
                        )
            del model

    raw = pd.DataFrame(records)
    raw.to_csv(args.output_dir / "structured_outage_metrics_raw.csv", index=False, encoding="utf-8-sig")
    task_rows = []
    key_cols = ["model", "seed", "condition_type", "condition", "repeat", "removed_count"]
    for key, group in raw.groupby(key_cols, dropna=False):
        base = dict(zip(key_cols, key))
        rates = group[group["Parameter"].isin(["Rate_1", "Rate_2", "Rate_3", "Rate_4"])]
        xy = group[group["Parameter"] == "XY_DISTANCE_km"]
        height = group[group["Parameter"] == "Z"]
        task_rows.extend(
            [
                {**base, "task": "Release rate", "metric": "Log10_MAE", "value": float(rates["Log10_MAE"].mean())},
                {**base, "task": "Horizontal location", "metric": "MAE_km", "value": float(xy["MAE"].iloc[0])},
                {**base, "task": "Release height", "metric": "Log10_MAE", "value": float(height["Log10_MAE"].iloc[0])},
            ]
        )
    tasks = pd.DataFrame(task_rows)
    clean = tasks[tasks["condition_type"] == "clean"][["model", "seed", "task", "value"]].rename(columns={"value": "clean_value"})
    tasks = tasks.merge(clean, on=["model", "seed", "task"], how="left")
    tasks["error_ratio_to_clean"] = tasks["value"] / tasks["clean_value"]
    tasks.to_csv(args.output_dir / "structured_outage_task_metrics.csv", index=False, encoding="utf-8-sig")

    summary = (
        tasks[tasks["condition_type"] != "clean"]
        .groupby(["model", "condition_type", "condition", "task", "removed_count"], dropna=False)["error_ratio_to_clean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
    summary.to_csv(args.output_dir / "structured_outage_summary.csv", index=False, encoding="utf-8-sig")
    plot_results(summary, args.output_dir)


def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    configure_style()
    conditions = list(CONDITION_LABELS)
    short_labels = ["I", "M", "O", "A1", "A2", "A3", "A4"]
    tasks = ["Release rate", "Horizontal location", "Release height"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.70, wspace=0.29)
    x = np.arange(len(conditions), dtype=float)
    offsets = {"CLA": -0.10, "LSTM_Attention": 0.10}
    for panel, (ax, task) in enumerate(zip(axes, tasks)):
        for model in ("CLA", "LSTM_Attention"):
            color = COLORS[model]
            for condition_type, marker, fill, linestyle in (
                ("structured", "o", color, "-"),
                ("matched_random", "s", "white", "--"),
            ):
                current = summary[
                    (summary["model"] == model)
                    & (summary["task"] == task)
                    & (summary["condition_type"] == condition_type)
                ].set_index("condition").reindex(conditions)
                ax.errorbar(
                    x + offsets[model],
                    current["mean"],
                    yerr=current["ci95"],
                    color=color,
                    marker=marker,
                    markerfacecolor=fill,
                    markeredgecolor=color,
                    markersize=4.2,
                    linewidth=0.9,
                    linestyle=linestyle,
                    capsize=2.0,
                    label=(f"{MODEL_LABELS[model]} structured" if condition_type == "structured" else f"{MODEL_LABELS[model]} matched random") if panel == 0 else None,
                )
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(0.9, 22.5)
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([1, 2, 5, 10, 20]))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_xlabel("Removed spatial block")
        ax.set_title(task, pad=8)
        if panel == 0:
            ax.set_ylabel("Error ratio to clean condition")
        style_axes(ax)
        ax.text(
            -0.13,
            1.14,
            f"({chr(97 + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            clip_on=False,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.2,
    )
    save_figure(fig, output_dir, "Fig02_structured_outage_performance")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_only:
        summary_path = args.output_dir / "structured_outage_summary.csv"
        plot_results(pd.read_csv(summary_path), args.output_dir)
        print(f"Performance figure recreated from: {summary_path}")
        return
    coords = load_coordinates(args.coordinates.resolve())
    masks = build_structured_masks(coords)
    export_masks(coords, masks, args.output_dir)
    plot_masks(coords, masks, args.output_dir)
    if args.mask_only:
        print("Spatial masks generated and validated; model inference was not requested.")
        return
    run_full(args, coords, masks)
    print(f"Structured outage outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
