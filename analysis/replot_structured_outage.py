from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
DEFAULT_INPUT = SCRIPT.parents[1] / "structured_outage" / "structured_outage"
DEFAULT_OUTPUT = SCRIPT.parents[1] / "publication_ready"

COLORS = {"CLA": "#0072B2", "LSTM_Attention": "#D55E00"}
MODEL_LABELS = {"CLA": "CLA", "LSTM_Attention": "LSTM-Attention"}
CONDITIONS = [
    "radial_inner",
    "radial_middle",
    "radial_outer",
    "azimuth_q1",
    "azimuth_q2",
    "azimuth_q3",
    "azimuth_q4",
]
SHORT_LABELS = ["I", "M", "O", "A1", "A2", "A3", "A4"]
TASKS = ["Release rate", "Horizontal location", "Release height"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot structured-outage results without model inference.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
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


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")


def build_source_table(summary: pd.DataFrame) -> pd.DataFrame:
    source = summary.pivot_table(
        index=["model", "task", "condition", "removed_count"],
        columns="condition_type",
        values=["mean", "ci95"],
    ).reset_index()
    source.columns = [
        "_".join([str(item) for item in col if str(item)]) if isinstance(col, tuple) else str(col)
        for col in source.columns
    ]
    source["structured_vs_random_percent"] = (
        source["mean_structured"] / source["mean_matched_random"] - 1.0
    ) * 100.0
    return source


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.70, wspace=0.29)
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = {"CLA": -0.09, "LSTM_Attention": 0.09}

    for panel, (ax, task) in enumerate(zip(axes, TASKS)):
        for model in ("CLA", "LSTM_Attention"):
            color = COLORS[model]
            for condition_type, marker, face, linestyle in (
                ("structured", "o", color, "-"),
                ("matched_random", "s", "white", "--"),
            ):
                current = (
                    summary[
                        (summary["model"] == model)
                        & (summary["task"] == task)
                        & (summary["condition_type"] == condition_type)
                    ]
                    .set_index("condition")
                    .reindex(CONDITIONS)
                )
                label = None
                if panel == 0:
                    suffix = "structured" if condition_type == "structured" else "matched random"
                    label = f"{MODEL_LABELS[model]} {suffix}"
                ax.errorbar(
                    x + offsets[model],
                    current["mean"],
                    yerr=current["ci95"],
                    color=color,
                    marker=marker,
                    markerfacecolor=face,
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    markersize=4.4,
                    linewidth=0.9,
                    linestyle=linestyle,
                    capsize=2.0,
                    capthick=0.8,
                    label=label,
                    zorder=3,
                )
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.9, 22.5)
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([1, 2, 5, 10, 20]))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_LABELS)
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
    save(fig, out_dir, "Fig_structured_outage_performance_JER")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.input_dir.resolve() / "structured_outage_summary.csv"
    summary = pd.read_csv(summary_path)
    expected = {"model", "condition_type", "condition", "task", "mean", "ci95"}
    missing = expected.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_path}: {sorted(missing)}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    build_source_table(summary).to_csv(
        out_dir / "Fig_structured_outage_performance_JER_source.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot(summary, out_dir)
    print(f"Publication-ready outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
