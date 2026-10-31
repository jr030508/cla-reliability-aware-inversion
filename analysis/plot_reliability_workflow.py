from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCRIPT = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT.parents[1] / "publication_ready"

INK = "#1F2933"
EDGE = "#456C7D"
SECTION = "#F3F8FA"
BLUE = "#D7EAF2"
GREEN = "#DCEFE7"
SAND = "#F5E8C8"
ROSE = "#F4D9D5"
ACCENT_BLUE = "#0072B2"
ACCENT_GREEN = "#009E73"
ACCENT_ORANGE = "#E69F00"
ACCENT_RED = "#D55E00"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = EDGE,
    fontsize: float = 7.3,
    weight: str = "normal",
    radius: float = 0.018,
    linewidth: float = 0.8,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.12,
        zorder=3,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    connectionstyle: str = "arc3,rad=0",
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=4,
        )
    )


def section(ax: plt.Axes, y: float, height: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.01, y),
            0.98,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.022",
            facecolor=SECTION,
            edgecolor=EDGE,
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.text(
        0.5,
        y + height - 0.018,
        title,
        ha="center",
        va="top",
        color=INK,
        fontsize=9.0,
        fontweight="bold",
        zorder=5,
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.03}
    fig.savefig(output_dir / f"{stem}.png", dpi=600, **common)
    fig.savefig(output_dir / f"{stem}.pdf", **common)
    fig.savefig(output_dir / f"{stem}.svg", **common)
    fig.savefig(
        output_dir / f"{stem}.tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )


def plot_workflow(output_dir: Path = OUTPUT_DIR) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.25, 5.55))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Stage 1: controlled benchmark and observable monitoring degradation.
    section(ax, 0.765, 0.225, "Controlled CALPUFF benchmark and monitoring degradation")
    top_y, top_h, top_w = 0.795, 0.115, 0.205
    top_x = [0.035, 0.285, 0.535, 0.785]
    top_text = [
        "Fixed Copenhagen event\nCALPUFF configuration",
        "Source sampling\n$q_1$–$q_4$, $X$, $Y$, $Z$",
        "Monitoring tensor\n12 time steps × 136 receptors",
        "Degradation protocols\nstation/time loss, LOD, bias",
    ]
    top_colors = [BLUE, GREEN, BLUE, SAND]
    for x, text, color in zip(top_x, top_text, top_colors):
        rounded_box(ax, x, top_y, top_w, top_h, text, facecolor=color, fontsize=7.2)
    for left, right in zip(top_x[:-1], top_x[1:]):
        arrow(ax, (left + top_w, top_y + top_h / 2), (right, top_y + top_h / 2))

    # Stage 2: CLA point estimator.
    section(ax, 0.515, 0.215, "CLA multi-parameter point estimator")
    model_y, model_h, model_w = 0.56, 0.105, 0.19
    model_x = [0.045, 0.285, 0.525, 0.765]
    model_text = [
        "2D-CNN receptor–time\nfeature extraction",
        "Stacked LSTM\nsequence encoding",
        "Temporal attention\nadaptive weighting",
        "Seven point estimates\n$q_1$–$q_4$, $X$, $Y$, $Z$",
    ]
    for x, text in zip(model_x, model_text):
        rounded_box(ax, x, model_y, model_w, model_h, text, facecolor=BLUE, fontsize=7.3)
    for left, right in zip(model_x[:-1], model_x[1:]):
        arrow(ax, (left + model_w, model_y + model_h / 2), (right, model_y + model_h / 2))
    rounded_box(
        ax,
        0.245,
        0.525,
        0.51,
        0.026,
        "Point prediction layer: accuracy is evaluated separately from reliability",
        facecolor="#E9F2F5",
        edgecolor="#9DB7C2",
        fontsize=6.7,
        radius=0.012,
        linewidth=0.6,
    )

    # Stage 3: calibration, conditional audit, and hybrid gate.
    section(ax, 0.245, 0.235, "Conditional uncertainty audit and hybrid reliability gate")
    audit_y, audit_h, audit_w = 0.345, 0.075, 0.175
    audit_x = [0.025, 0.225, 0.425, 0.625, 0.825]
    audit_text = [
        "Calibration\nresiduals",
        "Global split\nconformal",
        "Conditional\ncoverage audit",
        "Risk-Mondrian\nrecalibration",
        "Conditioned\nintervals",
    ]
    for x, text in zip(audit_x, audit_text):
        rounded_box(ax, x, audit_y, audit_w - 0.01, audit_h, text, facecolor=GREEN, fontsize=6.9)
    for left, right in zip(audit_x[:-1], audit_x[1:]):
        arrow(ax, (left + audit_w - 0.01, audit_y + audit_h / 2), (right, audit_y + audit_h / 2))

    rounded_box(
        ax,
        0.16,
        0.268,
        0.22,
        0.052,
        "Learned risk score\nfrom model features",
        facecolor=BLUE,
        fontsize=6.8,
    )
    rounded_box(
        ax,
        0.62,
        0.268,
        0.22,
        0.052,
        "Protocol metadata\nknown acquisition state",
        facecolor=SAND,
        fontsize=6.8,
    )
    rounded_box(
        ax,
        0.405,
        0.258,
        0.19,
        0.070,
        "Hybrid gate\naccept / flag / reject",
        facecolor="#DDE8F3",
        edgecolor=ACCENT_BLUE,
        fontsize=7.0,
        weight="bold",
        linewidth=1.0,
    )
    arrow(ax, (0.38, 0.294), (0.405, 0.294), color=ACCENT_BLUE)
    arrow(ax, (0.62, 0.294), (0.595, 0.294), color=ACCENT_ORANGE)

    # Stage 4: auditable decision rather than forced prediction.
    section(ax, 0.015, 0.195, "Auditable operational decision")
    rounded_box(
        ax,
        0.055,
        0.085,
        0.25,
        0.064,
        "ACCEPT\npoint estimate + conditioned interval",
        facecolor=GREEN,
        edgecolor=ACCENT_GREEN,
        fontsize=6.9,
        weight="bold",
    )
    rounded_box(
        ax,
        0.375,
        0.085,
        0.25,
        0.064,
        "FLAG\nconditional undercoverage or moderate shift",
        facecolor=SAND,
        edgecolor=ACCENT_ORANGE,
        fontsize=6.9,
        weight="bold",
    )
    rounded_box(
        ax,
        0.695,
        0.085,
        0.25,
        0.064,
        "REJECT\nsevere monitoring loss or OOD input",
        facecolor=ROSE,
        edgecolor=ACCENT_RED,
        fontsize=6.9,
        weight="bold",
    )
    rounded_box(
        ax,
        0.225,
        0.032,
        0.55,
        0.031,
        "Output: empirical operating boundary for the controlled event, not a cross-meteorology guarantee",
        facecolor="#E8F0F3",
        edgecolor="#9DB7C2",
        fontsize=6.6,
        radius=0.012,
        linewidth=0.6,
    )

    # Vertical stage connectors emphasize the main workflow without implying hidden feedback loops.
    arrow(ax, (0.5, 0.758), (0.5, 0.737), color=ACCENT_BLUE, linewidth=1.1)
    arrow(ax, (0.5, 0.508), (0.5, 0.487), color=ACCENT_BLUE, linewidth=1.1)
    arrow(ax, (0.5, 0.238), (0.5, 0.217), color=ACCENT_BLUE, linewidth=1.1)

    save(fig, output_dir, "Fig00_reliability_workflow_JER")
    plt.close(fig)


if __name__ == "__main__":
    plot_workflow()
    print(f"Reliability workflow written to: {OUTPUT_DIR}")
