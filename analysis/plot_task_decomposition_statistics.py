# -*- coding: utf-8 -*-
"""
把任务依赖负迁移的统计检验结果绘制为期刊格式图件。

(a) 配对秩双列相关效应量 r_rb：合并值为柱，逐种子值为点
(b) 联合训练相对单任务模型的误差增幅（以联合训练值为分母，与正文 47.56% 口径一致）

数据来源：ane_revision/outputs/completed_analysis/task_decomposition_wilcoxon_tests.csv
输出 PDF / PNG / SVG / TIFF，300 dpi，衬线字体。
"""
from __future__ import annotations

import csv
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT = Path(__file__).resolve()
JER = SCRIPT.parents[1]
SRC = (JER.parent / "ane_revision" / "outputs" / "completed_analysis"
       / "task_decomposition_wilcoxon_tests.csv")
OUT_DIR = JER / "publication_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_CLA = "#2E5B88"
C_LSTM = "#C0702C"
C_DOT = "#333333"
C_GRAY = "#B9C2CB"
C_ZERO = "#9C4A4A"

TASK_LABEL = {
    "release_rate": "Release rate",
    "horizontal_location": "Horizontal location",
    "release_height": "Release height",
}
MODEL_LABEL = {"CLA": "CLA", "LSTM_Attention": "LSTM-Attention"}
ORDER = [("CLA", "release_rate"), ("CLA", "horizontal_location"), ("CLA", "release_height"),
         ("LSTM_Attention", "release_rate"), ("LSTM_Attention", "horizontal_location"),
         ("LSTM_Attention", "release_height")]
MODEL_COLOR = {"CLA": C_CLA, "LSTM_Attention": C_LSTM}


def style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load():
    with open(SRC, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    pooled = {}
    seeds = {}
    for r in rows:
        key = (r["model"], r["task"])
        if r["seed"] == "pooled":
            pooled[key] = r
        else:
            seeds.setdefault(key, []).append(r)
    return pooled, seeds


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def save(fig, stem):
    for ext in ("pdf", "png", "svg", "tiff"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, format=ext)
    plt.close(fig)
    print("  ->", stem)


def main():
    style()
    pooled, seeds = load()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.48, 3.0))
    x = np.arange(len(ORDER))
    w = 0.55

    # ---------- (a) 效应量 ----------
    rr = [float(pooled[k]["rank_biserial_r"]) for k in ORDER]
    cols = [MODEL_COLOR[k[0]] for k in ORDER]
    ax_a.bar(x, rr, w, color=cols, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, k in enumerate(ORDER):
        vals = [float(r["rank_biserial_r"]) for r in sorted(seeds[k], key=lambda r: r["seed"])]
        ax_a.scatter([xi] * len(vals), vals, marker="o", s=16, color="white",
                     edgecolors=C_DOT, linewidths=0.8, zorder=5)
        p = float(pooled[k]["holm_p"])
        y = rr[xi]
        ax_a.text(xi, y + (0.09 if y >= 0 else -0.09), sig_marker(p), fontsize=7.5,
                  ha="center", va="bottom" if y >= 0 else "top", color="#333333")
    ax_a.axhline(0, color="#333333", linewidth=0.9, zorder=4)
    for ls, lab in ((0.1, "small"), (0.3, "medium"), (0.5, "large")):
        ax_a.axhline(ls, color=C_GRAY, linestyle=":", linewidth=0.7, zorder=2)
        ax_a.axhline(-ls, color=C_GRAY, linestyle=":", linewidth=0.7, zorder=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Rate", "Location", "Height"] * 2, fontsize=8)
    for x0, lab in ((1.0, "CLA"), (4.0, "LSTM-Attention")):
        ax_a.text(x0, 1.03, lab, transform=ax_a.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_a.plot([2.5, 2.5], [0, 1], transform=ax_a.get_xaxis_transform(),
              color="#333333", linewidth=0.8)
    ax_a.set_ylim(-0.45, 1.05)
    ax_a.set_ylabel("Rank-biserial correlation $r_{rb}$")
    ax_a.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m]) for m in ("CLA", "LSTM_Attention")]
    ax_a.legend(handles, ["CLA (pooled)", "LSTM-Attention (pooled)"],
                loc="lower right", frameon=False, fontsize=6.8)
    ax_a.text(0.02, 0.97, "dots: individual seeds 42 / 43 / 44", transform=ax_a.transAxes,
              fontsize=6.8, ha="left", va="top", color="#555555")
    ax_a.set_title("(a) Effect size of joint versus single-task training", loc="left",
                   fontweight="bold")

    # ---------- (b) 相对变化（vs 联合训练值） ----------
    rel = []
    for k in ORDER:
        r = pooled[k]
        s, j = float(r["single_mean"]), float(r["joint_mean"])
        rel.append(100 * (j - s) / j)
    cols = [MODEL_COLOR[k[0]] for k in ORDER]
    ax_b.bar(x, rel, w, color=cols, edgecolor="white", linewidth=0.6, zorder=3)
    for xi, k in enumerate(ORDER):
        vals = []
        for r in sorted(seeds[k], key=lambda r: r["seed"]):
            s, j = float(r["single_mean"]), float(r["joint_mean"])
            vals.append(100 * (j - s) / j)
        ax_b.scatter([xi] * len(vals), vals, marker="o", s=16, color="white",
                     edgecolors=C_DOT, linewidths=0.8, zorder=5)
    ax_b.axhline(0, color="#333333", linewidth=0.9, zorder=4)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Rate", "Location", "Height"] * 2, fontsize=8)
    for x0, lab in ((1.0, "CLA"), (4.0, "LSTM-Attention")):
        ax_b.text(x0, 1.03, lab, transform=ax_b.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_b.plot([2.5, 2.5], [0, 1], transform=ax_b.get_xaxis_transform(),
              color="#333333", linewidth=0.8)
    ax_b.set_ylim(-18, 60)
    ax_b.set_ylabel("Error increase of joint training (%)")
    ax_b.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("(b) Penalty / gain of joint training", loc="left", fontweight="bold")

    fig.tight_layout()
    save(fig, "Fig_task_decomposition_statistics")


if __name__ == "__main__":
    main()
