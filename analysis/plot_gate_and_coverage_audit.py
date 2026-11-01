# -*- coding: utf-8 -*-
"""
把门控审计与条件覆盖审计结果绘制为期刊格式图件。

Figure A (2x2)  可靠性门控审计
    (a) 失效检测判别力 AUROC / AUPRC（按任务 × 评估范围）
    (b) 被接受样本的残余失效率及其 Wilson 95% 上界（释放率，逐工况）
    (c) 门控阈值可行性（最优 Wilson 上界 vs 10% 容差）
    (d) 风险指标冗余（集成不确定性与保形半宽的 Pearson 相关）

Figure B (1x2)  条件覆盖子群的可观测性
    (a) 按真值划分 vs 按预测值划分的子群覆盖率
    (b) 两种方式之差（百分点）

输出 PDF / PNG / SVG / TIFF，300 dpi，衬线字体。
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT = Path(__file__).resolve()
JER = SCRIPT.parents[1]
TADRG_ANALYSIS = JER.parent / "ane_revision" / "tadrg" / "analysis"
OUT_DIR = JER / "publication_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 配色：印刷与色盲友好的低饱和方案
C_AUROC = "#2E5B88"
C_AUPRC = "#C0702C"
C_TRUTH = "#2E5B88"
C_PRED = "#E0A458"
C_OK = "#4F7C5B"
C_BAD = "#9C4A4A"
C_ACC = "#6E7F8D"
C_GRAY = "#B9C2CB"

TASKS = ["release_rate", "horizontal_location", "release_height"]
TASK_LABEL = {
    "release_rate": "Release rate",
    "horizontal_location": "Horizontal location",
    "release_height": "Release height",
}
SCOPES = ["all_conditions", "non_extreme", "clean_and_bias"]
SCOPE_LABEL = {
    "all_conditions": "All conditions",
    "non_extreme": "Non-extreme",
    "clean_and_bias": "Clean + bias",
}
COND_ORDER = ["clean", "bias_0.10", "bias_0.20", "lod_q05", "lod_q10"]
COND_LABEL = {
    "clean": "Clean",
    "bias_0.10": "Bias 10%",
    "bias_0.20": "Bias 20%",
    "lod_q05": "LOD q05",
    "lod_q10": "LOD q10",
}


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


def read_csv(path):
    import csv
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def save(fig, stem):
    for ext, dpi in (("pdf", 300), ("png", 300), ("svg", 300), ("tiff", 300)):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=dpi, format=ext)
    plt.close(fig)
    print("  ->", stem)


def fig_gate_audit():
    disc = read_csv(TADRG_ANALYSIS / "discrimination_scope_audit.csv")
    cond = read_csv(TADRG_ANALYSIS / "condition_audit.csv")
    gate = read_csv(TADRG_ANALYSIS / "gate_calibration_audit.csv")
    red = read_csv(TADRG_ANALYSIS / "feature_redundancy_audit.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # ---------- (a) AUROC / AUPRC ----------
    disc_rows = {(r["task"], r["scope"]): r for r in disc}
    shade = ["#8FB3D3", "#5A82AB", C_AUROC]
    x = np.arange(len(TASKS))
    w = 0.24
    for j, s in enumerate(SCOPES):
        vals = [float(disc_rows[(t, s)]["auroc"]) for t in TASKS]
        ax_a.bar(x + (j - 1) * w, vals, w, color=shade[j],
                 edgecolor="white", linewidth=0.6, zorder=3,
                 label=f"AUROC — {SCOPE_LABEL[s]}")
    auprc = [float(disc_rows[(t, "all_conditions")]["auprc"]) for t in TASKS]
    ax_a.scatter(x, auprc, marker="D", s=26, color=C_AUPRC, zorder=5,
                 edgecolor="white", linewidth=0.7,
                 label="AUPRC — all conditions")
    for xi, (t, v) in enumerate(zip(TASKS, auprc)):
        ax_a.text(xi, v + 0.030, f"{v:.3f}", fontsize=6.8,
                  ha="center", va="bottom", color="#7A4A18")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([TASK_LABEL[t].replace(" ", "\n") for t in TASKS], fontsize=7.5)
    ax_a.set_ylim(0, 1.32)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_ylabel("Score")
    ax_a.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper center", frameon=False, ncol=2, fontsize=6.8,
                handlelength=1.2, columnspacing=1.0)
    ax_a.set_title("(a) Failure-detection discrimination", loc="left", fontweight="bold")

    # ---------- (b) 被接受样本的残余失效率 ----------
    rows = {r["condition"]: r for r in cond if r["task"] == "release_rate"}
    vals, ups, labels, colors = [], [], [], []
    for c in COND_ORDER:
        r = rows[c]
        p = float(r["failure_rate_accepted"])
        u = float(r["wilson_upper_accepted"])
        vals.append(p)
        ups.append(u - p)
        labels.append(COND_LABEL[c])
        colors.append(C_OK if r["passes_10pct_wilson"] == "True" else C_BAD)
    xb = np.arange(len(labels))
    ax_b.bar(xb, vals, 0.55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax_b.errorbar(xb, vals, yerr=[np.zeros(len(vals)), ups], fmt="none",
                  ecolor="#333333", elinewidth=0.9, capsize=3, capthick=0.9, zorder=4)
    ax_b.axhline(0.10, color=C_BAD, linestyle="--", linewidth=1.0, zorder=5)
    ax_b.text(-0.42, 0.115, "10% tolerance", fontsize=7.5,
              color=C_BAD, ha="left", va="bottom")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(labels)
    ax_b.set_xlim(-0.6, len(labels) - 0.4)
    ax_b.set_ylim(0, 0.55)
    ax_b.set_ylabel("Residual failure rate among accepted")
    ax_b.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    for xi, (p, u) in enumerate(zip(vals, ups)):
        cap = p + u
        if 0.05 <= cap <= 0.11:    # 靠近容差线：放到误差棒右下方，避免压线
            ax_b.text(xi + 0.34, cap - 0.004, f"{100*p:.1f}%", fontsize=6.8,
                      ha="left", va="top", color="#333333")
        else:
            ax_b.text(xi, cap + 0.024, f"{100*p:.1f}%", fontsize=6.8,
                      ha="center", va="bottom", color="#333333")
    ax_b.set_title("(b) Accepted-sample residual failure (release rate)", loc="left",
                   fontweight="bold")

    # ---------- (c) 门控阈值可行性 ----------
    grows = {r["task"]: r for r in gate}
    us = [float(grows[t]["best_available_wilson_upper"]) for t in TASKS]
    feas = [grows[t]["gate_feasible"] == "True" for t in TASKS]
    cols = [C_OK if f else C_BAD for f in feas]
    xc = np.arange(len(TASKS))
    ax_c.bar(xc, us, 0.5, color=cols, edgecolor="white", linewidth=0.6, zorder=3)
    ax_c.axhline(0.10, color="#333333", linestyle="--", linewidth=1.0, zorder=5)
    for xi, (u, f) in enumerate(zip(us, feas)):
        ax_c.text(xi, u + 0.006, f"{100*u:.1f}%", fontsize=7.5, ha="center",
                  va="bottom", color="#333333")
        ax_c.text(xi, 0.008, "feasible" if f else "not feasible", fontsize=7,
                  ha="center", va="bottom", color="white", rotation=90)
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([TASK_LABEL[t].replace(" ", "\n") for t in TASKS], fontsize=7.5)
    ax_c.set_ylim(0, 0.13)
    ax_c.set_ylabel("Best available Wilson 95% upper bound")
    ax_c.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.set_title("(c) Gate threshold feasibility", loc="left", fontweight="bold")

    # ---------- (d) 风险指标冗余 ----------
    rrows = {r["task"]: r for r in red}
    ps = [float(rrows[t]["pearson_ensemble_uncertainty_vs_conformal_width"]) for t in TASKS]
    xd = np.arange(len(TASKS))
    ax_d.bar(xd, ps, 0.5, color=C_ACC, edgecolor="white", linewidth=0.6, zorder=3)
    ax_d.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, zorder=5)
    for xi, p in enumerate(ps):
        ax_d.text(xi, 1.02, f"{p:.3f}", fontsize=7.5, ha="center", va="bottom",
                  color="#333333")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([TASK_LABEL[t].replace(" ", "\n") for t in TASKS], fontsize=7.5)
    ax_d.set_ylim(0.9, 1.12)
    ax_d.set_yticks([0.9, 0.95, 1.0, 1.05])
    ax_d.set_ylabel("Pearson r (ensemble uncertainty vs interval half-width)")
    ax_d.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.set_title("(d) Risk-indicator redundancy", loc="left", fontweight="bold")

    fig.tight_layout()
    save(fig, "Fig_gate_audit")


def fig_coverage_observability():
    rows = read_csv(JER / "source_data" / "predicted_group_coverage_audit.csv")
    labels = [
        "Release intensity\n(upper tercile)",
        "Release height\n(1-40 m)",
        "Spatial boundary\n(X)",
        "Spatial boundary\n(Y)",
    ]
    truth = [float(r["coverage_truth_grouped"]) for r in rows]
    pred = [float(r["coverage_predicted_grouped"]) for r in rows]
    delta = [float(r["delta_percentage_points"]) for r in rows]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.48, 2.9))

    x = np.arange(len(labels))
    w = 0.36
    ax_a.bar(x - w / 2, truth, w, color=C_TRUTH, edgecolor="white", linewidth=0.6,
             label="Grouped by true value", zorder=3)
    ax_a.bar(x + w / 2, pred, w, color=C_PRED, edgecolor="white", linewidth=0.6,
             label="Grouped by predicted value", zorder=3)
    ax_a.axhline(0.90, color="#333333", linestyle="--", linewidth=1.0, zorder=5)
    ax_a.text(-0.42, 0.888, "90% nominal", fontsize=7.5,
              ha="left", va="top", color="#333333")
    for xi, (t, p) in enumerate(zip(truth, pred)):
        ax_a.text(xi - w / 2, t + 0.008, f"{100*t:.1f}", fontsize=7,
                  ha="center", va="bottom", color="#333333")
        ax_a.text(xi + w / 2, p + 0.008, f"{100*p:.1f}", fontsize=7,
                  ha="center", va="bottom", color="#333333")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=7.5)
    ax_a.set_ylim(0.60, 0.96)
    ax_a.set_ylabel("Empirical coverage")
    ax_a.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), frameon=False)
    ax_a.set_title("(a) Subgroup coverage by assignment variable", loc="left",
                   fontweight="bold")

    ax_b.bar(x, delta, 0.5, color=C_ACC, edgecolor="white", linewidth=0.6, zorder=3)
    ax_b.axhline(0, color="#333333", linewidth=0.9, zorder=5)
    ax_b.axhspan(-1.3, 1.3, color=C_GRAY, alpha=0.35, zorder=0)
    ax_b.text(-0.42, 1.38, "±1.3 pp band", fontsize=7.5, ha="left",
              va="bottom", color="#555555")
    for xi, d in enumerate(delta):
        ax_b.text(xi, d + (0.06 if d >= 0 else -0.06), f"{d:+.2f}", fontsize=7.5,
                  ha="center", va="bottom" if d >= 0 else "top", color="#333333")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=7.5)
    ax_b.set_ylim(-1.0, 1.8)
    ax_b.set_ylabel("Predicted − true (percentage points)")
    ax_b.grid(axis="y", linewidth=0.5, color=C_GRAY, alpha=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.set_title("(b) Deviation between the two assignments", loc="left",
                   fontweight="bold")

    fig.tight_layout()
    save(fig, "Fig_coverage_observability")


def main():
    style()
    print("Figure A: gate audit")
    fig_gate_audit()
    print("Figure B: coverage observability")
    fig_coverage_observability()
    print("output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
