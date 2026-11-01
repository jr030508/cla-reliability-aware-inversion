# -*- coding: utf-8 -*-
"""
MAJOR-4 响应：为"任务依赖负迁移"补齐统计证据。

对每个 (模型, 任务) 组合，用逐样本配对 Wilcoxon 符号秩检验比较
  联合七参数训练 (joint) vs 任务匹配单任务训练 (single-task)，
逐种子（42/43/44）各检验一次，并对 3 个种子做合并检验；
多重比较用 Holm-Bonferroni 校正（族 = 2 模型 × 3 任务 = 6 个比较）。

每样本任务度量（与正文聚合口径一致，见 analyze_completed_revision.task_scores）：
  release_rate         = mean(LOGAE_Rate_1..4)
  horizontal_location  = XY_DISTANCE_km
  release_height       = LOGAE_Z
差值 d = joint - single（d > 0 表示联合训练误差更高，即负迁移）。

效应量：配对秩双列相关 r_rb = (W+ - W-)/(W+ + W-)，及
平均配对差值的 bootstrap 95% CI（10,000 次重抽样）。

输出：ane_revision/outputs/completed_analysis/task_decomposition_wilcoxon_tests.csv
"""
from __future__ import annotations

import csv
import io
import os
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, rankdata

EXV = Path(r"C:\Users\12130\Desktop\CLA小论文撰写\experiments_v2")
REV = EXV / "ane_revision"
OUT_CSV = REV / "outputs" / "completed_analysis" / "task_decomposition_wilcoxon_tests.csv"
N_BOOT = 10000
RNG = np.random.default_rng(20261101)

TASKS = ["release_rate", "horizontal_location", "release_height"]
MODELS = ["CLA", "LSTM_Attention"]
SEEDS = ["42", "43", "44"]


def read_scores():
    path = REV / "outputs" / "completed_analysis" / "task_level_scores.csv"
    with io.open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def per_sample_metric(row, data):
    task = row["task"]
    if task == "release_rate":
        return np.mean([data[f"LOGAE_Rate_{i}"] for i in (1, 2, 3, 4)], axis=0)
    if task == "horizontal_location":
        return data["XY_DISTANCE_km"]
    if task == "release_height":
        return data["LOGAE_Z"]
    raise ValueError(task)


def load_predictions(run_dir):
    """返回 {sample_index: array} 形式的列字典。"""
    path = Path(run_dir) / "raw_predictions.csv"
    cols = {}
    idx = []
    with io.open(path, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = [f for f in reader.fieldnames
                  if f == "sample_index" or f.startswith(("LOGAE", "XY_DISTANCE"))]
        for f in fields:
            cols[f] = []
        for r in reader:
            idx.append(r["sample_index"])
            for f in fields:
                cols[f].append(float(r[f]) if r[f] != "" else np.nan)
    return idx, {k: np.asarray(v) for k, v in cols.items()}


def rank_biserial(diffs):
    """配对秩双列相关，diffs 已滤零。"""
    n = len(diffs)
    ranks = rankdata(np.abs(diffs))
    w_plus = ranks[diffs > 0].sum()
    w_minus = ranks[diffs < 0].sum()
    return (w_plus - w_minus) / (w_plus + w_minus)


def boot_ci_mean(d, n_boot=N_BOOT):
    n = len(d)
    means = np.empty(n_boot)
    chunk = 500
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = RNG.integers(0, n, size=(e - s, n))
        means[s:e] = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    """Holm-Bonferroni，返回校正后 p。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    rows = read_scores()
    joint_dirs = {}
    single_dirs = {}
    for r in rows:
        if r["experiment"] == "baseline" and r["scenario"] == "3" and r["model"] in MODELS:
            joint_dirs.setdefault((r["model"], r["seed"]), r["run_dir"])
        if r["experiment"] == "task_decomposition" and r["model"] in MODELS:
            single_dirs[(r["model"], r["task"], r["seed"])] = r["run_dir"]

    results = []
    for seed in SEEDS:
        for model in MODELS:
            jdir = joint_dirs[(model, seed)]
            jidx, jdata = load_predictions(jdir)
            for task in TASKS:
                sdir = single_dirs[(model, task, seed)]
                sidx, sdata = load_predictions(sdir)
                assert jidx == sidx, f"样本未配对: {model}/{task}/{seed}"
                a = per_sample_metric({"task": task}, sdata)
                b = per_sample_metric({"task": task}, jdata)
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                d = b - a  # joint - single（正值 = 负迁移）

                stat, p = wilcoxon(a, b, zero_method="wilcox",
                                   alternative="two-sided", method="approx")
                rrb = rank_biserial(d)
                lo, hi = boot_ci_mean(d)
                results.append({
                    "seed": seed, "model": model, "task": task, "n_pairs": len(d),
                    "single_mean": float(a.mean()), "joint_mean": float(b.mean()),
                    "mean_diff_joint_minus_single": float(d.mean()),
                    "diff_ci95_low": float(lo),
                    "diff_ci95_high": float(hi),
                    "median_diff": float(np.median(d)),
                    "relative_change_percent": float(100 * (b.mean() - a.mean()) / a.mean()),
                    "wilcoxon_W": float(stat), "wilcoxon_p": float(p),
                    "rank_biserial_r": float(rrb),
                })

    # 合并三种子的逐样本差值（同 split、同 seed 分组，样本不重叠）→ 每组合一个总检验
    for model in MODELS:
        jidx, _ = load_predictions(joint_dirs[(model, SEEDS[0])])
        for task in TASKS:
            pooled = []
            for seed in SEEDS:
                sidx, sdata = load_predictions(single_dirs[(model, task, seed)])
                _, jdata = load_predictions(joint_dirs[(model, seed)])
                a = per_sample_metric({"task": task}, sdata)
                b = per_sample_metric({"task": task}, jdata)
                ok = ~(np.isnan(a) | np.isnan(b))
                pooled.append((b - a)[ok])
            d = np.concatenate(pooled)
            a = d.copy()
            stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                               method="approx")
            rrb = rank_biserial(d)
            lo, hi = boot_ci_mean(d)
            results.append({
                "seed": "pooled", "model": model, "task": task, "n_pairs": len(d),
                "single_mean": np.nan, "joint_mean": np.nan,
                "mean_diff_joint_minus_single": float(d.mean()),
                "diff_ci95_low": float(lo),
                "diff_ci95_high": float(hi),
                "median_diff": float(np.median(d)),
                "relative_change_percent": np.nan,
                "wilcoxon_W": float(stat), "wilcoxon_p": float(p),
                "rank_biserial_r": float(rrb),
            })

    # Holm 校正：族 = 同一种子内的 2 模型 × 3 任务；pooled 单独一族
    for seed in SEEDS + ["pooled"]:
        sel = [i for i, r in enumerate(results) if r["seed"] == seed]
        ps = [results[i]["wilcoxon_p"] for i in sel]
        adj = holm(np.asarray(ps))
        for i, pa in zip(sel, adj):
            results[i]["holm_p"] = float(pa)

    fields = list(results[0].keys())
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print("written:", OUT_CSV)
    hdr = f"{'seed':>7} {'model':<15} {'task':<20} {'rel%':>8} {'p':>12} {'holm_p':>12} {'r_rb':>7}"
    print(hdr)
    for r in results:
        rel = f"{r['relative_change_percent']:.2f}" if not np.isnan(r["relative_change_percent"]) else "  --"
        print(f"{r['seed']:>7} {r['model']:<15} {r['task']:<20} {rel:>8} "
              f"{r['wilcoxon_p']:.3e} {r['holm_p']:.3e} {r['rank_biserial_r']:>7.3f}")


if __name__ == "__main__":
    main()
