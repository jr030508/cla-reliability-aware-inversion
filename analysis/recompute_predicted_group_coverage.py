# -*- coding: utf-8 -*-
"""
纯重算：按「模型预测值」而不是「真值」划分条件子群后的 90% 全局分割保形覆盖率。

目的：回答审稿意见 M3 —— 条件覆盖分组若依赖真值则在业务中不可用。
本脚本不训练模型、不运行 CALPUFF，仅读取已落盘的集成保形预测明细
(ane_revision/outputs/existing_analysis/ensemble_conformal_predictions.csv,
场景 3 / CLA / 4000 个测试样本)，用同一套数值边界分别按真值与按预测值
分配样本，重算各子群的经验覆盖率。

输出：jer_revision/source_data/predicted_group_coverage_audit.csv
"""
import csv, io, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.normpath(os.path.join(
    BASE, "..", "ane_revision", "outputs", "existing_analysis",
    "ensemble_conformal_predictions.csv"))
OUT = os.path.join(BASE, "source_data", "predicted_group_coverage_audit.csv")


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def q_index(sorted_vals, k):
    """按索引取分位数（与 np.quantile(..., method='nearest') 等价）"""
    return sorted_vals[int(round((len(sorted_vals) - 1) * k))]


def load(scenario="3"):
    with io.open(SRC, encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["scenario"] == scenario]


def covered(rec, p, level="90"):
    lo, hi = f(rec[f"Lower{level}_{p}"]), f(rec[f"Upper{level}_{p}"])
    tv = f(rec[f"True_{p}"])
    if lo is None or hi is None or tv is None:
        return None
    return lo <= tv <= hi


def rate(mask, rows, p):
    vals = [covered(r, p) for r, m in zip(rows, mask) if m and covered(r, p) is not None]
    return len(vals), (sum(vals) / len(vals) if vals else float("nan"))


def main():
    rows = load()
    print(f"scenario 3 test samples: {len(rows)}")
    for p in ("Rate_1", "X", "Y", "Z"):
        n, c = rate([True] * len(rows), rows, p)
        print(f"  marginal 90% coverage {p}: {c:.4f} (n={n})")

    xs = sorted(f(r["True_X"]) for r in rows)
    ys = sorted(f(r["True_Y"]) for r in rows)
    rs = sorted(f(r["True_Rate_1"]) for r in rows)
    t2 = q_index(rs, 2.0 / 3.0)
    xlo, xhi = q_index(xs, 0.2), q_index(xs, 0.8)
    ylo, yhi = q_index(ys, 0.2), q_index(ys, 0.8)

    specs = [
        ("Rate_1", "release_intensity", "high", f"Rate_1 >= {t2:.6g}",
         lambda r, truth: (f(r["True_Rate_1" if truth else "EnsembleMean_Rate_1"]) >= t2)),
        ("Z", "height_range", "1-40m", "Z <= 40 m",
         lambda r, truth: (f(r["True_Z" if truth else "EnsembleMean_Z"]) <= 40.0)),
        ("X", "spatial_region", "boundary",
         f"outside X[{xlo:.4g},{xhi:.4g}] or Y[{ylo:.4g},{yhi:.4g}]",
         lambda r, truth: not (
             xlo <= f(r["True_X" if truth else "EnsembleMean_X"]) <= xhi and
             ylo <= f(r["True_Y" if truth else "EnsembleMean_Y"]) <= yhi)),
        ("Y", "spatial_region", "boundary",
         f"outside X[{xlo:.4g},{xhi:.4g}] or Y[{ylo:.4g},{yhi:.4g}]",
         lambda r, truth: not (
             xlo <= f(r["True_X" if truth else "EnsembleMean_X"]) <= xhi and
             ylo <= f(r["True_Y" if truth else "EnsembleMean_Y"]) <= yhi)),
    ]

    out = []
    for p, gtype, group, bound, rule in specs:
        n_t, c_t = rate([rule(r, True) for r in rows], rows, p)
        n_p, c_p = rate([rule(r, False) for r in rows], rows, p)
        out.append({
            "parameter": p, "group_type": gtype, "group": group,
            "boundary_source": "truth distribution of the test set",
            "boundary_value": bound,
            "n_truth_grouped": n_t, "coverage_truth_grouped": round(c_t, 4),
            "n_predicted_grouped": n_p, "coverage_predicted_grouped": round(c_p, 4),
            "delta_percentage_points": round(100 * (c_p - c_t), 2),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\nwritten:", OUT)
    for r in out:
        print(r)
    print(f"\nmax |delta| = {max(abs(r['delta_percentage_points']) for r in out):.2f} pp")


if __name__ == "__main__":
    main()
