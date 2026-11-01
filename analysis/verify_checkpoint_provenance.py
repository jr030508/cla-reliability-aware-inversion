# -*- coding: utf-8 -*-
"""判定 CLA 三个场景的权重与结果到底来自哪一批运行。

论文报告的场景3 数字：释放速率 Log10MAE 0.0583、水平定位 0.12701 km、释放高度 0.0760。
场景1：四阶段释放速率平均 0.02029±0.00256。
"""
import csv, json, os

ROOT = r"C:\Users\12130\Desktop\CLA小论文撰写\experiments_v2"
BATCHES = {
    "frozen_revalidation(9/16)": os.path.join(ROOT, "outputs_frozen_manuscript"),
    "legacy_outputs(9/8)": os.path.join(ROOT, "outputs", "outputs_linux_robustness"),
    "legacy_s1(9/9)": os.path.join(ROOT, "outputs", "optputs_linux_s1"),
}
SEEDS = [42, 43, 44]


def read_metrics(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    d = {r["Parameter"]: r for r in rows}
    return d


def seed_numbers(base, scenario, model="CLA"):
    """返回 (rate_log10mae_mean, xy_dist_km_mean, z_log10mae_mean, 三个种子明细)。"""
    rate_vals, xy_vals, z_vals, detail = [], [], [], []
    for seed in SEEDS:
        p = os.path.join(base, "scenario_%d" % scenario, "iid", "order_original", model, "seed_%d" % seed)
        raw = os.path.join(p, "raw_predictions.csv")
        if not os.path.isfile(raw):
            detail.append("seed_%d: 无 raw_predictions" % seed)
            continue
        rows = list(csv.DictReader(open(raw, encoding="utf-8-sig")))
        cols = rows[0].keys()

        def mean(key):
            if key in cols:
                return sum(float(r[key]) for r in rows) / len(rows)
            return float("nan")

        # 场景1 反演四段释放速率（无 X/Y/Z），场景2 反演位置与高度（无 Rate），场景3 两者都有
        rate_keys = ["LOGAE_Rate_%d" % i for i in (1, 2, 3, 4)]
        rate = (sum(mean(k) for k in rate_keys) / 4
                if all(k in cols for k in rate_keys) else float("nan"))
        xy = mean("XY_DISTANCE_km")
        z = mean("LOGAE_Z")
        rate_vals.append(rate); xy_vals.append(xy); z_vals.append(z)
        detail.append("seed_%d: rate %.4f, XY %.4f, Z %.4f" % (seed, rate, xy, z))
    if not rate_vals:
        return None
    n = len(rate_vals)
    return (sum(rate_vals) / n, sum(xy_vals) / n, sum(z_vals) / n, detail)


def main():
    print("=" * 78)
    print("A. preflight 是否引用冻结参数文件")
    print("=" * 78)
    for name, base in BATCHES.items():
        for s in (1, 2, 3):
            pf = os.path.join(base, "preflight_scenario_%d_iid.json" % s)
            if not os.path.isfile(pf):
                continue
            d = json.load(open(pf, encoding="utf-8"))
            print("%-28s scenario_%d  parameter_file=%s" % (name, s, d.get("parameter_file", "—（无此字段）")))

    print()
    print("=" * 78)
    print("B. 场景3 CLA 结果（三子平均） vs 论文 0.0583 / 0.12701 / 0.0760")
    print("=" * 78)
    for name, base in BATCHES.items():
        res = seed_numbers(base, 3)
        if res is None:
            print("%-28s 无场景3 raw_predictions" % name)
            continue
        r, xy, z, detail = res
        print("%-28s rate=%.4f  XY=%.4f km  Z=%.4f" % (name, r, xy, z))
        for d_ in detail:
            print("      %s" % d_)

    print()
    print("=" * 78)
    print("C. 场景1 CLA 结果（三子平均，四阶段） vs 论文 0.02029")
    print("=" * 78)
    for name, base in BATCHES.items():
        res = seed_numbers(base, 1)
        if res is None:
            print("%-28s 无场景1 raw_predictions" % name)
            continue
        r, xy, z, detail = res
        print("%-28s rate=%.5f" % (name, r))
        for d_ in detail:
            print("      %s" % d_)

    print()
    print("=" * 78)
    print("D. 场景3 权重文件是否存在")
    print("=" * 78)
    for name, base in BATCHES.items():
        found = []
        for seed in SEEDS:
            p = os.path.join(base, "scenario_3", "iid", "order_original", "CLA", "seed_%d" % seed, "best_model.h5")
            if os.path.isfile(p):
                found.append("%d(%.1fMB)" % (seed, os.path.getsize(p) / 1048576))
        print("%-28s %s" % (name, ", ".join(found) if found else "无权重文件"))


if __name__ == "__main__":
    main()
