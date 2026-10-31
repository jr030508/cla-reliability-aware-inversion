# -*- coding: utf-8 -*-
"""
纯重算：源—受体几何异质性审计（场景 1 固定源点；场景 2/3 源点在 5 km x 5 km 框内重采样）。

不运行 CALPUFF，只读取已落盘的受体精确坐标与场景 3 测试集源点坐标。
输出：jer_revision/source_data/source_receptor_geometry_audit.csv
"""
import csv, io, math, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(BASE, "source_data", "receptor_coordinates_exact.csv")
PRED = os.path.normpath(os.path.join(
    BASE, "..", "ane_revision", "outputs", "existing_analysis",
    "ensemble_conformal_predictions.csv"))
OUT = os.path.join(BASE, "source_data", "source_receptor_geometry_audit.csv")


def bearing_deg(sx, sy, rx, ry):
    """自北顺时针方位角"""
    return math.degrees(math.atan2(rx - sx, ry - sy)) % 360.0


def main():
    with io.open(REC, encoding="utf-8-sig") as fh:
        recs = list(csv.DictReader(fh))
    rec = [(float(r["utm_x_km"]), float(r["utm_y_km"])) for r in recs]
    dist = sorted(float(r["distance_from_s1_source_km"]) for r in recs)
    bear = sorted(float(r["bearing_from_north_deg"]) for r in recs)
    n = len(dist)
    med = dist[n // 2] if n % 2 else (dist[n // 2 - 1] + dist[n // 2]) / 2

    srcs = []
    with io.open(PRED, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["scenario"] == "3" and r["True_X"] and r["True_Y"]:
                srcs.append((float(r["True_X"]), float(r["True_Y"])))

    xs = [s[0] for s in srcs]
    ys = [s[1] for s in srcs]
    lo, hi = bear[0], bear[-1]

    nearest = []
    outside = []
    for sx, sy in srcs:
        nearest.append(min(math.hypot(sx - rx, sy - ry) for rx, ry in rec))
        out = sum(1 for rx, ry in rec
                  if not (lo <= bearing_deg(sx, sy, rx, ry) <= hi))
        outside.append(out / len(rec))
    nearest.sort()
    outside.sort()

    rows = [
        {"metric": "scenario1_receptors", "value": len(rec)},
        {"metric": "scenario1_source_to_receptor_distance_min_km", "value": round(dist[0], 3)},
        {"metric": "scenario1_source_to_receptor_distance_median_km", "value": round(med, 3)},
        {"metric": "scenario1_source_to_receptor_distance_max_km", "value": round(dist[-1], 3)},
        {"metric": "scenario1_bearing_min_deg", "value": round(lo, 2)},
        {"metric": "scenario1_bearing_max_deg", "value": round(hi, 2)},
        {"metric": "scenario23_sampled_sources", "value": len(srcs)},
        {"metric": "scenario23_source_x_min_km", "value": round(min(xs), 3)},
        {"metric": "scenario23_source_x_max_km", "value": round(max(xs), 3)},
        {"metric": "scenario23_source_y_min_km", "value": round(min(ys), 3)},
        {"metric": "scenario23_source_y_max_km", "value": round(max(ys), 3)},
        {"metric": "scenario23_nearest_receptor_distance_min_km", "value": round(nearest[0], 3)},
        {"metric": "scenario23_nearest_receptor_distance_median_km",
         "value": round(nearest[len(nearest) // 2], 3)},
        {"metric": "scenario23_nearest_receptor_distance_max_km", "value": round(nearest[-1], 3)},
        {"metric": "scenario23_outside_sector_fraction_mean",
         "value": round(sum(outside) / len(outside), 4)},
        {"metric": "scenario23_outside_sector_fraction_median",
         "value": round(outside[len(outside) // 2], 4)},
        {"metric": "scenario23_outside_sector_fraction_max", "value": round(outside[-1], 4)},
        {"metric": "note_plume_sector_definition",
         "value": f"bearing window [{lo:.2f}, {hi:.2f}] deg from north, "
                  f"i.e. the bearing span of the receptor arc seen from the Scenario 1 source"},
    ]

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print("written:", OUT)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
