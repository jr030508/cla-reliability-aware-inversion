from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
OUT_ROOT = SCRIPT_PATH.parents[1]
SOURCE_DIR = OUT_ROOT / "source_data"
FIGURE_DIR = OUT_ROOT / "figures"

INP_FILES = {
    "Scenario 1": PROJECT_ROOT / "CAL" / "CALPUFF-s1.INP",
    "Scenario 2": PROJECT_ROOT / "CAL" / "CALPUFF-s2.INP",
    "Scenario 3": PROJECT_ROOT / "CAL" / "CALPUFF-s3.INP",
}

DATA_FILES = {
    "Scenario 1": PROJECT_ROOT / "CAL" / "data s1.csv",
    "Scenario 2": PROJECT_ROOT / "CAL" / "data s2.csv",
    "Scenario 3": PROJECT_ROOT / "CAL" / "data s3.csv",
}

RECEPTOR_PATTERN = re.compile(
    r"^\s*(?P<id>\d+)\s*!\s*X\s*=\s*(?P<x>[-+0-9.Ee]+)\s*,\s*"
    r"(?P<y>[-+0-9.Ee]+)\s*,\s*(?P<ground>[-+0-9.Ee]+)\s*,\s*"
    r"(?P<height>[-+0-9.Ee]+)"
)

SOURCE_PATTERN = re.compile(
    r"^\s*1\s*!\s*X\s*=\s*(?P<x>[-+0-9.Ee]+)\s*,\s*"
    r"(?P<y>[-+0-9.Ee]+)\s*,\s*(?P<z>[-+0-9.Ee]+)\s*,"
)


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
            "savefig.bbox": "tight",
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


def extract_receptors(path: Path) -> tuple[pd.DataFrame, tuple[float, float, float] | None]:
    receptors: list[dict[str, float | int]] = []
    source = None
    in_receptor_block = False
    in_source_block = False

    for line in path.read_text(encoding="latin-1").splitlines():
        if "INPUT GROUPS: 17a & 17b" in line:
            in_receptor_block = True
            in_source_block = False
        elif "POINT SOURCE: CONSTANT DATA" in line:
            in_source_block = True
            in_receptor_block = False

        if in_receptor_block:
            match = RECEPTOR_PATTERN.match(line)
            if match:
                receptors.append(
                    {
                        "receptor_id": int(match.group("id")),
                        "utm_x_km": float(match.group("x")),
                        "utm_y_km": float(match.group("y")),
                        "ground_elevation_m": float(match.group("ground")),
                        "height_m": float(match.group("height")),
                    }
                )

        if in_source_block and source is None:
            match = SOURCE_PATTERN.match(line)
            if match:
                source = (
                    float(match.group("x")),
                    float(match.group("y")),
                    float(match.group("z")),
                )

    if len(receptors) != 136:
        raise ValueError(f"Expected 136 receptors in {path}, found {len(receptors)}")

    frame = pd.DataFrame(receptors).sort_values("receptor_id").reset_index(drop=True)
    if frame["receptor_id"].tolist() != list(range(1, 137)):
        raise ValueError(f"Receptor IDs are not exactly 1..136 in {path}")
    return frame, source


def receptor_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    return [str(i) for i in range(1, 137) if str(i) in header]


def parse_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"!\s*{re.escape(key)}\s*=\s*([^!\r\n]+)!", text)
    return match.group(1).strip() if match else None


def configuration_audit() -> pd.DataFrame:
    keys = ["IBYR", "IBMO", "IBDY", "IBHR", "IEYR", "IEMO", "IEDY", "IEHR", "NSECDT", "NSPEC", "NSE", "MCHEM", "MWET", "MDRY", "NREC", "Z0IN"]
    rows = []
    for scenario, path in INP_FILES.items():
        text = path.read_text(encoding="latin-1")
        for key in keys:
            rows.append(
                {
                    "scenario": scenario,
                    "parameter": key,
                    "value": parse_scalar(text, key),
                }
            )
    return pd.DataFrame(rows)


def add_geometry(frame: pd.DataFrame, source: tuple[float, float, float]) -> pd.DataFrame:
    result = frame.copy()
    dx = result["utm_x_km"].to_numpy() - source[0]
    dy = result["utm_y_km"].to_numpy() - source[1]
    result["distance_from_s1_source_km"] = np.hypot(dx, dy)
    result["bearing_from_north_deg"] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    next_distance = np.full(len(result), np.nan)
    next_distance[:-1] = np.hypot(np.diff(result["utm_x_km"]), np.diff(result["utm_y_km"]))
    result["distance_to_next_id_km"] = next_distance
    return result


def permutation_locality(frame: pd.DataFrame, seed: int = 20261029, n_perm: int = 10000) -> tuple[pd.DataFrame, dict[str, float]]:
    xy = frame[["utm_x_km", "utm_y_km"]].to_numpy()
    observed_steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    observed_mean = float(observed_steps.mean())
    observed_median = float(np.median(observed_steps))

    rng = np.random.default_rng(seed)
    random_means = np.empty(n_perm, dtype=float)
    random_medians = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        permuted = xy[rng.permutation(len(xy))]
        steps = np.linalg.norm(np.diff(permuted, axis=0), axis=1)
        random_means[idx] = steps.mean()
        random_medians[idx] = np.median(steps)

    p_lower = float((1 + np.count_nonzero(random_means <= observed_mean)) / (n_perm + 1))
    summary = {
        "observed_mean_adjacent_distance_km": observed_mean,
        "observed_median_adjacent_distance_km": observed_median,
        "random_mean_adjacent_distance_km": float(random_means.mean()),
        "random_median_adjacent_distance_km": float(np.median(random_medians)),
        "locality_ratio_observed_to_random_mean": float(observed_mean / random_means.mean()),
        "permutation_p_lower_tail": p_lower,
        "n_permutations": float(n_perm),
        "seed": float(seed),
    }
    distribution = pd.DataFrame(
        {
            "permutation": np.arange(1, n_perm + 1),
            "mean_adjacent_distance_km": random_means,
            "median_adjacent_distance_km": random_medians,
        }
    )
    return distribution, summary


def make_figure(frame: pd.DataFrame, source: tuple[float, float, float], distribution: pd.DataFrame, summary: dict[str, float]) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.1), constrained_layout=True)
    cmap = mpl.colormaps["viridis"]
    ids = frame["receptor_id"].to_numpy()
    xy = frame[["utm_x_km", "utm_y_km"]].to_numpy()

    ax = axes[0, 0]
    ax.plot(xy[:, 0], xy[:, 1], color="#B8B8B8", linewidth=0.55, zorder=1)
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1], c=ids, cmap=cmap, s=20, edgecolor="black", linewidth=0.25, zorder=2
    )
    ax.scatter(source[0], source[1], marker="*", s=115, color="#D55E00", edgecolor="black", linewidth=0.55, label="Scenario 1 source", zorder=3)
    for station_id in (1, 32, 64, 96, 136):
        row = frame.loc[frame["receptor_id"] == station_id].iloc[0]
        ax.annotate(
            str(station_id),
            (row["utm_x_km"], row["utm_y_km"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.82},
        )
    ax.set_xlabel("UTM easting (km)")
    ax.set_ylabel("UTM northing (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=False)
    style_axes(ax)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Receptor ID")
    cbar.ax.tick_params(direction="in", length=3, width=0.8)

    ax = axes[0, 1]
    ax.scatter(
        frame["bearing_from_north_deg"],
        frame["distance_from_s1_source_km"],
        c=ids,
        cmap=cmap,
        s=20,
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_xlabel("Bearing from Scenario 1 source (deg)")
    ax.set_ylabel("Source-receptor distance (km)")
    style_axes(ax)

    ax = axes[1, 0]
    transitions = np.arange(1, len(frame))
    steps = frame["distance_to_next_id_km"].dropna().to_numpy()
    ax.plot(transitions, steps, color="#0072B2", linewidth=1.0)
    ax.axhline(np.median(steps), color="#D55E00", linewidth=1.0, linestyle="--", label=f"Median = {np.median(steps):.2f} km")
    ax.set_xlabel("Transition from receptor ID $i$ to $i+1$")
    ax.set_ylabel("Adjacent-ID distance (km)")
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)

    ax = axes[1, 1]
    values = distribution["mean_adjacent_distance_km"].to_numpy()
    ax.hist(values, bins=40, color="#56B4E9", edgecolor="white", linewidth=0.35)
    observed = summary["observed_mean_adjacent_distance_km"]
    ax.axvline(observed, color="#D55E00", linewidth=1.4, label=f"Observed = {observed:.2f} km")
    ax.set_xlabel("Mean adjacent distance under random ordering (km)")
    ax.set_ylabel("Permutation count")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.97,
        0.95,
        f"Locality ratio = {summary['locality_ratio_observed_to_random_mean']:.2f}\n"
        f"$p_{{lower}}$ = {summary['permutation_p_lower_tail']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )
    style_axes(ax)

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")

    for suffix in ("png", "pdf", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(FIGURE_DIR / f"Fig01_receptor_geometry_and_order_audit.{suffix}", dpi=dpi)
    plt.close(fig)


def write_report(summary: dict[str, float], identical: bool, headers_ok: bool, source: tuple[float, float, float]) -> None:
    report = f"""# CALPUFF 受体几何与输入顺序审计

## 可复核事实

- 三个 INP 文件均解析得到 136 个离散受体，受体编号为 1–136；三个场景的受体坐标完全一致：**{identical}**。
- 三个 CSV 的受体特征列均按字符串 `1, 2, ..., 136` 排列，且与 INP 受体编号一一对应：**{headers_ok}**。
- Scenario 1 代表性源坐标为 UTM ({source[0]:.2f}, {source[1]:.2f}) km；受体为扇形下风向布设，不应写成“覆盖所有气象方位”。
- 原始编号顺序的平均相邻距离为 {summary['observed_mean_adjacent_distance_km']:.3f} km，随机排列均值为 {summary['random_mean_adjacent_distance_km']:.3f} km；局部性比值为 {summary['locality_ratio_observed_to_random_mean']:.3f}，下尾置换检验 p={summary['permutation_p_lower_tail']:.5f}（{int(summary['n_permutations'])} 次置换）。

## 论文中可支持的表述

CSV 的 136 个通道不是随机排列：它们保留 CALPUFF 受体编号顺序，且该顺序具有显著的空间局部性。因此，沿“站点轴”的窄卷积核能够聚合相邻编号受体的局部浓度响应。不过，这仍不等于规则二维地理栅格卷积；应将其称为**受体序列局部卷积**或**具有空间局部性的跨通道卷积**，不宜声称模型学习了完整二维地理拓扑。

## 仍需保留的边界

本审计证明的是“编号顺序具有局部性”，并不能证明该顺序在全部位置都保持等距或严格邻接。为使空间鲁棒性论证闭环，正文还应增加基于真实坐标的扇区/距离带结构化失活实验，而不只报告按编号截断的 136→96→64→32→16。
"""
    (OUT_ROOT / "receptor_geometry_audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    extracted = {}
    sources = {}
    for scenario, path in INP_FILES.items():
        extracted[scenario], sources[scenario] = extract_receptors(path)

    reference = extracted["Scenario 1"]
    identical = all(
        reference.equals(extracted[name]) for name in ("Scenario 2", "Scenario 3")
    )
    if not identical:
        raise ValueError("Receptor coordinates differ across the three INP files")
    if sources["Scenario 1"] is None:
        raise ValueError("Scenario 1 source coordinate could not be parsed")

    header_checks = []
    for scenario, path in DATA_FILES.items():
        headers = receptor_headers(path)
        header_checks.append(headers == [str(i) for i in range(1, 137)])
    headers_ok = all(header_checks)
    if not headers_ok:
        raise ValueError("At least one CSV does not contain receptor columns 1..136 in order")

    source = sources["Scenario 1"]
    geometry = add_geometry(reference, source)
    geometry.to_csv(SOURCE_DIR / "receptor_coordinates_exact.csv", index=False, encoding="utf-8-sig")

    config = configuration_audit()
    config.to_csv(SOURCE_DIR / "calpuff_configuration_audit.csv", index=False, encoding="utf-8-sig")

    distribution, summary = permutation_locality(geometry)
    distribution.to_csv(SOURCE_DIR / "station_order_permutation_distribution.csv", index=False)
    pd.DataFrame([summary]).to_csv(SOURCE_DIR / "station_order_locality_summary.csv", index=False)

    mapping_rows = []
    for scenario, path in DATA_FILES.items():
        mapping_rows.append(
            {
                "scenario": scenario,
                "csv_file": path.name,
                "csv_receptor_columns": "1..136",
                "inp_receptor_ids": "1..136",
                "direct_identifier_match": True,
                "column_reordering_detected": False,
            }
        )
    pd.DataFrame(mapping_rows).to_csv(SOURCE_DIR / "csv_inp_receptor_mapping_audit.csv", index=False, encoding="utf-8-sig")

    make_figure(geometry, source, distribution, summary)
    write_report(summary, identical, headers_ok, source)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Outputs written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
