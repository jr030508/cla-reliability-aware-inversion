from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
POSITIVE_LABELS = {"Rate_1", "Rate_2", "Rate_3", "Rate_4", "Z"}


class ColumnStandardizer:
    def fit(self, values: np.ndarray) -> "ColumnStandardizer":
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = np.mean(values, axis=0)
        self.scale_ = np.std(values, axis=0)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale_ + self.mean_


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    labels: list[str]
    feature_cols: list[str]
    metadata: pd.DataFrame


class InputTransformer:
    """Log-transform non-negative concentrations and standardize using train data only."""

    def __init__(self, mode: str = "log1p_standard") -> None:
        if mode not in {"log1p_standard", "standard"}:
            raise ValueError(f"Unsupported input transform: {mode}")
        self.mode = mode
        self.scaler = ColumnStandardizer()

    def _base(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "log1p_standard":
            if np.nanmin(X) < 0:
                raise ValueError("Concentrations contain negative values; log1p is invalid.")
            return np.log1p(X)
        return X

    def fit(self, X: np.ndarray) -> "InputTransformer":
        base = self._base(X).reshape(-1, X.shape[-1])
        self.scaler.fit(base)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        base = self._base(X)
        out = self.scaler.transform(base.reshape(-1, X.shape[-1]))
        return out.reshape(X.shape).astype(np.float32)


class TargetTransformer:
    """Use log10 for strictly positive physical quantities and linear scaling for X/Y."""

    def __init__(self, labels: Iterable[str]) -> None:
        self.labels = list(labels)
        self.scaler = ColumnStandardizer()

    def _forward_physical(self, y: np.ndarray) -> np.ndarray:
        z = np.asarray(y, dtype=np.float64).copy()
        for i, label in enumerate(self.labels):
            if label in POSITIVE_LABELS:
                if np.any(z[:, i] <= 0):
                    raise ValueError(f"{label} must be strictly positive for log10 transformation.")
                z[:, i] = np.log10(z[:, i])
        return z

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        self.scaler.fit(self._forward_physical(y))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self._forward_physical(y)).astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        z = self.scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float64))
        for i, label in enumerate(self.labels):
            if label in POSITIVE_LABELS:
                # Exponentiation guarantees physical positivity. Clipping only prevents overflow.
                z[:, i] = np.power(10.0, np.clip(z[:, i], -30.0, 30.0))
        return z


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_dataset(
    csv_path: Path,
    time_steps: int,
    feature_dim: int,
    labels: list[str],
) -> DatasetBundle:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    header = pd.read_csv(csv_path, nrows=0)
    normalized = {str(c).strip().replace("'", ""): c for c in header.columns}
    feature_cols = [str(i) for i in range(1, feature_dim + 1)]
    optional = ["NUM", "YEAR", "DAY", "TIME", "EVENT_ID"]
    required = feature_cols + labels
    missing = [c for c in required if c not in normalized]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected_names = required + [c for c in optional if c in normalized]
    original_names = [normalized[c] for c in selected_names]
    df = pd.read_csv(csv_path, usecols=original_names)
    df.columns = [str(c).strip().replace("'", "") for c in df.columns]

    if len(df) % time_steps:
        raise ValueError(f"Row count {len(df)} is not divisible by time_steps={time_steps}.")
    n_samples = len(df) // time_steps
    first_rows = np.arange(0, len(df), time_steps)

    X = df[feature_cols].to_numpy(dtype=np.float32).reshape(n_samples, time_steps, feature_dim)
    y = df[labels].iloc[first_rows].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("Input features contain NaN or infinite values.")
    if not np.isfinite(y).all():
        raise ValueError("Labels in the first row of at least one sample contain NaN or infinity.")

    meta = pd.DataFrame({"sample_index": np.arange(n_samples, dtype=np.int64)})
    for col in ["NUM", "YEAR", "DAY"]:
        if col in df:
            meta[col.lower()] = df[col].iloc[first_rows].to_numpy()

    if "EVENT_ID" in df:
        event_ids = df["EVENT_ID"].iloc[first_rows].astype(str).to_numpy()
    elif "YEAR" in df and "DAY" in df:
        event_ids = (
            df["YEAR"].iloc[first_rows].astype(str)
            + "_day_"
            + df["DAY"].iloc[first_rows].astype(str)
        ).to_numpy()
    else:
        event_ids = np.repeat("unknown_single_event", n_samples)
    meta["event_id"] = event_ids

    if "TIME" in df:
        time_matrix = df["TIME"].to_numpy().reshape(n_samples, time_steps)
        time_signatures = ["-".join(map(str, row.tolist())) for row in time_matrix]
        meta["time_signature"] = time_signatures

    return DatasetBundle(X=X, y=y, labels=labels, feature_cols=feature_cols, metadata=meta)


def dataset_summary(bundle: DatasetBundle, csv_path: Path) -> dict:
    summary = {
        "data_path": str(csv_path.resolve()),
        "samples": int(bundle.X.shape[0]),
        "time_steps": int(bundle.X.shape[1]),
        "stations": int(bundle.X.shape[2]),
        "labels": bundle.labels,
        "event_count": int(bundle.metadata["event_id"].nunique()),
        "events": sorted(bundle.metadata["event_id"].astype(str).unique().tolist()),
        "zero_fraction": float(np.mean(bundle.X == 0)),
        "feature_min": float(np.min(bundle.X)),
        "feature_max": float(np.max(bundle.X)),
        "label_ranges": {
            label: {"min": float(bundle.y[:, i].min()), "max": float(bundle.y[:, i].max())}
            for i, label in enumerate(bundle.labels)
        },
    }
    if "time_signature" in bundle.metadata:
        summary["time_signature_count"] = int(bundle.metadata["time_signature"].nunique())
    return summary


def split_indices(
    bundle: DatasetBundle,
    split: str,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
    calibration_fraction: float,
    edge_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    indices = np.arange(bundle.X.shape[0])
    fractions = [test_fraction, validation_fraction, calibration_fraction]
    if not all(0 < value < 1 for value in fractions):
        raise ValueError("test, validation and calibration fractions must be in (0, 1).")
    if sum(fractions) >= 1:
        raise ValueError("test + validation + calibration fractions must be < 1.")

    rng = np.random.default_rng(seed)
    split_specific_details: dict = {}

    def random_partition(values: np.ndarray, fraction: float, local_rng: np.random.Generator):
        shuffled = local_rng.permutation(values)
        n_holdout = max(1, int(round(len(values) * fraction)))
        n_holdout = min(n_holdout, len(values) - 1)
        return shuffled[n_holdout:], shuffled[:n_holdout]

    def group_partition(values: np.ndarray, group_values: np.ndarray, fraction: float, local_rng: np.random.Generator):
        unique = local_rng.permutation(np.unique(group_values))
        target = max(1, int(round(len(values) * fraction)))
        selected = []
        selected_count = 0
        for group in unique[:-1]:
            selected.append(group)
            selected_count += int(np.sum(group_values == group))
            if selected_count >= target:
                break
        holdout_mask = np.isin(group_values, selected)
        if not np.any(holdout_mask) or np.all(holdout_mask):
            raise ValueError("Unable to create non-empty grouped train and holdout partitions.")
        return values[~holdout_mask], values[holdout_mask]

    if split == "iid":
        train_val, test = random_partition(indices, test_fraction, rng)
        relative_aux = (validation_fraction + calibration_fraction) / (1.0 - test_fraction)
        train, aux = random_partition(train_val, relative_aux, np.random.default_rng(seed + 1000))
        relative_cal = calibration_fraction / (validation_fraction + calibration_fraction)
        val, calibration = random_partition(aux, relative_cal, np.random.default_rng(seed + 2000))
        groups = None
    elif split == "group_event":
        groups = bundle.metadata["event_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) < 4:
            raise ValueError(
                f"group_event requires at least 4 independent events; found {len(unique_groups)}. "
                "Add EVENT_ID and generate multiple meteorological episodes."
            )
        train_val, test = group_partition(indices, groups, test_fraction, rng)
        relative_aux = (validation_fraction + calibration_fraction) / (1.0 - test_fraction)
        train, aux = group_partition(
            train_val, groups[train_val], relative_aux, np.random.default_rng(seed + 1000)
        )
        relative_cal = calibration_fraction / (validation_fraction + calibration_fraction)
        val, calibration = group_partition(
            aux, groups[aux], relative_cal, np.random.default_rng(seed + 2000)
        )
    elif split == "spatial_block":
        if "X" not in bundle.labels or "Y" not in bundle.labels:
            raise ValueError("spatial_block requires X and Y labels.")
        x = bundle.y[:, bundle.labels.index("X")]
        y = bundle.y[:, bundle.labels.index("Y")]
        x_edges = np.unique(np.quantile(x, np.linspace(0, 1, 5)))
        y_edges = np.unique(np.quantile(y, np.linspace(0, 1, 5)))
        if len(x_edges) < 3 or len(y_edges) < 3:
            raise ValueError("Insufficient spatial variation for block splitting.")
        x_bin = np.clip(np.digitize(x, x_edges[1:-1]), 0, 3)
        y_bin = np.clip(np.digitize(y, y_edges[1:-1]), 0, 3)
        groups = (x_bin * 4 + y_bin).astype(str)
        split_specific_details = {
            "block_definition": "4_by_4_quantile_blocks_on_source_XY",
            "x_block_edges": [float(value) for value in x_edges],
            "y_block_edges": [float(value) for value in y_edges],
            "available_block_count": int(len(np.unique(groups))),
        }
        train_val, test = group_partition(indices, groups, test_fraction, rng)
        relative_aux = (validation_fraction + calibration_fraction) / (1.0 - test_fraction)
        train, aux = group_partition(
            train_val, groups[train_val], relative_aux, np.random.default_rng(seed + 1000)
        )
        relative_cal = calibration_fraction / (validation_fraction + calibration_fraction)
        val, calibration = group_partition(
            aux, groups[aux], relative_cal, np.random.default_rng(seed + 2000)
        )
    elif split == "parameter_edge":
        selected_label = edge_label or ("Rate_1" if "Rate_1" in bundle.labels else "Z")
        if selected_label not in bundle.labels:
            raise ValueError(f"edge_label={selected_label} is not one of {bundle.labels}.")
        target = bundle.y[:, bundle.labels.index(selected_label)]
        lower_threshold = float(np.quantile(target, test_fraction / 2.0))
        upper_threshold = float(np.quantile(target, 1.0 - test_fraction / 2.0))
        lower_mask = target <= lower_threshold
        upper_mask = target >= upper_threshold
        test = indices[lower_mask | upper_mask]
        train_val = indices[~(lower_mask | upper_mask)]
        n_low = int(np.sum(lower_mask))
        n_test = int(len(test))
        if len(train_val) < 4 or n_low == 0 or int(np.sum(upper_mask)) == 0:
            raise ValueError(f"Unable to create non-empty edge partitions for {selected_label}.")
        relative_aux = (validation_fraction + calibration_fraction) / (1.0 - test_fraction)
        train, aux = random_partition(train_val, relative_aux, np.random.default_rng(seed + 1000))
        relative_cal = calibration_fraction / (validation_fraction + calibration_fraction)
        val, calibration = random_partition(aux, relative_cal, np.random.default_rng(seed + 2000))
        groups = None
    else:
        raise ValueError(f"Unknown split: {split}")

    partitions = [set(train), set(val), set(calibration), set(test)]
    overlap = set()
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            overlap |= partitions[i] & partitions[j]
    if overlap:
        raise RuntimeError("Split leakage detected.")

    details = {
        "split": split,
        "seed": int(seed),
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
    }
    if split == "parameter_edge":
        details["edge_label"] = selected_label
        details["edge_rule"] = "lowest_and_highest_equal_tail_quantiles_with_all_boundary_ties_held_out"
        details["lower_quantile_threshold"] = lower_threshold
        details["upper_quantile_threshold"] = upper_threshold
        details["lower_test_max"] = float(np.max(target[lower_mask]))
        details["upper_test_min"] = float(np.min(target[upper_mask]))
        details["lower_edge_samples"] = int(n_low)
        details["upper_edge_samples"] = int(np.sum(upper_mask))
        details["train_target_min"] = float(
            np.min(bundle.y[train, bundle.labels.index(selected_label)])
        )
        details["train_target_max"] = float(
            np.max(bundle.y[train, bundle.labels.index(selected_label)])
        )
    details.update(split_specific_details)
    if groups is not None:
        details["train_groups"] = sorted(np.unique(groups[train]).tolist())
        details["validation_groups"] = sorted(np.unique(groups[val]).tolist())
        details["calibration_groups"] = sorted(np.unique(groups[calibration]).tolist())
        details["test_groups"] = sorted(np.unique(groups[test]).tolist())
    return np.sort(train), np.sort(val), np.sort(calibration), np.sort(test), details


def load_station_coordinates(path: Path, feature_dim: int) -> pd.DataFrame:
    coords = pd.read_csv(path)
    required = {"Station_ID", "X_km", "Y_km"}
    if not required.issubset(coords.columns):
        raise ValueError(f"Station coordinate file must contain {sorted(required)}.")
    coords = coords[list(required)].copy()
    coords["Station_ID"] = coords["Station_ID"].astype(int)
    coords = coords.sort_values("Station_ID")
    expected = np.arange(1, feature_dim + 1)
    if len(coords) != feature_dim or not np.array_equal(coords["Station_ID"].to_numpy(), expected):
        raise ValueError(f"Station_ID must contain every integer from 1 to {feature_dim} exactly once.")
    if not np.isfinite(coords[["X_km", "Y_km"]].to_numpy()).all():
        raise ValueError("Station coordinates contain NaN or infinity.")
    return coords


def station_permutation(
    order: str,
    feature_dim: int,
    seed: int,
    coordinates: pd.DataFrame | None = None,
) -> np.ndarray:
    if order == "original":
        return np.arange(feature_dim)
    if order == "random":
        return np.random.default_rng(seed).permutation(feature_dim)
    if order == "spatial_xy":
        if coordinates is None:
            raise ValueError("spatial_xy order requires --station-coordinates.")
        return np.lexsort((coordinates["Y_km"].to_numpy(), coordinates["X_km"].to_numpy()))
    raise ValueError(f"Unknown station order: {order}")


def build_normalized_adjacency(coordinates: pd.DataFrame, k: int) -> np.ndarray:
    xy = coordinates[["X_km", "Y_km"]].to_numpy(dtype=np.float64)
    if not 1 <= k < len(xy):
        raise ValueError(f"graph_k must be between 1 and {len(xy) - 1}.")
    pairwise = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2))
    indices = np.argsort(pairwise, axis=1)[:, : k + 1]
    distances = np.take_along_axis(pairwise, indices, axis=1)
    positive = distances[:, 1:][distances[:, 1:] > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    adjacency = np.zeros((len(xy), len(xy)), dtype=np.float32)
    for i in range(len(xy)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            weight = np.exp(-((distance / scale) ** 2))
            adjacency[i, j] = max(adjacency[i, j], weight)
            adjacency[j, i] = max(adjacency[j, i], weight)
    adjacency += np.eye(len(xy), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    return (inv_sqrt @ adjacency @ inv_sqrt).astype(np.float32)
