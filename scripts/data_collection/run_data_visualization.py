#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_DIR = "/home/ubuntu/Junyi/LumiGrasp/data/pickcube-v1"


@dataclass(frozen=True)
class SampleRecord:
    metadata_path: Path
    cube_range_name: str
    cube_range_x: tuple[float, float] | None
    cube_range_y: tuple[float, float] | None
    cube_position: np.ndarray
    eef_position: np.ndarray
    eef_yaw: float | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_position(value: Any, *, field_name: str, path: Path) -> np.ndarray:
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"{path}: {field_name} must be a length-3 position, got shape {position.shape}")
    return position


def _tuple_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    values = tuple(float(v) for v in value)
    if len(values) != 2:
        return None
    return values


def _sample_metadata_paths(sample_parent: Path, *, max_samples: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for sample_dir in sorted(sample_parent.glob("sample_*")):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        paths.append(metadata_path)
        if max_samples is not None and len(paths) >= max_samples:
            break
    return paths


def _metadata_paths(input_path: Path, *, max_samples: int | None = None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if (input_path / "metadata.json").is_file():
        return [input_path / "metadata.json"]

    direct_sample_paths = _sample_metadata_paths(input_path, max_samples=max_samples)
    if direct_sample_paths:
        return direct_sample_paths

    paths: list[Path] = []
    for child_dir in sorted(path for path in input_path.iterdir() if path.is_dir()):
        paths.extend(_sample_metadata_paths(child_dir, max_samples=max_samples))
    return paths


def _actual_eef_position(metadata: dict[str, Any], path: Path) -> np.ndarray:
    poses = metadata.get("state", {}).get("poses", {})
    pose = poses.get("gripper0_right_eef")
    if not pose or "position" not in pose:
        raise KeyError(f"{path}: state.poses.gripper0_right_eef.position not found")
    return _as_position(pose["position"], field_name="state.poses.gripper0_right_eef.position", path=path)


def _sampled_eef_position(metadata: dict[str, Any], path: Path) -> np.ndarray:
    sample = metadata.get("initial_eef_sample", {})
    if "position" not in sample:
        raise KeyError(f"{path}: initial_eef_sample.position not found")
    return _as_position(sample["position"], field_name="initial_eef_sample.position", path=path)


def _record_from_metadata(path: Path, *, eef_source: str) -> SampleRecord:
    metadata = _read_json(path)
    cube_geom = metadata.get("state", {}).get("cube_geom", {})
    if "position" not in cube_geom:
        raise KeyError(f"{path}: state.cube_geom.position not found")

    cube_range = metadata.get("cube_range") or {}
    eef_sample = metadata.get("initial_eef_sample") or {}
    eef_position = _sampled_eef_position(metadata, path) if eef_source == "sampled" else _actual_eef_position(metadata, path)

    return SampleRecord(
        metadata_path=path,
        cube_range_name=str(cube_range.get("name") or "unknown"),
        cube_range_x=_tuple_range(cube_range.get("object_x_range")),
        cube_range_y=_tuple_range(cube_range.get("object_y_range")),
        cube_position=_as_position(cube_geom["position"], field_name="state.cube_geom.position", path=path),
        eef_position=eef_position,
        eef_yaw=float(eef_sample["yaw"]) if "yaw" in eef_sample else None,
    )


def load_records(input_path: Path, *, eef_source: str, max_samples: int | None = None) -> tuple[list[SampleRecord], list[str]]:
    paths = _metadata_paths(input_path, max_samples=max_samples)
    records: list[SampleRecord] = []
    errors: list[str] = []
    for path in paths:
        try:
            records.append(_record_from_metadata(path, eef_source=eef_source))
        except Exception as exc:  # Keep long batch jobs useful when one sample is malformed.
            errors.append(str(exc))
    return records, errors


def _find_summary_config(input_path: Path) -> dict[str, Any] | None:
    start = input_path if input_path.is_dir() else input_path.parent
    for directory in (start, *start.parents):
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            try:
                summary = _read_json(summary_path)
            except json.JSONDecodeError:
                continue
            config = summary.get("config")
            if isinstance(config, dict):
                return config
    return None


def _bounds_from_values(values: np.ndarray) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, 1e-3)
        return low - pad, high + pad
    pad = (high - low) * 0.02
    return low - pad, high + pad


def _cube_bounds(records: list[SampleRecord]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_ranges = [record.cube_range_x for record in records if record.cube_range_x is not None]
    y_ranges = [record.cube_range_y for record in records if record.cube_range_y is not None]
    cube_positions = np.asarray([record.cube_position for record in records], dtype=np.float64)
    if x_ranges and y_ranges:
        x_bounds = (min(r[0] for r in x_ranges), max(r[1] for r in x_ranges))
        y_bounds = (min(r[0] for r in y_ranges), max(r[1] for r in y_ranges))
        return x_bounds, y_bounds
    return _bounds_from_values(cube_positions[:, 0]), _bounds_from_values(cube_positions[:, 1])


def _eef_bounds(
    records: list[SampleRecord],
    input_path: Path,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], str]:
    config = _find_summary_config(input_path)
    if config and "xyz_low" in config and "xyz_high" in config:
        xyz_low = np.asarray(config["xyz_low"], dtype=np.float64)
        xyz_high = np.asarray(config["xyz_high"], dtype=np.float64)
        if xyz_low.shape == (3,) and xyz_high.shape == (3,):
            return (
                (float(xyz_low[0]), float(xyz_high[0])),
                (float(xyz_low[1]), float(xyz_high[1])),
                (float(xyz_low[2]), float(xyz_high[2])),
                "summary.json config",
            )

    eef_positions = np.asarray([record.eef_position for record in records], dtype=np.float64)
    return (
        _bounds_from_values(eef_positions[:, 0]),
        _bounds_from_values(eef_positions[:, 1]),
        _bounds_from_values(eef_positions[:, 2]),
        "observed min/max",
    )


def _yaw_bounds(input_path: Path) -> tuple[float, float] | None:
    config = _find_summary_config(input_path)
    if config and "yaw_low" in config and "yaw_high" in config:
        return float(config["yaw_low"]), float(config["yaw_high"])
    return None


def _uniform_stats_1d(values: np.ndarray, bounds: tuple[float, float], bins: int) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins, range=bounds)
    expected = float(values.size / bins) if bins else 0.0
    chi_square = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else 0.0
    return {
        "n": int(values.size),
        "bounds": [float(bounds[0]), float(bounds[1])],
        "bins": int(bins),
        "expected_per_bin": expected,
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
        "min_count": int(np.min(counts)) if counts.size else 0,
        "max_count": int(np.max(counts)) if counts.size else 0,
        "mean_count": float(np.mean(counts)) if counts.size else 0.0,
        "std_count": float(np.std(counts)) if counts.size else 0.0,
        "count_cv": float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0,
        "chi_square": chi_square,
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
    }


def _uniform_stats_2d(
    xy: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    bins: int,
) -> dict[str, Any]:
    counts, x_edges, y_edges = np.histogram2d(
        xy[:, 0],
        xy[:, 1],
        bins=bins,
        range=[x_bounds, y_bounds],
    )
    expected = float(xy.shape[0] / (bins * bins)) if bins else 0.0
    chi_square = float(np.sum((counts - expected) ** 2 / expected)) if expected > 0 else 0.0
    return {
        "n": int(xy.shape[0]),
        "x_bounds": [float(x_bounds[0]), float(x_bounds[1])],
        "y_bounds": [float(y_bounds[0]), float(y_bounds[1])],
        "bins": int(bins),
        "expected_per_cell": expected,
        "min_count": int(np.min(counts)) if counts.size else 0,
        "max_count": int(np.max(counts)) if counts.size else 0,
        "mean_count": float(np.mean(counts)) if counts.size else 0.0,
        "std_count": float(np.std(counts)) if counts.size else 0.0,
        "count_cv": float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0,
        "chi_square": chi_square,
        "x_edges": x_edges.astype(float).tolist(),
        "y_edges": y_edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
    }


def build_stats(records: list[SampleRecord], input_path: Path, bins: int, eef_source: str) -> dict[str, Any]:
    cube_positions = np.asarray([record.cube_position for record in records], dtype=np.float64)
    eef_positions = np.asarray([record.eef_position for record in records], dtype=np.float64)
    cube_x_bounds, cube_y_bounds = _cube_bounds(records)
    eef_x_bounds, eef_y_bounds, eef_z_bounds, eef_bounds_source = _eef_bounds(records, input_path)

    cube_per_range: dict[str, Any] = {}
    for range_name in sorted({record.cube_range_name for record in records}):
        group = [record for record in records if record.cube_range_name == range_name]
        group_xy = np.asarray([record.cube_position[:2] for record in group], dtype=np.float64)
        x_ranges = [record.cube_range_x for record in group if record.cube_range_x is not None]
        y_ranges = [record.cube_range_y for record in group if record.cube_range_y is not None]
        x_bounds = (min(r[0] for r in x_ranges), max(r[1] for r in x_ranges)) if x_ranges else _bounds_from_values(group_xy[:, 0])
        y_bounds = (min(r[0] for r in y_ranges), max(r[1] for r in y_ranges)) if y_ranges else _bounds_from_values(group_xy[:, 1])
        cube_per_range[range_name] = _uniform_stats_2d(group_xy, x_bounds, y_bounds, bins)

    stats: dict[str, Any] = {
        "input_path": str(input_path),
        "n_samples": len(records),
        "bins": bins,
        "eef_source": eef_source,
        "cube": {
            "xy": _uniform_stats_2d(cube_positions[:, :2], cube_x_bounds, cube_y_bounds, bins),
            "x": _uniform_stats_1d(cube_positions[:, 0], cube_x_bounds, bins),
            "y": _uniform_stats_1d(cube_positions[:, 1], cube_y_bounds, bins),
            "z_observed": {
                "min": float(np.min(cube_positions[:, 2])),
                "max": float(np.max(cube_positions[:, 2])),
                "mean": float(np.mean(cube_positions[:, 2])),
                "std": float(np.std(cube_positions[:, 2])),
            },
            "per_cube_range": cube_per_range,
        },
        "eef": {
            "bounds_source": eef_bounds_source,
            "xy": _uniform_stats_2d(eef_positions[:, :2], eef_x_bounds, eef_y_bounds, bins),
            "x": _uniform_stats_1d(eef_positions[:, 0], eef_x_bounds, bins),
            "y": _uniform_stats_1d(eef_positions[:, 1], eef_y_bounds, bins),
            "z": _uniform_stats_1d(eef_positions[:, 2], eef_z_bounds, bins),
        },
    }

    yaws = np.asarray([record.eef_yaw for record in records if record.eef_yaw is not None], dtype=np.float64)
    yaw_bounds = _yaw_bounds(input_path)
    if yaws.size and yaw_bounds is not None:
        stats["eef"]["yaw"] = _uniform_stats_1d(yaws, yaw_bounds, bins)
    elif yaws.size:
        stats["eef"]["yaw"] = _uniform_stats_1d(yaws, _bounds_from_values(yaws), bins)
        stats["eef"]["yaw"]["bounds_source"] = "observed min/max"
    return stats


def _import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    return plt, Rectangle


def _scatter_subset(xy: np.ndarray, max_points: int) -> np.ndarray:
    if xy.shape[0] <= max_points:
        return xy
    indexes = np.linspace(0, xy.shape[0] - 1, max_points, dtype=np.int64)
    return xy[indexes]


def _draw_bounds(ax: Any, x_bounds: tuple[float, float], y_bounds: tuple[float, float], Rectangle: Any) -> None:
    rect = Rectangle(
        (x_bounds[0], y_bounds[0]),
        x_bounds[1] - x_bounds[0],
        y_bounds[1] - y_bounds[0],
        fill=False,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.add_patch(rect)


def _plot_hist2d(
    ax: Any,
    xy: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    bins: int,
    title: str,
) -> Any:
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[x_bounds, y_bounds])
    image = ax.imshow(
        counts.T,
        origin="lower",
        extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return image


def plot_overview(
    records: list[SampleRecord],
    input_path: Path,
    output_dir: Path,
    bins: int,
    scatter_max: int,
    eef_source: str,
) -> Path:
    plt, Rectangle = _import_matplotlib()
    cube_positions = np.asarray([record.cube_position for record in records], dtype=np.float64)
    eef_positions = np.asarray([record.eef_position for record in records], dtype=np.float64)
    cube_x_bounds, cube_y_bounds = _cube_bounds(records)
    eef_x_bounds, eef_y_bounds, _, eef_bounds_source = _eef_bounds(records, input_path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    cube_xy = _scatter_subset(cube_positions[:, :2], scatter_max)
    axes[0, 0].scatter(cube_xy[:, 0], cube_xy[:, 1], s=5, alpha=0.45, linewidths=0)
    for record in records:
        if record.cube_range_x and record.cube_range_y:
            _draw_bounds(axes[0, 0], record.cube_range_x, record.cube_range_y, Rectangle)
    axes[0, 0].set_title(f"Cube XY scatter (n={len(records)})")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal", adjustable="box")

    image = _plot_hist2d(axes[0, 1], cube_positions[:, :2], cube_x_bounds, cube_y_bounds, bins, "Cube XY density")
    fig.colorbar(image, ax=axes[0, 1], label="count")

    eef_xy = _scatter_subset(eef_positions[:, :2], scatter_max)
    axes[1, 0].scatter(eef_xy[:, 0], eef_xy[:, 1], s=5, alpha=0.45, linewidths=0, color="#d95f02")
    _draw_bounds(axes[1, 0], eef_x_bounds, eef_y_bounds, Rectangle)
    axes[1, 0].set_title(f"EEF XY scatter ({eef_source}, bounds: {eef_bounds_source})")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    axes[1, 0].set_aspect("equal", adjustable="box")

    image = _plot_hist2d(axes[1, 1], eef_positions[:, :2], eef_x_bounds, eef_y_bounds, bins, "EEF XY density")
    fig.colorbar(image, ax=axes[1, 1], label="count")

    output_path = output_dir / "sampling_overview.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_axis_hist(ax: Any, values: np.ndarray, bounds: tuple[float, float], bins: int, title: str) -> None:
    counts, _, _ = ax.hist(values, bins=bins, range=bounds, color="#4c78a8", alpha=0.82)
    expected = values.size / bins
    ax.axhline(expected, color="#f58518", linestyle="--", linewidth=1.3, label=f"expected={expected:.1f}")
    ax.set_title(f"{title}  min={np.min(counts):.0f}, max={np.max(counts):.0f}")
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)


def plot_marginals(records: list[SampleRecord], input_path: Path, output_dir: Path, bins: int) -> Path:
    plt, _ = _import_matplotlib()
    cube_positions = np.asarray([record.cube_position for record in records], dtype=np.float64)
    eef_positions = np.asarray([record.eef_position for record in records], dtype=np.float64)
    cube_x_bounds, cube_y_bounds = _cube_bounds(records)
    eef_x_bounds, eef_y_bounds, eef_z_bounds, _ = _eef_bounds(records, input_path)

    yaws = np.asarray([record.eef_yaw for record in records if record.eef_yaw is not None], dtype=np.float64)
    variables: list[tuple[str, np.ndarray, tuple[float, float]]] = [
        ("cube x", cube_positions[:, 0], cube_x_bounds),
        ("cube y", cube_positions[:, 1], cube_y_bounds),
        ("eef x", eef_positions[:, 0], eef_x_bounds),
        ("eef y", eef_positions[:, 1], eef_y_bounds),
        ("eef z", eef_positions[:, 2], eef_z_bounds),
    ]
    if yaws.size:
        variables.append(("eef yaw", yaws, _yaw_bounds(input_path) or _bounds_from_values(yaws)))

    ncols = 3
    nrows = math.ceil(len(variables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, (title, values, bounds) in zip(axes_array.ravel(), variables):
        _plot_axis_hist(ax, values, bounds, bins, title)
    for ax in axes_array.ravel()[len(variables):]:
        ax.axis("off")

    output_path = output_dir / "sampling_marginals.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_cube_ranges(records: list[SampleRecord], output_dir: Path, bins: int) -> Path | None:
    grouped = {name: [r for r in records if r.cube_range_name == name] for name in sorted({r.cube_range_name for r in records})}
    if len(grouped) <= 1:
        return None

    plt, _ = _import_matplotlib()
    ncols = min(3, len(grouped))
    nrows = math.ceil(len(grouped) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, (name, group) in zip(axes_array.ravel(), grouped.items()):
        xy = np.asarray([record.cube_position[:2] for record in group], dtype=np.float64)
        x_ranges = [record.cube_range_x for record in group if record.cube_range_x is not None]
        y_ranges = [record.cube_range_y for record in group if record.cube_range_y is not None]
        x_bounds = (min(r[0] for r in x_ranges), max(r[1] for r in x_ranges)) if x_ranges else _bounds_from_values(xy[:, 0])
        y_bounds = (min(r[0] for r in y_ranges), max(r[1] for r in y_ranges)) if y_ranges else _bounds_from_values(xy[:, 1])
        image = _plot_hist2d(ax, xy, x_bounds, y_bounds, bins, f"{name}\nn={len(group)}")
        fig.colorbar(image, ax=ax, label="count")
    for ax in axes_array.ravel()[len(grouped):]:
        ax.axis("off")

    output_path = output_dir / "cube_ranges_xy_density.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_stats(stats: dict[str, Any], errors: list[str], output_dir: Path) -> Path:
    stats_with_errors = dict(stats)
    stats_with_errors["skipped_metadata_count"] = len(errors)
    stats_with_errors["skipped_metadata_errors"] = errors[:50]
    output_path = output_dir / "sampling_uniformity_stats.json"
    output_path.write_text(json.dumps(stats_with_errors, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cube and EEF sampling uniformity from data collection metadata.json files.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_DATASET_DIR,
        help="Dataset root, cube range directory, sample directory, or metadata.json path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated PNG/JSON outputs. Defaults to <input>/visualizations for directories.",
    )
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins per axis.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--scatter-max", type=int, default=12000, help="Max points shown in scatter plots.")
    parser.add_argument(
        "--eef-source",
        choices=("sampled", "actual"),
        default="sampled",
        help="Use initial_eef_sample.position or actual state.poses.gripper0_right_eef.position.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if args.scatter_max <= 0:
        raise ValueError("--scatter-max must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        input_path.parent / "visualizations" if input_path.is_file() else input_path / "visualizations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records, errors = load_records(input_path, eef_source=args.eef_source, max_samples=args.max_samples)
    if not records:
        raise RuntimeError(f"No valid metadata.json files found under {input_path}. First errors: {errors[:3]}")

    stats = build_stats(records, input_path, args.bins, args.eef_source)
    outputs = [
        plot_overview(records, input_path, output_dir, args.bins, args.scatter_max, args.eef_source),
        plot_marginals(records, input_path, output_dir, args.bins),
        write_stats(stats, errors, output_dir),
    ]
    per_range_path = plot_cube_ranges(records, output_dir, args.bins)
    if per_range_path is not None:
        outputs.append(per_range_path)

    print(json.dumps({
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_samples": len(records),
        "skipped_metadata_count": len(errors),
        "outputs": [str(path) for path in outputs],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
