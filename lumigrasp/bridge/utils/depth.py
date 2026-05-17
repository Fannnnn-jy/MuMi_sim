from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import json


def _linearize_mujoco_depth(sim: Any, depth_buffer: np.ndarray) -> np.ndarray:
    """important: mujoco 原生深度渲染的逆过程，原生深度经过 normalized --- /home/ubuntu/miniconda3/envs/mubl_sim/lib/python3.11/site-packages/robosuite/utils/camera_utils.py"""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return (near / (1.0 - depth_buffer * (1.0 - near / far))).astype(np.float32)


def _save_depth_png(depth: np.ndarray, path: Path) -> None:
    finite = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    if finite.any():
        values = depth[finite]
        d_min = float(values.min())
        d_max = float(values.max())
        if d_max > d_min:
            normalized[finite] = (depth[finite] - d_min) / (d_max - d_min)
    image = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def _save_depth_l2_outputs(
    mujoco_depth: np.ndarray,
    blender_depth: np.ndarray,
    error_path: Path,
    error_png_path: Path,
) -> dict[str, float]:
    valid = np.isfinite(mujoco_depth) & np.isfinite(blender_depth)
    diff = np.zeros(mujoco_depth.shape, dtype=np.float32)
    diff[valid] = mujoco_depth[valid] - blender_depth[valid]
    error = np.abs(diff)
    np.save(error_path, error)
    _save_error_png(error, valid, error_png_path)

    values = diff[valid].astype(np.float64)
    if values.size == 0:
        return {"valid_pixels": 0, "l2_distance": float("nan"), "rmse": float("nan"), "mean_abs_error": float("nan")}
    return {
        "valid_pixels": int(values.size),
        "l2_distance": float(np.linalg.norm(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mean_abs_error": float(np.mean(np.abs(values))),
    }


def _save_error_png(error: np.ndarray, valid: np.ndarray, path: Path) -> None:
    """红色像素无效，颜色越亮误差越大"""
    normalized = np.zeros(error.shape, dtype=np.float32)

    if valid.any():
        max_error = float(error[valid].max())
        if max_error > 0.0:
            normalized[valid] = error[valid] / max_error

    gray = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1)

    # invalid pixels: red
    rgb[~valid] = np.array([255, 0, 0], dtype=np.uint8)

    Image.fromarray(rgb, mode="RGB").save(path)
