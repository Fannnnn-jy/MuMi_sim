from __future__ import annotations

import array
from pathlib import Path

import numpy as np


def save_rgb_exr(image: np.ndarray, path: str | Path) -> Path:
    import Imath
    import OpenEXR

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {rgb.shape}")

    rgb_float = rgb[:, :, :3].astype(np.float32, copy=False)
    if np.issubdtype(rgb.dtype, np.integer):
        rgb_float = rgb_float / np.float32(np.iinfo(rgb.dtype).max)

    height, width = rgb_float.shape[:2]
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pixel_type),
        "G": Imath.Channel(pixel_type),
        "B": Imath.Channel(pixel_type),
    }
    exr = OpenEXR.OutputFile(str(output_path), header)
    try:
        exr.writePixels({
            "R": array.array("f", rgb_float[:, :, 0].ravel()).tobytes(),
            "G": array.array("f", rgb_float[:, :, 1].ravel()).tobytes(),
            "B": array.array("f", rgb_float[:, :, 2].ravel()).tobytes(),
        })
    finally:
        exr.close()
    return output_path


def save_depth_exr(depth: np.ndarray, path: str | Path) -> Path:
    import Imath
    import OpenEXR

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth_float = np.asarray(depth, dtype=np.float32)
    if depth_float.ndim != 2:
        raise ValueError(f"Expected an HxW depth image, got shape {depth_float.shape}")

    height, width = depth_float.shape
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {"Depth": Imath.Channel(pixel_type)}
    exr = OpenEXR.OutputFile(str(output_path), header)
    try:
        exr.writePixels({"Depth": array.array("f", depth_float.ravel()).tobytes()})
    finally:
        exr.close()
    return output_path
