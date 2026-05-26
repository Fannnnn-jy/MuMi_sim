from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import OpenEXR
import Imath


DEFAULT_IMAGE_KEYS = (
    "blender_rgb_no_fingerlight_raw_exr_path",
    "blender_rgb_blue_only_raw_exr_path",
    "blender_rgb_red_only_raw_exr_path",
)

def read_exr_rgb_gamma(
    exr_path: str | Path,
    gamma: float = 2.2,
    exposure: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Read RGB channels from an EXR file and apply gamma correction.

    Returns:
        rgb_uint8: H x W x 3 uint8 image in [0, 255]
    """
    exr_path = Path(exr_path)
    exr = OpenEXR.InputFile(str(exr_path))

    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    channels = []
    for ch in ("R", "G", "B"):
        if ch not in header["channels"]:
            raise ValueError(f"{exr_path} missing channel {ch}")

        raw = exr.channel(ch, pixel_type)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        channels.append(arr)

    rgb = np.stack(channels, axis=-1)

    # exposure / brightness scaling
    rgb = rgb * exposure

    # linear RGB -> display RGB
    rgb = np.maximum(rgb, 0.0)
    if gamma is not None and gamma > 0:
        rgb = rgb ** (1.0 / gamma)

    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)

    rgb_uint8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    return rgb_uint8


@dataclass(frozen=True)
class ImitationSample:
    metadata_path: Path
    image_paths: tuple[Path, Path, Path]
    qpos6: tuple[float, ...]
    action: tuple[float, ...]


def _resolve_path(path_text: str, sample_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return sample_dir / path


def _get_first_robot(metadata: dict[str, Any]) -> dict[str, Any]:
    robots = metadata.get("state", {}).get("robot_state", {}).get("robots", [])
    if not robots:
        raise KeyError("metadata['state']['robot_state']['robots'] is empty")
    return robots[0]


def _action_from_metadata(metadata: dict[str, Any], quat_key: str) -> list[float]:
    gt_action = metadata["gt_action"]
    delta_pos = gt_action["delta_pos"]
    delta_quat = gt_action[quat_key]
    if len(delta_pos) != 3:
        raise ValueError(f"delta_pos must have 3 values, got {len(delta_pos)}")
    if len(delta_quat) != 4:
        raise ValueError(f"{quat_key} must have 4 values, got {len(delta_quat)}")
    return [float(v) for v in delta_pos] + [float(v) for v in delta_quat]


def clip_vector(values: Sequence[float], max_norm: float | None) -> list[float]:
    if max_norm is None or max_norm <= 0:
        return [float(v) for v in values]
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= max_norm or norm == 0.0:
        return [float(v) for v in values]
    scale = max_norm / norm
    return [float(v) * scale for v in values]


def discover_samples(
    data_root: str | Path,
    *,
    image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
    quat_key: str = "delta_quat_xyzw",
    dpos_clip_norm: float | None = None,
) -> list[ImitationSample]:
    root = Path(data_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")

    samples: list[ImitationSample] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        sample_dir = metadata_path.parent
        render_outputs = metadata.get("render_outputs", {})
        image_paths = tuple(_resolve_path(render_outputs[key], sample_dir) for key in image_keys)
        missing = [str(path) for path in image_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing image file(s) for {metadata_path}: {missing}")

        robot = _get_first_robot(metadata)
        qpos = robot.get("arm_qpos") or robot.get("joint_qpos")
        if qpos is None or len(qpos) < 6:
            raise ValueError(f"{metadata_path} does not contain at least 6 arm joint angles")

        action = _action_from_metadata(metadata, quat_key=quat_key)
        # action = clip_vector(action, action_clip_norm)
        pos = clip_vector(action[:3], dpos_clip_norm)
        quat = [float(v) for v in action[3:7]]
        action = pos + quat

        samples.append(
            ImitationSample(
                metadata_path=metadata_path,
                image_paths=image_paths,  # type: ignore[arg-type]
                qpos6=tuple(float(v) for v in qpos[:6]),
                action=tuple(action),
            )
        )

    if not samples:
        raise FileNotFoundError(f"no metadata.json files found under {root}")
    return samples


def split_samples(
    samples: Sequence[ImitationSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[ImitationSample], list[ImitationSample]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(samples), generator=generator).tolist()
    n_val = int(round(len(samples) * val_fraction))
    val_indexes = set(order[:n_val])
    train = [sample for index, sample in enumerate(samples) if index not in val_indexes]
    val = [sample for index, sample in enumerate(samples) if index in val_indexes]
    if not train:
        raise ValueError("train split is empty!")
    return train, val


class LumiGraspImitationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: Sequence[ImitationSample],
        *,
        image_size: int = 128,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        path = Path(path)

        if path.suffix.lower() == ".exr":
            array = read_exr_rgb_gamma(path, gamma=2.2)
        else:
            image = Image.open(path).convert("RGB")
            array = np.array(image, dtype=np.uint8, copy=True)

        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return (tensor - self.image_mean) / self.image_std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        images = [self._load_image(path) for path in sample.image_paths]
        return {
            "images": torch.cat(images, dim=0),
            "qpos": torch.tensor(sample.qpos6, dtype=torch.float32),
            "action": torch.tensor(sample.action, dtype=torch.float32),
        }
