from __future__ import annotations

from typing import Any

import numpy as np


def _linearize_mujoco_depth(sim: Any, depth_buffer: np.ndarray) -> np.ndarray:
    """Convert MuJoCo normalized depth buffer values to metric depth."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return (near / (1.0 - depth_buffer * (1.0 - near / far))).astype(np.float32)
