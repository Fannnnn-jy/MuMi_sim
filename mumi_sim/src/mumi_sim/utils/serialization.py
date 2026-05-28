from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..mujoco_state import CameraState, GeomState, Pose


def _serialize_poses(poses: dict[str, Pose]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "position": list(pose.position),
            "quaternion_wxyz": list(pose.quaternion_wxyz),
        }
        for name, pose in poses.items()
    }


def _serialize_camera(camera: CameraState) -> dict[str, Any]:
    return {
        "name": camera.name,
        "position": list(camera.position),
        "quaternion_wxyz": list(camera.quaternion_wxyz),
        "fovy_degrees": camera.fovy_degrees,
    }


def _serialize_geoms(geoms: dict[str, GeomState]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "geom_type": geom.geom_type,
            "position": list(geom.position),
            "quaternion_wxyz": list(geom.quaternion_wxyz),
            "size": list(geom.size),
            "rgba": list(geom.rgba),
            "material_name": geom.material_name,
            "mesh_position": list(geom.mesh_position) if geom.mesh_position is not None else None,
            "mesh_quaternion_wxyz": (
                list(geom.mesh_quaternion_wxyz) if geom.mesh_quaternion_wxyz is not None else None
            ),
        }
        for name, geom in geoms.items()
    }
