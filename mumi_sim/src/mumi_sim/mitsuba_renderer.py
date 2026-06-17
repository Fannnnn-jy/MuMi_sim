from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import drjit as dr
import mitsuba as mi
import numpy as np
import json

from .mujoco_state import CameraState, GeomState, Pose
from .panda_assets import MeshSpec as PandaMeshSpec, load_panda_visual_meshes

from .utils.coords import _quat_wxyz_to_mat, _quat_conjugate

mi.set_variant("cuda_ad_rgb") # cuda
# mi.set_variant("scalar_rgb") # cpu



@dataclass(frozen=True)
class MeshSpec:
    mesh_path: Path
    material_name: str
    rgba: tuple[float, float, float, float]
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class RenderObject:
    name: str
    kind: str
    material: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    radius: float = 1.0
    half_height: float = 1.0
    half_size: tuple[float, float] = (1.0, 1.0)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mesh_specs: list[MeshSpec] | None = None


@dataclass
class LightObject:
    name: str
    parent_body_name: str
    local_position: tuple[float, float, float]
    color: tuple[float, float, float]
    light_type: str
    energy: float
    cutoff_angle: float
    beam_width: float
    enabled: bool = False


def _transform_from_pose(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> mi.ScalarTransform4f:
    rot = _quat_wxyz_to_mat(quaternion_wxyz)
    matrix = mi.ScalarMatrix4f(
        [
            [rot[0][0], rot[0][1], rot[0][2], position[0]],
            [rot[1][0], rot[1][1], rot[1][2], position[1]],
            [rot[2][0], rot[2][1], rot[2][2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return mi.ScalarTransform4f(matrix)


def _transform_point(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    local_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    rot = _quat_wxyz_to_mat(quaternion_wxyz)
    return (
        position[0] + rot[0][0] * local_position[0] + rot[0][1] * local_position[1] + rot[0][2] * local_position[2],
        position[1] + rot[1][0] * local_position[0] + rot[1][1] * local_position[1] + rot[1][2] * local_position[2],
        position[2] + rot[2][0] * local_position[0] + rot[2][1] * local_position[1] + rot[2][2] * local_position[2],
    )


def _transform_direction(
    quaternion_wxyz: tuple[float, float, float, float],
    local_direction: tuple[float, float, float],
) -> tuple[float, float, float]:
    rot = _quat_wxyz_to_mat(quaternion_wxyz)
    return (
        rot[0][0] * local_direction[0] + rot[0][1] * local_direction[1] + rot[0][2] * local_direction[2],
        rot[1][0] * local_direction[0] + rot[1][1] * local_direction[1] + rot[1][2] * local_direction[2],
        rot[2][0] * local_direction[0] + rot[2][1] * local_direction[1] + rot[2][2] * local_direction[2],
    )



def _inverse_local_pose(
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    inv_quat = _quat_conjugate(quaternion_wxyz)
    inv_pos = _transform_point((0.0, 0.0, 0.0), inv_quat, (-position[0], -position[1], -position[2]))
    return inv_pos, inv_quat


def _rgba(text: str) -> tuple[float, float, float, float]:
    values = [float(x) for x in text.split()]
    return (values[0], values[1], values[2], values[3])


class MitsubaRenderer:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        samples: int = 64,
        use_gpu: bool = True,
        robot_asset_dir: str | Path | None = "assets/robot_asset/franka_emika_panda",
        robosuite_asset_dir: str | Path = "assets/robosuite_assets",
        # gripper_light_type: str = "SPOT",
        # gripper_light_energy: float = 2.0,
        # gripper_light_size: float = 0.015,
        # gripper_spot_size: float = math.radians(90.0),
        # gripper_spot_blend: float = 0.35,
        # show_gripper_light_marker: bool = False,
        gripper_light_config_path: str | Path = "configs/gripper_light.json",

        preview_exposure: float = 0.5,
        preview_tonemap: str = "auto_exposure",
        auto_exposure_percentile: float = None,
    ) -> None:
        self.width = width
        self.height = height
        self.samples = samples
        self.use_gpu = use_gpu
        self.robot_asset_dir = Path(robot_asset_dir) if robot_asset_dir else None
        self.robosuite_asset_dir = Path(robosuite_asset_dir)

        self.load_gripper_light_config(gripper_light_config_path)

        # self.gripper_light_type = gripper_light_type
        # self.gripper_light_energy = gripper_light_energy
        # self.gripper_light_size = gripper_light_size
        # self.gripper_spot_size = gripper_spot_size
        # self.gripper_spot_blend = gripper_spot_blend
        # self.show_gripper_light_marker = show_gripper_light_marker

        self.preview_exposure = preview_exposure
        self.preview_tonemap = preview_tonemap
        self.auto_exposure_percentile = auto_exposure_percentile

        self.materials: dict[str, tuple[float, float, float, float]] = {}
        self.textures: dict[str, tuple[Path, tuple[float, float]]] = {}
        self.robot_mesh_specs = self._load_robot_mesh_specs()
        self.objects: dict[str, RenderObject] = {}
        self.arena_objects: dict[str, RenderObject] = {}
        self.gripper_finger_lights: dict[str, LightObject] = {}
        self.gripper_finger_light_states: dict[str, bool] = {"left": False, "right": False}
        self._gripper_lights_added = False
        self.camera = CameraState(
            name="default",
            position=(1.8, -2.4, 2.6),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            fovy_degrees=45.0,
        )
    
    def load_gripper_light_config(self, config_path: str | Path | None) -> None:
        config = {}
        assert config_path is not None, "Gripper light config path is required"
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            config.update(loaded)

        self.gripper_light_type = str(config["light_type"])
        self.gripper_light_energy = float(config["energy"])
        # self.gripper_light_size = float(config["size"])
        # self.gripper_spot_size = float(config["spot_size"])
        # self.gripper_spot_blend = float(config["spot_blend"])
        self.cutoff_angle = float(config.get("cutoff_angle", 90.0))
        self.beam_width = float(config.get("beam_width", 90.0))
        self.show_gripper_light_marker = bool(config["show_marker"])

    def setup_scene(self) -> None:
        self._make_material("table_mat", (0.38, 0.34, 0.28, 1.0))
        self._make_material("cube_mat", (0.8, 0.16, 0.1, 1.0))
        self._make_material("robot_mat", (0.62, 0.66, 0.7, 1.0))
        self._make_material("gripper_mat", (0.12, 0.12, 0.12, 1.0))
        self._make_material("mount_dark", (0.2, 0.2, 0.2, 1.0))
        self._make_material("mount_collision", (0.5, 0.5, 0.0, 1.0))
        self._make_material(
            "arena_floorplane",
            (0.78, 0.76, 0.72, 1.0),
            texture_path=self.robosuite_asset_dir / "textures" / "light-gray-floor-tile.png",
            texture_repeat=(6.0, 6.0),
        )
        self._make_material(
            "arena_walls_mat",
            (0.68, 0.65, 0.60, 1.0),
            texture_path=self.robosuite_asset_dir / "textures" / "light-gray-plaster.png",
            texture_repeat=(2.0, 2.0),
        )
        self._make_material(
            "arena_table_ceramic",
            (0.86, 0.84, 0.78, 1.0),
            texture_path=self.robosuite_asset_dir / "textures" / "ceramic.png",
            texture_repeat=(0.8, 0.8),
        )
        self._make_material(
            "arena_table_legs_metal",
            (0.45, 0.45, 0.43, 1.0),
            texture_path=self.robosuite_asset_dir / "textures" / "steel-brushed.png",
            texture_repeat=(1.0, 1.0),
        )

    def add_gripper_light(
        self,
        parent_body_name: str = "gripper0_right_eef",
        local_position: tuple[float, float, float] = (0.0, 0.0, 0.4),
        energy: float = 1.0,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        light_type: str = "SPOT",
        cutoff_angle: float = 90.0,
        beam_width: float = 90.0,
        # show_marker: bool = False,
        # marker_length: float = 0.04,
        # marker_radius: float = 0.004,
    ) -> LightObject | None:
        if parent_body_name not in self.objects:
            return None
        light = LightObject(
            name=f"{parent_body_name}_{light_type.lower()}_light",
            parent_body_name=parent_body_name,
            local_position=local_position,
            color=color,
            energy=energy,
            light_type=light_type,
            cutoff_angle=cutoff_angle,
            beam_width=beam_width,
        )
        return light

    def add_gripper_finger_lights(
        self,
        left_body_name: str = "gripper0_right_leftfinger",
        right_body_name: str = "gripper0_right_rightfinger",
        local_position: tuple[float, float, float] = (0.0, 0.0, 0.055),
        energy: float = None,
        light_type: str = None,
        cutoff_angle: float = None,
        beam_width: float = None,
        # show_marker: bool = False,
    ) -> None:
        left_light = self.add_gripper_light(
            parent_body_name=left_body_name,
            local_position=local_position,
            energy=energy,
            color=(1.0, 0.0, 0.0),
            light_type=light_type,
            cutoff_angle=cutoff_angle,
            beam_width=beam_width,
            # show_marker=show_marker,
        )
        right_light = self.add_gripper_light(
            parent_body_name=right_body_name,
            local_position=local_position,
            energy=energy,
            color=(0.0, 0.0, 1.0),
            light_type=light_type,
            cutoff_angle=cutoff_angle,
            beam_width=beam_width,
            # show_marker=show_marker,
        )
        if left_light is not None:
            self.gripper_finger_lights["left"] = left_light
        if right_light is not None:
            self.gripper_finger_lights["right"] = right_light

    def set_gripper_finger_lights(self, params, finger: str, enabled: bool) -> None:
        if finger not in {"left", "right"}:
            raise ValueError(f"Unsupported gripper finger light: {finger!r}")
        light = self.gripper_finger_lights.get(finger)
        assert light is not None
        light.enabled = enabled
        self.gripper_finger_light_states[finger] = enabled
        if params:
            key = f"{light.name}.intensity.value"
            if key not in params:
                raise KeyError(f"Light intensity key not found in Mitsuba params: {key!r}")
            if enabled:
                params[key] = [
                    light.energy * light.color[0],
                    light.energy * light.color[1],
                    light.energy * light.color[2],
                ]
            else:
                params[key] = [0.0, 0.0, 0.0]

            params.update()

    def ensure_default_objects(self, poses: Mapping[str, Pose]) -> None:
        # poses.pop("robot0_right_hand", None) ### debug
        for name in poses:
            if name in self.objects:
                continue
            if name == "table" or name == "cube_main":
                continue
            if name in self.robot_mesh_specs:
                self.objects[name] = RenderObject(
                    name=name,
                    kind="mesh_body",
                    material="robot_mat",
                    mesh_specs=self.robot_mesh_specs[name],
                )
            elif name.startswith("robot0_") or name.startswith("gripper0_"):
                if not self.robot_mesh_specs:
                    self.objects[name] = RenderObject(
                        name=name,
                        kind="cube",
                        material="robot_mat",
                        scale=(0.06, 0.06, 0.06),
                    )

    def update_poses(self, poses: Mapping[str, Pose]) -> None:
        self.ensure_default_objects(poses)
        for name, pose in poses.items():
            obj = self.objects.get(name)
            if obj is not None:
                obj.position = pose.position
                obj.quaternion_wxyz = pose.quaternion_wxyz
        if not self._gripper_lights_added and {
            "gripper0_right_leftfinger",
            "gripper0_right_rightfinger",
        }.issubset(self.objects):
            self.add_gripper_finger_lights(
                energy=self.gripper_light_energy,
                light_type=self.gripper_light_type,
                cutoff_angle=self.cutoff_angle,
                beam_width = self.beam_width,
                # show_marker=self.show_gripper_light_marker,
            )
            self._gripper_lights_added = True

    def sync_arena_geoms(self, geoms: Mapping[str, GeomState]) -> None:
        for name, geom in geoms.items():
            if name not in self.arena_objects:
                self.arena_objects[name] = self._add_arena_geom(geom)
            obj = self.arena_objects[name]
            obj.position = geom.position
            obj.quaternion_wxyz = geom.quaternion_wxyz

    def sync_cube_geom(self, geom: GeomState) -> None:
        scale = (2.0 * geom.size[0], 2.0 * geom.size[1], 2.0 * geom.size[2])
        obj = self.objects.get("cube_main")
        if obj is None:
            obj = RenderObject(name="cube_main", kind="cube", material="cube_mat", scale=scale)
            self.objects["cube_main"] = obj
        else:
            obj.scale = scale
        obj.position = geom.position
        obj.quaternion_wxyz = geom.quaternion_wxyz

    def set_camera_from_mujoco(self, camera: CameraState) -> None:
        self.camera = camera

    def render(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene_dict = self._scene_dict("path")
        t0 = time.perf_counter()
        scene = mi.load_dict(scene_dict)
        t1 = time.perf_counter()
        image = mi.render(scene)
        dr.eval(image)          
        dr.sync_thread() # force synchronization: Dr.Jit lazy evaluation
        t2 = time.perf_counter()
        if path.suffix.lower() == ".exr":
            mi.Bitmap(image).write(str(path))
            preview_path = path.with_suffix(".png")
            preview = self._preview_bitmap(image)
            preview.write(str(preview_path))
        else:
            preview = self._preview_bitmap(image)
            preview.write(str(path))
        t3 = time.perf_counter()
        print("----------Render Timing----------")
        print(f"load_dict:         {(t1 - t0) * 1000:.1f}ms")
        print(f"render:            {(t2 - t1) * 1000:.1f}ms")
        print(f"write outputs:     {(t3 - t2) * 1000:.1f}ms")
        print(f"TOTAL:             {(t3 - t0) * 1000:.1f}ms")
        return path
    
    def render_with_scene(self, scene, params, output_path, render_depth=False):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = mi.render(scene, params=params)
        dr.eval(image)
        dr.sync_thread()

        mi.Bitmap(image).write(str(path))
        preview_path = path.with_suffix(".png")
        preview = self._preview_bitmap(image)
        preview.write(str(preview_path))

        return Path(output_path)

    # ========== 持久场景（修复 load_dict 内存泄漏，支持多 integrator）==========
    @staticmethod
    def _np_mat(transform) -> np.ndarray:
        """ScalarTransform4f -> 4x4 float64 numpy（约定: T @ [x,y,z,1]）。"""
        return np.asarray(transform.matrix).reshape(4, 4).astype(np.float64)

    def _build_vertex_cache(self, params) -> None:
        """从给定 params 反推并缓存每个动态物体的 body 坐标系顶点（只需建一次，各 integrator 共用）。"""
        self._dynamic_meshes = []
        for name, obj in self.objects.items():
            T_body_init = self._np_mat(
                _transform_from_pose(obj.position, obj.quaternion_wxyz)
            )
            T_body_inv = np.linalg.inv(T_body_init)
            keys = []
            if obj.kind == "mesh_body":
                for idx, spec in enumerate(obj.mesh_specs or []):
                    keys.append(f"{obj.name}_{idx}_{spec.mesh_path.stem}.vertex_positions")
            elif obj.kind == "cube":
                keys.append(f"{obj.name}.vertex_positions")
            for key in keys:
                if key not in params:
                    continue
                wv = np.asarray(params[key]).reshape(-1, 3).astype(np.float64)
                homo = np.concatenate([wv, np.ones((wv.shape[0], 1))], axis=1).T
                body_homo = T_body_inv @ homo
                self._dynamic_meshes.append((key, name, body_homo))

    def build_persistent_scene(self, integrator: str = "path") -> None:
        """用 optimize=False load 一个 integrator 的场景。顶点缓存首次建立、各 integrator 共用。"""
        if not hasattr(self, "_persistent_scenes"):
            self._persistent_scenes = {}
            self._persistent_params = {}
        scene_dict = self._scene_dict(integrator)
        scene = mi.load_dict(scene_dict, optimize=False)
        params = mi.traverse(scene)
        self._persistent_scenes[integrator] = scene
        self._persistent_params[integrator] = params
        # 顶点缓存只建一次（几何在所有 integrator 间一致）
        if not getattr(self, "_dynamic_meshes", None):
            self._build_vertex_cache(params)
        self._persistent_lights = list(self.gripper_finger_lights.values())

    def update_persistent_scene(self, integrator: str = "path"):
        """每步调用：首次自动 build 该 integrator；之后只更新顶点 + 灯光/相机。
        返回 (scene, params) 方便调用方直接使用。"""
        if not hasattr(self, "_persistent_scenes") or integrator not in self._persistent_scenes:
            self.build_persistent_scene(integrator)
            return self._persistent_scenes[integrator], self._persistent_params[integrator]

        params = self._persistent_params[integrator]
        # 1) 动态 mesh/cube 顶点（用共用缓存）
        for key, body_name, body_homo in self._dynamic_meshes:
            if key not in params:
                continue
            obj = self.objects[body_name]
            T_body = self._np_mat(
                _transform_from_pose(obj.position, obj.quaternion_wxyz)
            )
            world = (T_body @ body_homo)[:3]
            flat = np.ascontiguousarray(world.T.reshape(-1), dtype=np.float32)
            params[key] = mi.Float(flat)
        # 2) spot 灯 to_world
        for light in self._persistent_lights:
            lkey = f"{light.name}.to_world"
            if lkey in params:
                parent = self.objects[light.parent_body_name]
                params[lkey] = (
                    _transform_from_pose(parent.position, parent.quaternion_wxyz)
                    @ mi.ScalarTransform4f.translate(light.local_position)
                )
        # 3) 相机
        if "sensor.to_world" in params:
            params["sensor.to_world"] = (
                _transform_from_pose(self.camera.position, self.camera.quaternion_wxyz)
                @ mi.ScalarTransform4f.rotate([0.0, 1.0, 0.0], 180.0)
            )
        params.update()
        return self._persistent_scenes[integrator], self._persistent_params[integrator]
    # ========== 持久场景结束 ==========

    def render_depth(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene_dict = self._scene_dict("depth")
        t0 = time.perf_counter()
        scene = mi.load_dict(scene_dict)
        t1 = time.perf_counter()
        image = mi.render(scene)
        dr.eval(image)
        dr.sync_thread()
        t2 = time.perf_counter()
        mi.Bitmap(image).write(str(path))
        t3 = time.perf_counter()
        print("----------Depth Render Timing----------")
        print(f"load_dict:         {(t1 - t0) * 1000:.1f}ms")
        print(f"render:            {(t2 - t1) * 1000:.1f}ms")
        print(f"write:             {(t3 - t2) * 1000:.1f}ms")
        print(f"TOTAL:             {(t3 - t0) * 1000:.1f}ms")
        return path

    def _scene_dict(self, integrator: str) -> dict[str, object]:
        scene: dict[str, object] = {
            "type": "scene",
            "integrator": {"type": integrator, "hide_emitters": True,},
            "sensor": self._sensor_dict(pixel_format="luminance" if integrator == "depth" else "rgb"),
            "key_area_light": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.translate([0.0, -3.5, 5.0])
                @ mi.ScalarTransform4f.rotate([1.0, 0.0, 0.0], 180.0)
                @ mi.ScalarTransform4f.scale([2.5, 2.5, 1.0]),
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": [8.0, 8.0, 8.0]},
                },
            },
        }
        self._add_materials_to_scene(scene)
        for obj in self.arena_objects.values():
            self._add_object_to_scene(scene, obj)
        for obj in self.objects.values():
            self._add_object_to_scene(scene, obj)
        for finger, light in self.gripper_finger_lights.items():
            self._add_light_to_scene(scene, light)
        return scene
    
    def _mi_load_dict(self, scene_dict: dict[str, object]) -> mi.Scene:
        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)
        return scene, params

    def _sensor_dict(self, pixel_format: str) -> dict[str, object]:
        return {
            "type": "perspective",
            "fov_axis": "y",
            "fov": self.camera.fovy_degrees,
            "near_clip": 0.001,
            "far_clip": 100.0,
            "to_world": _transform_from_pose(self.camera.position, self.camera.quaternion_wxyz)
            @ mi.ScalarTransform4f.rotate([0.0, 1.0, 0.0], 180.0),
            "sampler": {"type": "independent", "sample_count": self.samples},
            "film": {
                "type": "hdrfilm",
                "width": self.width,
                "height": self.height,
                "pixel_format": pixel_format,
                "component_format": "float32",
                "rfilter": {"type": "gaussian"},
            },
        }

    def _add_materials_to_scene(self, scene: dict[str, object]) -> None:
        for name, color in self.materials.items():
            reflectance: dict[str, object] = {"type": "rgb", "value": [color[0], color[1], color[2]]}
            if name in self.textures:
                texture_path, texture_repeat = self.textures[name]
                reflectance = {
                    "type": "bitmap",
                    "filename": str(texture_path),
                    "raw": False,
                    "wrap_mode": "repeat",
                    "to_uv": mi.ScalarTransform4f.scale([texture_repeat[0], texture_repeat[1], 1.0]),
                }
            scene[name] = {"type": "diffuse", "reflectance": reflectance}

    def _add_object_to_scene(self, scene: dict[str, object], obj: RenderObject) -> None:
        if obj.kind == "mesh_body":
            for idx, spec in enumerate(obj.mesh_specs or []):
                scene[f"{obj.name}_{idx}_{spec.mesh_path.stem}"] = {
                    "type": "obj",
                    "filename": str(spec.mesh_path),
                    "to_world": _transform_from_pose(obj.position, obj.quaternion_wxyz)
                    @ mi.ScalarTransform4f.translate(spec.local_position)
                    @ _transform_from_pose((0.0, 0.0, 0.0), spec.local_quaternion_wxyz),
                    "bsdf": {"type": "ref", "id": spec.material_name},
                }
        elif obj.kind == "cube":
            scene[obj.name] = {
                "type": "cube",
                "to_world": _transform_from_pose(obj.position, obj.quaternion_wxyz)
                @ mi.ScalarTransform4f.scale([0.5 * obj.scale[0], 0.5 * obj.scale[1], 0.5 * obj.scale[2]]),
                "bsdf": {"type": "ref", "id": obj.material},
            }
        elif obj.kind == "plane":
            scene[obj.name] = {
                "type": "rectangle",
                "to_world": _transform_from_pose(obj.position, obj.quaternion_wxyz)
                @ mi.ScalarTransform4f.scale([obj.half_size[0], obj.half_size[1], 1.0]),
                "bsdf": {"type": "ref", "id": obj.material},
            }
        elif obj.kind == "cylinder":
            scene[obj.name] = {
                "type": "cylinder",
                "to_world": _transform_from_pose(obj.position, obj.quaternion_wxyz)
                @ mi.ScalarTransform4f.translate([0.0, 0.0, -obj.half_height])
                @ mi.ScalarTransform4f.scale([obj.radius, obj.radius, 2.0 * obj.half_height]),
                "bsdf": {"type": "ref", "id": obj.material},
            }
        elif obj.kind == "empty":
            return

    def _add_light_to_scene(self, scene: dict[str, object], light: LightObject) -> None:
        parent = self.objects[light.parent_body_name]
        position = _transform_point(parent.position, parent.quaternion_wxyz, light.local_position)
        intensity = {
            "type": "rgb",
            "value": [
                light.energy * light.color[0],
                light.energy * light.color[1],
                light.energy * light.color[2],
            ],
        }
        if light.light_type.upper() == "SPOT":
            scene[light.name] = {
                "type": "spot",
                # Blender's validated setup parents the light to the finger and
                # rotates the child light by pi around X, which makes Blender's
                # local -Z emission axis align with the parent's +Z axis.
                # Mitsuba spot emitters already emit along local +Z, so matching
                # Blender means inheriting the parent frame directly.
                "to_world": _transform_from_pose(parent.position, parent.quaternion_wxyz)
                @ mi.ScalarTransform4f.translate(light.local_position),
                "intensity": intensity,
                "cutoff_angle": light.cutoff_angle,
                "beam_width": light.beam_width,
            }
            return
        scene[light.name] = {
            "type": "point",
            "position": position,
            "intensity": intensity,
        }
        # scene[light.name] = {
        #     "type": "sphere",
        #     "to_world": (
        #         mi.ScalarTransform4f.translate(position)
        #         @ mi.ScalarTransform4f.scale(0.005)
        #     ),
        #     "emitter": {
        #         "type": "area",
        #         "radiance": {
        #             "type": "rgb",
        #             "value": [10000.0, 0.0, 0.0],
        #         }
        #     },
        # }

    def _make_material(
        self,
        name: str,
        color: tuple[float, float, float, float],
        texture_path: Path | None = None,
        texture_repeat: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.materials[name] = color
        if texture_path is not None:
            self.textures[name] = (texture_path, texture_repeat)

    # def _validate_gripper_finger_light_states(self) -> None:
    #     return

    # def _write_raw_exr(self, image: mi.TensorXf, path: Path) -> None:
    #     mi.Bitmap(image).write(str(path))

    # def _write_preview_png(self, image: mi.TensorXf, path: Path) -> None:
    #     preview = self._preview_bitmap(image)
    #     preview.write(str(path))

    def _preview_bitmap(self, image: mi.TensorXf) -> mi.Bitmap:
        image_np = np.asarray(image, dtype=np.float32)
        if self.preview_tonemap == "auto_exposure":
            luminance = (
                0.2126 * image_np[:, :, 0]
                + 0.7152 * image_np[:, :, 1]
                + 0.0722 * image_np[:, :, 2]
            )
            percentile = np.percentile(luminance, self.auto_exposure_percentile * 100.0)
            if percentile > 1e-8:
                self.preview_exposure = 0.5 / percentile
            mapped = np.maximum(image_np * self.preview_exposure, 0.0)
            mapped = np.clip(mapped, 0.0, 1.0)  
        else:
            mapped = np.maximum(image_np * self.preview_exposure, 0.0)
            if self.preview_tonemap == "clip":
                mapped = np.clip(mapped, 0.0, 1.0)
            elif self.preview_tonemap == "reinhard":
                mapped = mapped / (1.0 + mapped)
            else:
                raise ValueError(f"Unsupported preview tonemap: {self.preview_tonemap!r}")
        return mi.Bitmap(mapped).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )

    def _add_arena_geom(self, geom: GeomState) -> RenderObject:
        material = self._arena_material_name(geom)
        if geom.name == "fixed_mount0_torso_vis":
            return RenderObject(name=geom.name, kind="empty", material=material)
        if geom.name == "fixed_mount0_pedestal_vis":
            return self._add_mount_pedestal_mesh(geom, material)
        if geom.geom_type == 0:
            return RenderObject(name=geom.name, kind="plane", material=material, half_size=geom.size[:2])
        if geom.geom_type == 5:
            return RenderObject(
                name=geom.name,
                kind="cylinder",
                material=material,
                radius=geom.size[0],
                half_height=geom.size[1],
            )
        if geom.geom_type == 6:
            return RenderObject(
                name=geom.name,
                kind="cube",
                material=material,
                scale=(2.0 * geom.size[0], 2.0 * geom.size[1], 2.0 * geom.size[2]),
            )
        return RenderObject(name=geom.name, kind="cube", material=material, scale=(0.02, 0.02, 0.02))

    def _add_mount_pedestal_mesh(self, geom: GeomState, material: str) -> RenderObject:
        mesh_path = self._mount_pedestal_obj_path()
        local_position = (0.0, 0.0, 0.0)
        local_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
        if geom.mesh_position is not None and geom.mesh_quaternion_wxyz is not None:
            local_position, local_quaternion_wxyz = _inverse_local_pose(
                geom.mesh_position,
                geom.mesh_quaternion_wxyz,
            )
        return RenderObject(
            name=geom.name,
            kind="mesh_body",
            material=material,
            mesh_specs=[
                MeshSpec(
                    mesh_path=mesh_path,
                    material_name=material,
                    rgba=self.materials[material],
                    local_position=local_position,
                    local_quaternion_wxyz=local_quaternion_wxyz,
                )
            ],
        )

    def _mount_pedestal_obj_path(self) -> Path:
        return self.robosuite_asset_dir / "bases" / "meshes" / "rethink_mount" / "pedestal.obj"

    def _arena_material_name(self, geom: GeomState) -> str:
        if geom.name == "floor":
            return "arena_floorplane"
        if geom.name.startswith("wall_"):
            return "arena_walls_mat"
        if geom.name == "table_visual":
            return "arena_table_ceramic"
        if geom.name.startswith("table_leg"):
            return "arena_table_legs_metal"
        mapping = {
            "floorplane": "arena_floorplane",
            "walls_mat": "arena_walls_mat",
            "table_ceramic": "arena_table_ceramic",
            "table_legs_metal": "arena_table_legs_metal",
            "table_mat": "arena_table_ceramic",
        }
        if geom.name.startswith("fixed_mount0_"):
            if geom.name.endswith("_col"):
                return "mount_collision"
            return "mount_dark"
        if geom.material_name in mapping:
            return mapping[geom.material_name]
        return "table_mat"

    def _load_robot_mesh_specs(self) -> dict[str, list[MeshSpec]]:
        if self.robot_asset_dir is None:
            return {}
        specs_by_body = load_panda_visual_meshes(self.robot_asset_dir)
        result: dict[str, list[MeshSpec]] = {}
        for body_name, specs in specs_by_body.items():
            result[body_name] = []
            for spec in specs:
                material_name = f"panda_{spec.material_name}" if spec.material_name else "robot_mat"
                rgba = spec.rgba or (0.62, 0.66, 0.7, 1.0)
                self.materials[material_name] = rgba
                result[body_name].append(
                    MeshSpec(
                        mesh_path=spec.mesh_path,
                        material_name=material_name,
                        rgba=rgba,
                        local_position=spec.local_position,
                        local_quaternion_wxyz=spec.local_quaternion_wxyz,
                    )
                )
        return result
