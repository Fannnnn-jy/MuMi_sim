from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .mujoco_state import CameraState, GeomState, Pose
from .panda_assets import MeshSpec, load_panda_visual_meshes

def _import_bpy():
    import bpy  # type: ignore
    bpy.context.scene.render.threads_mode = "FIXED"
    bpy.context.scene.render.threads = 48

    return bpy


def _quat_wxyz_to_blender(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return quat


def _mujoco_camera_quat_to_blender(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """mujoco 与 blender 的相机 quat 定义一致，本函数用于占位"""
    return quat


class PrimitiveBlenderRenderer:
    """Small Blender-side renderer that consumes MuJoCo body poses.

    Robot bodies use Panda visual meshes when the project asset directory is
    available. Table and cube remain primitives for the current bridge target.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        samples: int = 64,
        use_gpu: bool = True,
        robot_asset_dir: str | Path | None = "assets/robot_asset/franka_emika_panda",
        robosuite_asset_dir: str | Path = "assets/robosuite_assets",
        gripper_light_type: str = "SPOT",
        gripper_light_energy: float = 35.0,
        gripper_light_size: float = 0.015,
        gripper_spot_size: float = 0.55,
        gripper_spot_blend: float = 0.35,
        show_gripper_light_marker: bool = False,
    ) -> None:
        self.bpy = _import_bpy()
        self.width = width
        self.height = height
        self.samples = samples
        self.use_gpu = use_gpu
        self.robot_asset_dir = Path(robot_asset_dir) if robot_asset_dir else None
        self.robosuite_asset_dir = Path(robosuite_asset_dir)
        self.gripper_light_type = gripper_light_type
        self.gripper_light_energy = gripper_light_energy
        self.gripper_light_size = gripper_light_size
        self.gripper_spot_size = gripper_spot_size
        self.gripper_spot_blend = gripper_spot_blend
        self.show_gripper_light_marker = show_gripper_light_marker
        self.robot_mesh_specs = self._load_robot_mesh_specs()
        self.objects: dict[str, object] = {}
        self.arena_objects: dict[str, object] = {}
        self.gripper_finger_lights: dict[str, object] = {}
        self.gripper_light_markers: dict[str, object] = {}
        self.gripper_finger_light_states: dict[str, bool | None] = {
            "left": None,
            "right": None,
        }
        self._gripper_lights_added = False
        self._imported_mesh_count = 0

    def setup_scene(self) -> None:
        bpy = self.bpy
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.samples = self.samples
        scene.render.resolution_x = self.width
        scene.render.resolution_y = self.height
        scene.render.resolution_percentage = 100
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = -0.45
        scene.view_settings.gamma = 1.0
        if self.use_gpu:
            self._configure_cycles_gpu()

        bpy.ops.object.light_add(type="AREA", location=(0.0, -3.5, 5.0))
        key = bpy.context.object
        key.name = "key_area_light"
        key.data.energy = 200.0
        key.data.size = 5.0

        bpy.ops.object.camera_add(location=(1.8, -2.4, 2.6), rotation=(1.1, 0.0, 0.65))
        scene.camera = bpy.context.object

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
        energy: float = 80.0,
        size: float = 0.03,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        light_type: str = "SPOT",
        spot_size: float = 0.55,
        spot_blend: float = 0.35,
        show_marker: bool = False,
        marker_length: float = 0.04,
        marker_radius: float = 0.004,
    ) -> object | None:
        parent = self.objects.get(parent_body_name)
        if parent is None:
            return None

        bpy = self.bpy
        light_kind = light_type.upper()
        light_name = f"{parent_body_name}_{light_kind.lower()}_light"
        existing_light = self.bpy.data.objects.get(light_name)
        if existing_light is not None:
            return existing_light

        bpy.ops.object.light_add(type=light_kind, location=(0.0, 0.0, 0.0))
        light = bpy.context.object
        light.name = light_name
        light.data.energy = energy
        light.data.color = color
        if light_kind == "SPOT":
            light.data.shadow_soft_size = size
            light.data.spot_size = spot_size
            light.data.spot_blend = spot_blend
        elif light_kind == "POINT":
            light.data.shadow_soft_size = size

        light_rotation = (math.pi, 0.0, 0.0)
        light.parent = parent
        light.matrix_parent_inverse.identity()
        light.location = local_position
        light.rotation_euler = light_rotation
        self._set_light_object_enabled(light, False)

        if show_marker:
            marker = self._add_light_marker(
                parent=parent,
                name=f"{light.name}_marker",
                local_position=local_position,
                local_rotation=light_rotation,
                color=color,
                length=marker_length,
                radius=marker_radius,
            )
            self.gripper_light_markers[parent_body_name] = marker
            self._set_object_visible(marker, False)
        return light

    def add_gripper_finger_lights(
        self,
        left_body_name: str = "gripper0_right_leftfinger",
        right_body_name: str = "gripper0_right_rightfinger",
        local_position: tuple[float, float, float] = (0.0, 0.0, 0.055),
        energy: float = 35.0,
        size: float = 0.015,
        light_type: str = "SPOT",
        spot_size: float = 0.55,
        spot_blend: float = 0.35,
        show_marker: bool = False,
    ) -> None:
        left_light = self.add_gripper_light(
            parent_body_name=left_body_name,
            local_position=local_position,
            energy=energy,
            size=size,
            color=(1.0, 0.0, 0.0),
            light_type=light_type,
            spot_size=spot_size,
            spot_blend=spot_blend,
            show_marker=show_marker,
        )
        right_light = self.add_gripper_light(
            parent_body_name=right_body_name,
            local_position=local_position,
            energy=energy,
            size=size,
            color=(0.0, 0.0, 1.0),
            light_type=light_type,
            spot_size=spot_size,
            spot_blend=spot_blend,
            show_marker=show_marker,
        )
        if left_light is not None:
            self.gripper_finger_lights["left"] = left_light
            left_marker = self.gripper_light_markers.get(left_body_name)
            if left_marker is not None:
                self.gripper_light_markers["left"] = left_marker
        if right_light is not None:
            self.gripper_finger_lights["right"] = right_light
            right_marker = self.gripper_light_markers.get(right_body_name)
            if right_marker is not None:
                self.gripper_light_markers["right"] = right_marker

    def set_gripper_finger_lights(self, finger: str, enabled: bool) -> None:
        """暴露 renderer 灯光控制接口"""
        if finger not in {"left", "right"}:
            raise ValueError(f"Unsupported gripper finger light: {finger!r}")
        self.gripper_finger_light_states[finger] = enabled
        light = self.gripper_finger_lights.get(finger)
        if light is not None:
            self._set_light_object_enabled(light, enabled)
        marker = self.gripper_light_markers.get(finger)
        if marker is not None:
            self._set_object_visible(marker, enabled)

    def ensure_default_objects(self, poses: Mapping[str, Pose]) -> None:
        for name in poses:
            if name in self.objects:
                continue
            if name == "table" or name == "cube_main":
                continue
            if name in self.robot_mesh_specs:
                self.objects[name] = self._add_mesh_body(name, self.robot_mesh_specs[name])
            elif name.startswith("robot0_") or name.startswith("gripper0_"):
                if not self.robot_mesh_specs:
                    self.objects[name] = self._add_cube(name, scale=(0.06, 0.06, 0.06), material="robot_mat")

    def update_poses(self, poses: Mapping[str, Pose]) -> None:
        self.ensure_default_objects(poses)
        for name, pose in poses.items():
            obj = self.objects.get(name)
            if obj is None:
                continue
            obj.location = pose.position
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = _quat_wxyz_to_blender(pose.quaternion_wxyz)
        if not self._gripper_lights_added and {
            "gripper0_right_leftfinger",
            "gripper0_right_rightfinger",
        }.issubset(self.objects):
            self.add_gripper_finger_lights(
                energy=self.gripper_light_energy,
                size=self.gripper_light_size,
                light_type=self.gripper_light_type,
                spot_size=self.gripper_spot_size,
                spot_blend=self.gripper_spot_blend,
                show_marker=self.show_gripper_light_marker,
            )
            self._gripper_lights_added = True

    def sync_arena_geoms(self, geoms: Mapping[str, GeomState]) -> None:
        for name, geom in geoms.items():
            if name not in self.arena_objects:
                self.arena_objects[name] = self._add_arena_geom(geom)
            obj = self.arena_objects[name]
            obj.location = geom.position
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = geom.quaternion_wxyz

    def sync_cube_geom(self, geom: GeomState) -> None:
        scale = (2.0 * geom.size[0], 2.0 * geom.size[1], 2.0 * geom.size[2])
        obj = self.objects.get("cube_main")
        if obj is None:
            obj = self._add_cube("cube_main", scale=scale, material="cube_mat")
            self.objects["cube_main"] = obj
        else:
            obj.scale = scale

        obj.location = geom.position
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = geom.quaternion_wxyz

    def _sync_debug_sphere(
        self,
        name: str,
        position: tuple[float, float, float],
        radius: float = 0.012,
        color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    ) -> None:
        mat_name = f"{name}_mat"
        if mat_name not in self.bpy.data.materials:
            self._make_emission_material(mat_name, color, strength=3.0)
        obj = self.objects.get(name)
        if obj is None:
            self.bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius)
            obj = self.bpy.context.object
            obj.name = name
            obj.data.materials.append(self.bpy.data.materials[mat_name])
            self.objects[name] = obj
        obj.location = position


    def set_camera_from_mujoco(self, camera: CameraState) -> None:
        scene_camera = self.bpy.context.scene.camera
        if scene_camera is None:
            self.bpy.ops.object.camera_add()
            scene_camera = self.bpy.context.object
            self.bpy.context.scene.camera = scene_camera
        scene_camera.name = f"mujoco_{camera.name}"
        scene_camera.location = camera.position
        scene_camera.rotation_mode = "QUATERNION"
        # scene_camera.rotation_quaternion = _mujoco_camera_quat_to_blender(camera.quaternion_wxyz)
        scene_camera.rotation_quaternion = camera.quaternion_wxyz
        scene_camera.data.sensor_fit = "VERTICAL" # 对齐 mujoco
        scene_camera.data.angle_y = math.radians(camera.fovy_degrees)

        # print("sensor_fit", scene_camera.data.sensor_fit)
        # print("angle_x", math.degrees(scene_camera.data.angle_x))
        # print("angle_y", math.degrees(scene_camera.data.angle_y))

    def render(self, output_path: str | Path) -> Path:
        self._validate_gripper_finger_light_states()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        scene = self.bpy.context.scene
        image_settings = scene.render.image_settings
        previous = (
            scene.render.filepath,
            image_settings.file_format,
            image_settings.color_mode,
            image_settings.color_depth,
        )
        try:
            scene.render.filepath = str(path)
            if path.suffix.lower() == ".exr":
                image_settings.file_format = "OPEN_EXR"
                image_settings.color_mode = "RGB"
                image_settings.color_depth = "32"
            self.bpy.ops.render.render(write_still=True)
        finally:
            (
                scene.render.filepath,
                image_settings.file_format,
                image_settings.color_mode,
                image_settings.color_depth,
            ) = previous
        return path

    def render_rgb_outputs(self, raw_exr_path: str | Path, preview_png_path: str | Path) -> dict[str, Path]:
        self._validate_gripper_finger_light_states()
        raw_path = Path(raw_exr_path)
        preview_path = Path(preview_png_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)

        scene = self.bpy.context.scene
        image_settings = scene.render.image_settings
        previous = (
            scene.render.filepath,
            image_settings.file_format,
            image_settings.color_mode,
            image_settings.color_depth,
        )
        try:
            self.bpy.ops.render.render(write_still=False)
            render_result = self.bpy.data.images["Render Result"]

            scene.render.filepath = str(raw_path)
            image_settings.file_format = "OPEN_EXR"
            image_settings.color_mode = "RGB"
            image_settings.color_depth = "32"
            render_result.save_render(str(raw_path), scene=scene)

            scene.render.filepath = str(preview_path)
            image_settings.file_format = "PNG"
            image_settings.color_mode = "RGB"
            image_settings.color_depth = "8"
            render_result.save_render(str(preview_path), scene=scene)
        finally:
            (
                scene.render.filepath,
                image_settings.file_format,
                image_settings.color_mode,
                image_settings.color_depth,
            ) = previous
        return {"raw_exr_path": raw_path, "preview_png_path": preview_path}

    def _configure_cycles_gpu(self) -> None:
        bpy = self.bpy
        scene = bpy.context.scene
        scene.cycles.device = "GPU"

        try:
            cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
        except Exception as exc:
            print(f"Warning: could not access Blender Cycles preferences for GPU rendering: {exc}")
            return

        for compute_device_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cycles_preferences.compute_device_type = compute_device_type
                if hasattr(cycles_preferences, "refresh_devices"):
                    cycles_preferences.refresh_devices()
                else:
                    cycles_preferences.get_devices()
            except Exception:
                continue

            devices = list(getattr(cycles_preferences, "devices", ()))
            gpu_devices = [
                device for device in devices
                if getattr(device, "type", "CPU") != "CPU"
            ]
            if not gpu_devices:
                continue

            for device in devices:
                try:
                    device.use = getattr(device, "type", "CPU") != "CPU"
                except Exception:
                    pass

            gpu_names = ", ".join(getattr(device, "name", "GPU") for device in gpu_devices)
            print(f"Blender Cycles GPU rendering enabled: {compute_device_type} ({gpu_names})")
            return

        print("Warning: Blender Cycles GPU rendering was requested, but no GPU device was enabled.")

    def render_depth(self, output_path: str | Path) -> Path:
        self._validate_gripper_finger_light_states()
        import glob

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        scene = self.bpy.context.scene
        previous_compositing_group = scene.compositing_node_group
        previous_use_compositing = scene.render.use_compositing
        previous_use_pass_z = scene.view_layers[0].use_pass_z

        tree = self.bpy.data.node_groups.new("LumiGraspDepthCompositor", "CompositorNodeTree")
        try:
            scene.view_layers[0].use_pass_z = True
            scene.render.use_compositing = True
            scene.compositing_node_group = tree
            render_layers = tree.nodes.new(type="CompositorNodeRLayers")
            output = tree.nodes.new(type="CompositorNodeOutputFile")
            output.directory = str(path.parent)
            output.file_name = path.stem
            output.file_output_items.clear()
            item = output.file_output_items.new("FLOAT", "Depth")
            item.format.file_format = "OPEN_EXR"
            item.format.color_depth = "32"
            tree.links.new(render_layers.outputs["Depth"], output.inputs["Depth"])

            existing = set(glob.glob(str(path.parent / f"{path.stem}*")))
            self.bpy.ops.render.render(write_still=False)
            produced = sorted(set(glob.glob(str(path.parent / f"{path.stem}*"))) - existing)
            exr_path = path if path.exists() else Path(produced[0])
            if exr_path != path:
                exr_path.replace(path)
            return path
        finally:
            scene.compositing_node_group = previous_compositing_group
            scene.render.use_compositing = previous_use_compositing
            scene.view_layers[0].use_pass_z = previous_use_pass_z
            self.bpy.data.node_groups.remove(tree)

    def _make_material(
        self,
        name: str,
        color: tuple[float, float, float, float],
        texture_path: Path | None = None,
        texture_repeat: tuple[float, float] = (1.0, 1.0),
        specular: float = 0.5,
        shininess: float = 0.25,
    ) -> object:
        mat = self.bpy.data.materials.get(name) or self.bpy.data.materials.new(name)
        mat.diffuse_color = color

        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
        output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = max(0.08, 1.0 - shininess)
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = specular
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = specular
        if texture_path is not None and texture_path.exists():
            tex_coord = mat.node_tree.nodes.new(type="ShaderNodeTexCoord")
            mapping = mat.node_tree.nodes.new(type="ShaderNodeMapping")
            tex_node = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
            tex_node.image = self.bpy.data.images.load(str(texture_path), check_existing=True)
            tex_node.extension = "REPEAT"
            mapping.inputs["Scale"].default_value[0] = texture_repeat[0]
            mapping.inputs["Scale"].default_value[1] = texture_repeat[1]
            mat.node_tree.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
            mat.node_tree.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
            mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        return mat

    def _set_light_object_enabled(self, obj: object, enabled: bool) -> None:
        obj.hide_render = not enabled
        obj.hide_viewport = not enabled

    def _set_object_visible(self, obj: object, visible: bool) -> None:
        obj.hide_render = not visible
        obj.hide_viewport = not visible

    def _validate_gripper_finger_light_states(self) -> None:
        unknown = [
            finger
            for finger, enabled in self.gripper_finger_light_states.items()
            if enabled is None
        ]
        if unknown:
            missing = ", ".join(unknown)
            raise RuntimeError(
                "Gripper finger light state is unknown. "
                f"Call set_gripper_finger_lights() for: {missing}"
            )

    def _add_cube(
        self,
        name: str,
        scale: tuple[float, float, float],
        material: str,
    ) -> object:
        self.bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = self.bpy.context.object
        obj.name = name
        obj.scale = scale
        obj.data.materials.append(self.bpy.data.materials[material])
        self._assign_box_uv(obj)
        return obj

    def _add_arena_geom(self, geom: GeomState) -> object:
        material = self._arena_material_name(geom)
        if geom.name == "fixed_mount0_torso_vis" and self._mount_pedestal_obj_path().exists():
            obj = self.bpy.data.objects.new(geom.name, None)
            obj.empty_display_type = "PLAIN_AXES"
            self.bpy.context.collection.objects.link(obj)
        elif geom.name == "fixed_mount0_pedestal_vis":
            obj = self._add_mount_pedestal_mesh(geom, material)
        elif geom.geom_type == 0:
            obj = self._add_plane(geom.name, half_size=geom.size[:2], material=material)
        elif geom.geom_type == 5:
            obj = self._add_cylinder(
                geom.name,
                radius=geom.size[0],
                half_height=geom.size[1],
                material=material,
            )
        elif geom.geom_type == 6:
            obj = self._add_cube(
                geom.name,
                scale=(2.0 * geom.size[0], 2.0 * geom.size[1], 2.0 * geom.size[2]),
                material=material,
            )
        else:
            obj = self._add_cube(geom.name, scale=(0.02, 0.02, 0.02), material=material)
        return obj

    def _add_mount_pedestal_mesh(self, geom: GeomState, material: str) -> object:
        mesh_path = self._mount_pedestal_obj_path()
        if not mesh_path.exists():
            return self._add_cube(geom.name, scale=(0.1, 0.1, 0.1), material=material)

        mesh = self._load_obj_mesh_data(mesh_path, geom.name)
        if mesh is None:
            return self._add_cube(geom.name, scale=(0.1, 0.1, 0.1), material=material)

        root = self.bpy.data.objects.new(geom.name, None)
        root.empty_display_type = "PLAIN_AXES"
        self.bpy.context.collection.objects.link(root)

        obj = self.bpy.data.objects.new(f"{geom.name}_mesh", mesh)
        self.bpy.context.collection.objects.link(obj)
        obj.parent = root
        obj.matrix_parent_inverse.identity()
        obj.rotation_mode = "QUATERNION"
        if geom.mesh_position is not None and geom.mesh_quaternion_wxyz is not None:
            from mathutils import Matrix, Quaternion, Vector

            mesh_quat = Quaternion(geom.mesh_quaternion_wxyz)
            mesh_pos = Vector(geom.mesh_position)
            inverse_mesh_transform = Matrix.LocRotScale(mesh_pos, mesh_quat, None).inverted()
            obj.matrix_basis = inverse_mesh_transform
        else:
            obj.location = (0.0, 0.0, 0.0)
            obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        obj.data.materials.append(self.bpy.data.materials[material])
        return root

    def _load_obj_mesh_data(self, mesh_path: Path, mesh_name: str) -> object | None:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, ...]] = []
        with mesh_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z, *_ = line.split()
                    vertices.append((float(x), float(y), float(z)))
                elif line.startswith("f "):
                    indices = []
                    for token in line.split()[1:]:
                        vertex_idx = token.split("/", maxsplit=1)[0]
                        if not vertex_idx:
                            continue
                        idx = int(vertex_idx)
                        if idx < 0:
                            idx = len(vertices) + idx + 1
                        indices.append(idx - 1)
                    if len(indices) >= 3:
                        faces.append(tuple(indices))

        if not vertices or not faces:
            return None
        mesh = self.bpy.data.meshes.new(f"{mesh_name}_mesh_data")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        return mesh

    def _mount_pedestal_obj_path(self) -> Path:
        return self.robosuite_asset_dir / "bases" / "meshes" / "rethink_mount" / "pedestal.obj"

    def _add_plane(
        self,
        name: str,
        half_size: tuple[float, float],
        material: str,
    ) -> object:
        self.bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = self.bpy.context.object
        obj.name = name
        obj.scale = (2.0 * half_size[0], 2.0 * half_size[1], 1.0)
        obj.data.materials.append(self.bpy.data.materials[material])
        self._assign_plane_uv(obj)
        return obj

    def _add_cylinder(
        self,
        name: str,
        radius: float,
        half_height: float,
        material: str,
    ) -> object:
        self.bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=2.0 * half_height)
        obj = self.bpy.context.object
        obj.name = name
        obj.data.materials.append(self.bpy.data.materials[material])
        self._assign_cylinder_uv(obj)
        return obj

    def _add_light_marker(
        self,
        parent: object,
        name: str,
        local_position: tuple[float, float, float],
        local_rotation: tuple[float, float, float],
        color: tuple[float, float, float],
        length: float,
        radius: float,
    ) -> object:
        mat = self._make_emission_material(f"{name}_mat", (*color, 1.0), strength=2.5)

        marker_root = self.bpy.data.objects.new(name, None)
        marker_root.empty_display_type = "PLAIN_AXES"
        self.bpy.context.collection.objects.link(marker_root)
        marker_root.parent = parent
        marker_root.matrix_parent_inverse.identity()
        marker_root.location = local_position
        marker_root.rotation_euler = local_rotation

        self.bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=length)
        shaft = self.bpy.context.object
        shaft.name = f"{name}_shaft"
        shaft.data.materials.append(mat)
        shaft.parent = marker_root
        shaft.matrix_parent_inverse.identity()
        shaft.location = (0.0, 0.0, -0.5 * length)
        shaft.rotation_euler = (0.0, 0.0, 0.0)

        head_depth = length * 0.35
        self.bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=0.0, radius2=radius * 2.4, depth=head_depth)
        head = self.bpy.context.object
        head.name = f"{name}_head"
        head.data.materials.append(mat)
        head.parent = marker_root
        head.matrix_parent_inverse.identity()
        head.location = (0.0, 0.0, -length - 0.5 * head_depth)
        head.rotation_euler = (0.0, 0.0, 0.0)
        return marker_root

    def _make_emission_material(
        self,
        name: str,
        color: tuple[float, float, float, float],
        strength: float,
    ) -> object:
        mat = self.bpy.data.materials.get(name) or self.bpy.data.materials.new(name)
        mat.diffuse_color = color
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        emission = mat.node_tree.nodes.new(type="ShaderNodeEmission")
        output = mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        emission.inputs["Color"].default_value = color
        emission.inputs["Strength"].default_value = strength
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    def _assign_plane_uv(self, obj: object) -> None:
        mesh = obj.data
        uv_layer = mesh.uv_layers.new(name="UVMap")
        scale = obj.scale
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                co = mesh.vertices[mesh.loops[loop_idx].vertex_index].co
                uv_layer.data[loop_idx].uv = (co.x * scale.x + 0.5 * scale.x, co.y * scale.y + 0.5 * scale.y)

    def _assign_box_uv(self, obj: object) -> None:
        mesh = obj.data
        uv_layer = mesh.uv_layers.new(name="UVMap")
        scale = obj.scale
        for poly in mesh.polygons:
            normal = poly.normal
            axis = max(range(3), key=lambda idx: abs(normal[idx]))
            for loop_idx in poly.loop_indices:
                co = mesh.vertices[mesh.loops[loop_idx].vertex_index].co
                if axis == 2:
                    uv = (co.x * scale.x + 0.5 * scale.x, co.y * scale.y + 0.5 * scale.y)
                elif axis == 1:
                    uv = (co.x * scale.x + 0.5 * scale.x, co.z * scale.z + 0.5 * scale.z)
                else:
                    uv = (co.y * scale.y + 0.5 * scale.y, co.z * scale.z + 0.5 * scale.z)
                uv_layer.data[loop_idx].uv = uv

    def _assign_cylinder_uv(self, obj: object) -> None:
        import math

        mesh = obj.data
        uv_layer = mesh.uv_layers.new(name="UVMap")
        z_values = [vertex.co.z for vertex in mesh.vertices]
        z_min = min(z_values)
        z_max = max(z_values)
        z_span = z_max - z_min
        radius = max((vertex.co.x**2 + vertex.co.y**2) ** 0.5 for vertex in mesh.vertices)
        for poly in mesh.polygons:
            cap = abs(poly.normal.z) > 0.5
            for loop_idx in poly.loop_indices:
                co = mesh.vertices[mesh.loops[loop_idx].vertex_index].co
                if cap:
                    uv = (0.5 + 0.5 * co.x / radius, 0.5 + 0.5 * co.y / radius)
                else:
                    u = (math.atan2(co.y, co.x) / (2.0 * math.pi)) % 1.0
                    v = (co.z - z_min) / z_span
                    uv = (u, v)
                uv_layer.data[loop_idx].uv = uv

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
        if self.robot_asset_dir is None or not self.robot_asset_dir.exists():
            return {}
        return load_panda_visual_meshes(self.robot_asset_dir)

    def _add_mesh_body(self, name: str, specs: list[MeshSpec]) -> object:
        bpy = self.bpy
        root = bpy.data.objects.new(name, None)
        root.empty_display_type = "PLAIN_AXES"
        bpy.context.collection.objects.link(root)
        for idx, spec in enumerate(specs):
            self._import_mesh_object(parent=root, spec=spec, suffix=idx)
        return root

    def _import_mesh_object(self, parent: object, spec: MeshSpec, suffix: int) -> None:
        bpy = self.bpy
        before = set(bpy.data.objects)
        path = spec.mesh_path
        if path.suffix.lower() == ".obj":
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        elif path.suffix.lower() == ".stl":
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            return

        imported = [obj for obj in bpy.data.objects if obj not in before]
        mat_name = self._material_name(spec)
        if spec.rgba is not None:
            self._make_material(mat_name, spec.rgba, specular=spec.specular, shininess=spec.shininess)

        for obj_idx, obj in enumerate(imported):
            obj.name = f"{parent.name}_{spec.mesh_name}_{suffix}_{obj_idx}"
            obj.parent = parent
            obj.matrix_parent_inverse.identity()
            obj.location = spec.local_position
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = spec.local_quaternion_wxyz
            if spec.rgba is not None and hasattr(obj.data, "materials"):
                obj.data.materials.clear()
                obj.data.materials.append(bpy.data.materials[mat_name])
            self._imported_mesh_count += 1

    def _material_name(self, spec: MeshSpec) -> str:
        if spec.material_name:
            return f"panda_{spec.material_name}"
        return "robot_mat"
    
