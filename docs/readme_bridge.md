# LumiGrasp MuJoCo / Blender Bridge 设计说明

## 目标

LumiGrasp 的目标是把 robosuite / MuJoCo 的物理仿真和 Blender 的外观、光照、材质渲染结合起来：

- MuJoCo / robosuite 负责环境创建、动力学、关节控制、物体状态和相机状态。
- Blender 负责最终图像渲染，包括材质、贴图、光照、阴影和更真实的外观。
- 不修改 MuJoCo、robosuite、Blender 或 `third_party/MuBlE` 的库文件。
- 需要使用的 robosuite XML / mesh / texture 资产复制到项目内 `assets/`，避免运行时直接依赖 site-packages 中的资产文件。

当前阶段实现的是一个薄桥接层：运行 robosuite `Lift + Panda` 环境，读取 MuJoCo 编译后的状态，在 Blender 中创建对应对象并同步位姿，然后输出 Blender 渲染结果。

## 参考项目

`third_party/MuBlE` 做过类似工作，主要参考文件包括：

- `third_party/MuBlE/environment/tabletop_env.py`
- `third_party/MuBlE/environment/renderer.py`
- `third_party/MuBlE/environment/objects.py`
- `third_party/MuBlE/environment/scene_parser.py`

MuBlE 的思路对位姿同步和 Blender 端对象管理有参考价值，但它和旧版 robosuite API 绑定较深，也包含较多硬编码对象名。当前实现选择在 LumiGrasp 内新建独立 adapter 层，让第三方库保持不变（仅用来参考）。

## 当前文件结构

核心代码：

- `lumigrasp/bridge/robosuite_env.py`：创建 robosuite demo 环境，当前默认是 `Lift + Panda`，并提供 random action 采样。
- `lumigrasp/bridge/mujoco_state.py`：从 `env.sim` 读取 MuJoCo 编译后的 body、geom、camera 状态。
- `lumigrasp/bridge/panda_assets.py`：解析 `assets/robot_asset/franka_emika_panda/panda.xml`，得到 Panda link 到 visual mesh 的映射。
- `lumigrasp/bridge/blender_renderer.py`：创建 Blender 场景，导入 Panda mesh，重建 arena primitive，绑定材质贴图，同步位姿并渲染。
- `lumigrasp/bridge/pipeline.py`：组织完整流程，包括静态对齐 demo 和 random action demo。

入口脚本：

- `scripts/run_static_bridge_demo.py`：渲染同一个初始 state 的 MuJoCo 原生图和 Blender 图，用于检查两者是否对齐。
- `scripts/run_bridge_demo.py`（调试中）：执行 random action，并按帧同步 MuJoCo 状态到 Blender。

配置和输出：

- `configs/bridge_demo.yaml`：默认 bridge 配置。
- `outputs/static_bridge_demo*/`：静态对齐渲染输出。
- `outputs/bridge_demo/`：random action demo 输出帧和 metadata。

## 项目内资产

Panda 机械臂资产：

```text
assets/robot_asset/franka_emika_panda/
```

这份资产来自当前项目内的 menagerie Panda 资产。robosuite 自带 Panda 资产已经检查过，黑色材质本质上也是约 `rgba=(0.25, 0.25, 0.25, 1)`，直接替换并不能解决“黑色不够深”的问题。当前保留 menagerie 版本，但确保它位于默认路径下。

robosuite 场景和底座资产：

```text
assets/robosuite_assets/
```
本场景是原生 robosuite 中 `Lift` 场景。

底座使用 `pedestal.obj`，因为该 OBJ 经检查包含完整正确的底座外观。MuJoCo XML 原始引用的是 STL，但 Blender 渲染侧使用 OBJ 以获得完整视觉结构。

## 运行命令
`python scripts/run_static_bridge_demo.py`

更多参数在脚本内自行配置，主要包括：

- `SETTLE_STEPS`: 决定是否进行初步物理仿真（robosuite 实现中，cube 初始位置有 0.01m 的 offset，settling phase 用于在环境初始化之后等待 cube 自然下落直到静止）

- `RENDER_DEPTH`: 决定是否渲染 mujoco 与 blender 两侧的相机深度图，以及二者深度的 L2 error map

- `GRIPPER_LIGHT_TYPE`: 选择夹爪灯光两种模式，点光源 `"POINT"` 或者射灯 `"SPOT"`



---

以下部分由 codex 主动总结，还未审阅（2026-5-17）

---

## MuJoCo / Blender 状态对齐规则

桥接层把 MuJoCo / robosuite 当作状态源。Blender 不自己推进物理状态，只根据 MuJoCo 当前帧状态更新对象。

### 动态 body

机器人 link、夹爪 link、cube 等动态对象从 MuJoCo body state 读取：

- 位置：`sim.data.body_xpos`
- 姿态：`sim.data.body_xquat`

在 Blender 中直接映射为：

- `body_xpos -> object.location`
- `body_xquat -> object.rotation_quaternion`

当前桥接中 MuJoCo 和 Blender 的四元数顺序都按 `wxyz` 使用，因此不需要额外重排。

### Panda visual mesh

Panda 机械臂由 `panda_assets.py` 解析 XML 得到 visual mesh：

- 每个 MuJoCo body 对应一个 Blender empty root。
- 每个 visual mesh 作为 root 的 child object。
- root 跟随 MuJoCo body 的世界位姿。
- child mesh 保留 XML 中相对 link 的 `pos / quat`。

这就是为什么 Panda mesh 不像底座那样额外处理 `mesh_pos / mesh_quat`：Panda 的 visual local transform 已经在 XML 解析阶段保留到了 child mesh 上。

右夹爪需要保留 body-level 翻转。`panda_assets.py` 中对 `right_finger` 做了局部 quaternion 修正，否则两个夹爪会朝同一方向。

### Cube visual geom

Lift 环境中的 cube 不再由 Blender 固定边长创建，而是每帧从 MuJoCo visual geom `cube_g0_vis` 同步：

- 位置：`sim.data.geom_xpos[cube_g0_vis] -> object.location`
- 姿态：`sim.data.geom_xmat[cube_g0_vis] -> object.rotation_quaternion`
- 尺寸：`2 * sim.model.geom_size[cube_g0_vis] -> object.scale`

这里必须乘以 `2`，因为 MuJoCo box geom 的 `size` 是半尺寸。robosuite `Lift` 中 `BoxObject(size_min=[0.020, 0.020, 0.020], size_max=[0.022, 0.022, 0.022])` 表示完整边长约为 `0.040m ~ 0.044m`，不是 `0.020m ~ 0.022m`。

动态 demo 中每次 MuJoCo step 后都会重新读取 `cube_g0_vis`，因此 cube 的移动、旋转和随机尺寸都会跟随 MuJoCo。

### Arena primitive

arena 中的地面、墙、桌面、桌腿来自 MuJoCo compiled geom state：

- 位置：`sim.data.geom_xpos`
- 姿态矩阵：`sim.data.geom_xmat`
- 尺寸：`sim.model.geom_size`
- 类型：`sim.model.geom_type`

当前转换规则：

- MuJoCo `plane`：Blender plane，XY scale 为 `2 * geom.size[:2]`。
- MuJoCo `box`：Blender cube，XYZ scale 为 `2 * geom.size`。
- MuJoCo `cylinder`：Blender cylinder，半径为 `geom.size[0]`，高度为 `2 * geom.size[1]`。

注意：MuJoCo 的 box / plane size 是半尺寸，Blender primitive 使用完整尺寸，因此必须乘以 `2`。当前代码已经按这个规则处理。

### Mesh geom：底座 pedestal

mesh geom 比 primitive 多一层 mesh asset frame。MuJoCo 编译 mesh 后会生成：

- `sim.model.mesh_pos`
- `sim.model.mesh_quat`

`fixed_mount0_pedestal_vis` 的 Blender 处理方式是：

1. Blender root object 跟随 MuJoCo `geom_xpos / geom_xmat`。
2. 子 OBJ mesh 使用项目内 `pedestal.obj`。
3. 子 OBJ mesh 再应用 `inverse(mesh_pos, mesh_quat)`。

这样做的原因是 raw OBJ 顶点坐标和 MuJoCo 最终 geom frame 并不总是一致。如果只把 OBJ 放到 `geom_xpos / geom_xmat`，底座会倾倒或偏移。

由于完整 `pedestal.obj` 已经包含可见底座结构，`fixed_mount0_torso_vis` 在 OBJ 存在时只创建 empty，不再额外画一个 cube，避免重复重叠。

### Camera

静态对齐 demo 使用 MuJoCo camera state：

- `sim.data.cam_xpos`
- `sim.data.cam_xmat`
- `sim.model.cam_fovy`

映射到 Blender：

- `cam_xpos -> camera.location`
- `cam_xmat -> camera.rotation_quaternion`
- `cam_fovy -> camera.data.angle_y`

MuJoCo offscreen 渲染保存前使用 `np.flipud` 竖直翻转，修正图像原点方向差异。

## 材质、贴图和光照

### Arena 贴图

arena 材质绑定不再依赖 MuJoCo material id，因为当前 robosuite wrapper 中 `geom.material_name` 可能为空。代码按稳定 geom name 绑定：

- `floor -> light-gray-floor-tile.png`
- `wall_* -> light-gray-plaster.png`
- `table_visual -> ceramic.png`
- `table_leg*_visual -> steel-brushed.png`

Blender 材质节点为：

```text
Texture Coordinate: UV -> Mapping -> Image Texture -> Principled BSDF Base Color
```

由于 arena primitive 是在 Blender 中重建的，不是直接导入原 mesh，所以代码显式生成 UV。UV 使用对象 scale 后的实际物理尺寸，避免 MuJoCo 半尺寸放大后贴图仍按 1x1 坐标铺开。

当前地板贴图重复数设置为 `6 x 6`，比 XML 里的 `2 x 2` 更适合 6m 级别的 Blender 可视场景；否则木纹条带会显得过宽。

### Panda 材质

Panda 本身没有图片贴图，主要依赖 XML 中的 material rgba：

- white: `1 1 1 1`
- off_white: `0.901961 0.921569 0.929412 1`
- black: `0.25 0.25 0.25 1`
- light_blue: `0.039216 0.541176 0.780392 1`

`panda_assets.py` 会解析 `rgba / specular / shininess`，`blender_renderer.py` 会将这些值写入 Blender Principled BSDF。

robosuite 自带 Panda 资产也使用类似的黑色值，因此如果只为了让黑色更深，直接换 robosuite Panda 资产不是有效方案。更合理的控制点是 Blender 的光照、曝光和色彩管理。

### 当前 Blender 光照设置

当前 Blender 场景使用 Cycles：

- render engine: `CYCLES`
- key area light location: `(0.0, -3.5, 5.0)`
- key area light energy: `180.0`
- key area light size: `5.0`
- view transform: `Filmic`
- look: `Medium High Contrast`
- exposure: `-0.45`

之前 `energy=450` 且 exposure 不压低时，白色桌面和 Panda 白色外壳容易过曝成一片。当前设置保留高光层次，同时让黑色部件更接近合理暗度。

### 夹爪灯光

夹爪灯光在 `lumigrasp/bridge/blender_renderer.py` 中由 `add_gripper_light()` 和 `add_gripper_finger_lights()` 创建。默认把红色灯挂到 `gripper0_right_leftfinger`，蓝色灯挂到 `gripper0_right_rightfinger`，这样灯会跟随左右手指各自的开合运动，而不是只跟随 `gripper0_right_eef`。

关键参数含义：

- `light_type`：`"POINT"` 为点光源，向所有方向发光；`"SPOT"` 为射灯，默认沿 Blender 局部 `-Z` 方向发光。
- `energy`：灯光强度。
- `size`：不是可见灯泡尺寸。对点光源和射灯都映射到 Blender 的 `shadow_soft_size`，主要影响阴影软硬和等效发光半径。
- `spot_size`：仅用于射灯，表示射灯锥角，单位是弧度。
- `spot_blend`：仅用于射灯，控制射灯边缘从亮到暗的过渡柔和度。
- `show_marker`：是否额外渲染一个可见箭头标记来指示灯光位置和方向。箭头尾部中心就是光源点，箭头尖端指向射灯照射方向。该标记是 Blender mesh，只用于调试可视化，不参与 MuJoCo 物理。

示例：

```python
renderer.add_gripper_finger_lights(
    energy=35.0,
    size=0.015,
    light_type="SPOT",
    spot_size=0.55,
    spot_blend=0.35,
    show_marker=True,
)
```

脚本中不通过 parser 暴露这些参数，而是在入口文件顶部硬编码配置：

```python
GRIPPER_LIGHT_TYPE = "SPOT"
GRIPPER_LIGHT_ENERGY = 35.0
GRIPPER_LIGHT_SIZE = 0.015
GRIPPER_SPOT_SIZE = 0.55
GRIPPER_SPOT_BLEND = 0.35
SHOW_GRIPPER_LIGHT_MARKER = True
```

## 当前已解决的问题

1. MuJoCo 原生渲染上下颠倒：保存前使用 `np.flipud`。
2. Panda 两个夹爪朝同一方向：保留 right finger body-level local quaternion 修正。
3. random action 看起来不动：增加 `--action-repeat`，并输出 `eef_delta_from_initial_m`。
4. Blender 缺少墙、地面、桌腿：从 MuJoCo compiled arena geoms 重建 primitive。
5. Blender 缺少 arena 贴图：复制 texture 到项目内，按 geom name 绑定材质并生成 UV。
6. 底座 pedestal 姿态错误：使用 `pedestal.obj`，并应用 MuJoCo `mesh_pos / mesh_quat` 的逆变换。
7. 白色区域过曝：降低灯光能量，使用 Filmic 并降低 exposure。
8. 地板贴图条带过宽：UV 改为物理尺寸尺度，并提高 floor repeat。

## 后续建议

1. 将 arena / robot / mount 的资产选择写入配置文件，避免在代码中固定路径。
2. 增加一个自动化对齐检查脚本，固定输出 `agentview / sideview / birdview` 三个视角。
3. 为 Blender 光照参数增加命令行选项，例如 `--light-energy`、`--exposure`、`--view-transform`。
4. 如果需要更接近 MuJoCo 原生画面，可以增加一套 “MuJoCo-like render preset”；如果目标是真实感，可以继续在 Blender 中调 HDRI、area light、roughness、shadow softness。
5. 未来如果替换成更多 mesh-based 场景物体，需要统一处理 MuJoCo mesh asset frame，即 `geom pose + inverse(mesh_pos, mesh_quat)`。
