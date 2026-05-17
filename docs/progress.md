# Progress Log
> 本文档用于记录 mujoco (robosuite) + blender 路径开发仿真环境进行主动光抓取的日志

## 2026-05-15

进行了 mujoco 原生光照渲染实验，其光影贴图有比较严重的立方体投影样式光晕，无法渲染真实光影。确认进行 mujoco + blender 路径探索。


## 2026-05-17

[x] 依赖新版代码库复现所需要的 MuBlE 基座并命名过程为 bridge。

[x] 完成静态环境下场景搭建，mujoco 与 blender 中建模对齐已经过可视化验证。

[x] 完成机械臂夹爪光照配置，可视化验证。

[x] 暴露深度渲染和 error map 接口。

# TODO

[ ] cube-lifting 强化学习环境搭建。

[ ] Dense reward 设置

[ ] 不同主动光渲染图处理

[ ] 初步训练抓取