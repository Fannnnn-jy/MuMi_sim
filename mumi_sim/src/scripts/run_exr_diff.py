from pathlib import Path
import mitsuba as mi
import numpy as np

base = Path("/home/ubuntu/Junyi/LumiGrasp/mumi_sim/outputs/static_bridge_demo")

a_path = base / "no_light_mitsuba.exr"
b_path = base / "right_light_mitsuba.exr"

a = np.array(mi.Bitmap(str(a_path)), dtype=np.float32)
b = np.array(mi.Bitmap(str(b_path)), dtype=np.float32)

diff = b - a
abs_diff = np.abs(diff)

mi.Bitmap(diff).write(str(base / "right_minus_no_light.exr"))
mi.Bitmap(abs_diff).write(str(base / "abs_right_minus_no_light.exr"))

# 可视化 PNG：自动拉伸
vis = abs_diff
scale = np.percentile(vis, 99)
if scale > 1e-8:
    vis = np.clip(vis / scale, 0, 1)

mi.Bitmap(vis).convert(
    pixel_format=mi.Bitmap.PixelFormat.RGB,
    component_format=mi.Struct.Type.UInt8,
    srgb_gamma=True,
).write(str(base / "abs_right_minus_no_light.png"))

print("saved:")
print(base / "right_minus_no_light.exr")
print(base / "abs_right_minus_no_light.exr")
print(base / "abs_right_minus_no_light.png")