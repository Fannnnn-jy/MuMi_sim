# DrQ-v2 Training (LumiGrasp)

DrQ-v2 RL training on the `mumi_sim` cube-lift task, with three image observation
modes: RGB, RGB-D, and light-difference (Mitsuba) images.

## Setup

Dependencies are managed by [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync
```

This creates `.venv` and installs everything from `pyproject.toml` / `uv.lock`.
Prefix every command below with `uv run` to use it.

> Requires a CUDA GPU. Rendering uses EGL (`MUJOCO_GL=egl`, set automatically).

## Train

The entry point is `train.py`, configured via Hydra (`cfgs/config.yaml`).

```bash
# pick an observation mode and an output dir
uv run python train.py \
  env.observation_mode=rgb_stack \
  hydra.run.dir=./exp_local/rgb
```

Observation modes (`env.observation_mode`):

| mode         | observation                          |
|--------------|--------------------------------------|
| `rgb_stack`  | stacked RGB frames (default)         |
| `rgbd`       | RGB + depth                          |
| `light_diff` | Mitsuba light-difference images      |

Common overrides (any field in `cfgs/config.yaml`):

```bash
uv run python train.py \
  env.observation_mode=rgbd \
  hydra.run.dir=./exp_local/rgbd \
  num_train_frames=1000000 \
  batch_size=256 \
  seed=42 \
  device=cuda
```

Outputs land in the `hydra.run.dir`: TensorBoard logs, `train.csv`/`eval.csv`,
`cubelift.log`, eval videos, and checkpoints (`snapshot.pt` +
`checkpoints/snapshot_<frame>.pt`). Re-running with the same dir auto-resumes
from `snapshot.pt`.

### Background runs

Helper scripts loop training in the background and tee to `train.log`:

```bash
bash run_train_rgbd.sh        # observation_mode=rgbd
bash run_train_lightdiff.sh   # observation_mode=light_diff
bash run_train_rgb.sh         # observation_mode=rgb_stack
```

---
> Stuff below this line is not checked yet. 
---

## Render an eval episode

Render one episode from a trained checkpoint to an MP4:

```bash
uv run python render_eval.py \
  --work-dir ./exp_local/rgb \
  --checkpoint latest
```

`--checkpoint` accepts `latest`, a name like `snapshot_50000`, or a path.
Useful flags: `--spp` (render samples), `--size`, `--fps`, `--max-steps`,
`--output-dir`. See `--help` for all options.

## Monitor

```bash
uv run tensorboard --logdir ./exp_local
```
