import argparse
import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import lumi_env
import utils


def _checkpoint_frame(path: Path) -> int:
    match = re.match(r"snapshot_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _resolve_checkpoint(work_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "latest":
        checkpoint_dir = work_dir / "checkpoints"
        checkpoints = sorted(
            checkpoint_dir.glob("snapshot_*.pt"),
            key=_checkpoint_frame,
        )
        if checkpoints:
            return checkpoints[-1]
        return work_dir / "snapshot.pt"

    path = Path(checkpoint).expanduser()
    if path.is_absolute():
        return path
    if checkpoint.isdigit():
        return work_dir / "checkpoints" / f"snapshot_{checkpoint}.pt"
    if re.fullmatch(r"snapshot_\d+", checkpoint):
        return work_dir / "checkpoints" / f"{checkpoint}.pt"
    return work_dir / path


def _infer_work_dir(args) -> Path:
    if args.work_dir is not None:
        return Path(args.work_dir).expanduser().resolve()

    checkpoint = Path(args.checkpoint).expanduser()
    if checkpoint.is_absolute():
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent.resolve()
        return checkpoint.parent.resolve()

    return Path.cwd()


def _load_cfg(work_dir: Path):
    hydra_cfg = work_dir / ".hydra" / "config.yaml"
    if hydra_cfg.exists():
        return OmegaConf.load(hydra_cfg)
    return OmegaConf.load(Path(__file__).resolve().parent / "cfgs" / "config.yaml")


def _move_agent_to_device(agent, device: torch.device):
    agent.device = device
    agent.encoder.to(device)
    agent.actor.to(device)
    agent.critic.to(device)
    agent.critic_target.to(device)
    return agent


def _resize_observation(obs: np.ndarray, height: int, width: int) -> np.ndarray:
    if obs.shape[-2:] == (height, width):
        return obs
    x = torch.as_tensor(obs[None].astype(np.float32))
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.squeeze(0).clamp(0, 255).byte().numpy()


def _chw_rgb_to_hwc(frame: np.ndarray) -> np.ndarray:
    return frame.transpose(1, 2, 0).astype(np.uint8, copy=False)


def _gray_to_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.squeeze(frame)
    return np.repeat(frame[:, :, None], 3, axis=2).astype(np.uint8, copy=False)


def _visualize_observation(obs: np.ndarray, observation_mode: str) -> np.ndarray:
    mode = str(observation_mode).lower()
    if mode == "light_diff":
        if obs.shape[0] < 9:
            raise ValueError(f"light_diff visualization expects 9 channels, got {obs.shape}")
        rgb = _chw_rgb_to_hwc(obs[0:3])
        left_diff = _chw_rgb_to_hwc(obs[3:6])
        right_diff = _chw_rgb_to_hwc(obs[6:9])
        return np.concatenate([rgb, left_diff, right_diff], axis=1)

    if mode == "rgbd":
        if obs.shape[0] < 4:
            raise ValueError(f"rgbd visualization expects 4 channels, got {obs.shape}")
        rgb = _chw_rgb_to_hwc(obs[0:3])
        depth = _gray_to_rgb(obs[3:4].transpose(1, 2, 0))
        return np.concatenate([rgb, depth], axis=1)

    if mode in {"rgb", "rgb_stack"}:
        return _chw_rgb_to_hwc(obs[-3:])

    if obs.shape[0] >= 3:
        return _chw_rgb_to_hwc(obs[:3])
    return _gray_to_rgb(obs[0])


def _save_video(path: Path, frames: list[np.ndarray], fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, macro_block_size=1)


def _median_adjusted_exposure(image: np.ndarray, exposure: float) -> float:
    luminance = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )
    luminance = luminance[luminance > 1e-8]
    if luminance.size == 0:
        return float(exposure)
    return float(exposure) / float(np.median(luminance))


def _render_light_diff_video_frame(env) -> np.ndarray:
    env._sync_renderer()
    scene_dict = env._renderer._scene_dict("path")
    scene, params = env._renderer._mi_load_dict(scene_dict)
    try:
        ambient = env._render_lights(scene, params, left_on=False, right_on=False)
        both_lights = env._render_lights(scene, params, left_on=True, right_on=True)
        exposure = _median_adjusted_exposure(ambient, env.cfg.observation_exposure)
        ambient_rgb = env._do_gamma_and_to_chw_uint8(ambient, exposure)
        both_lights_rgb = env._do_gamma_and_to_chw_uint8(both_lights, exposure)
    finally:
        del scene, params
    return np.concatenate(
        [_chw_rgb_to_hwc(ambient_rgb), _chw_rgb_to_hwc(both_lights_rgb)],
        axis=1,
    )


def _render_video_frame(env, obs: np.ndarray, observation_mode: str) -> np.ndarray:
    if str(observation_mode).lower() == "light_diff":
        return _render_light_diff_video_frame(env)
    return _visualize_observation(obs, observation_mode)


def render_eval(args):
    work_dir = _infer_work_dir(args)
    checkpoint = _resolve_checkpoint(work_dir, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    cfg = _load_cfg(work_dir)

    policy_height = int(cfg.env.height)
    policy_width = int(cfg.env.width)
    if args.spp is not None:
        cfg.env.samples = int(args.spp)
    if args.size is not None:
        cfg.env.width = int(args.size)
        cfg.env.height = int(args.size)
    if args.device is not None:
        cfg.device = args.device

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = _move_agent_to_device(payload["agent"], device)
    global_step = int(payload.get("_global_step", 0))

    env = lumi_env.make(cfg.env)
    frames = []
    episode_reward = 0.0
    step = 0
    try:
        time_step = env.reset()
        frames.append(_render_video_frame(env, time_step.observation, cfg.env.observation_mode))
        while not time_step.last():
            policy_obs = _resize_observation(time_step.observation, policy_height, policy_width)
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(policy_obs, global_step, eval_mode=True)
            time_step = env.step(action)
            frames.append(_render_video_frame(env, time_step.observation, cfg.env.observation_mode))
            episode_reward += float(time_step.reward)
            step += 1
            if args.max_steps is not None and step >= int(args.max_steps):
                break
    finally:
        env.close()

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else work_dir / "standalone_eval_video"
    size_text = f"{int(cfg.env.width)}x{int(cfg.env.height)}"
    output = output_dir / f"{checkpoint.stem}_{cfg.env.observation_mode}_spp{cfg.env.samples}_{size_text}.mp4"
    _save_video(output, frames, int(args.fps))
    if args.save_first_frame:
        imageio.imwrite(output.with_suffix(".png"), frames[0])

    print(f"checkpoint: {checkpoint}")
    print(f"saved: {output}")
    print(f"frames: {len(frames)}")
    print(f"steps: {step}")
    print(f"episode_reward: {episode_reward:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one eval episode from a trained DrQ-v2 checkpoint."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Training run directory. Defaults to the checkpoint parent, or cwd.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="snapshot.pt",
        help="Checkpoint path/name, or 'latest' for the largest checkpoints/snapshot_*.pt.",
    )
    parser.add_argument("--spp", type=int, default=None, help="Override env.samples for this render.")
    parser.add_argument("--size", type=int, default=None, help="Override square render width/height.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the rendered video.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for quick test renders.")
    parser.add_argument(
        "--save-first-frame",
        action="store_true",
        help="Also save the first composite frame as a PNG.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    render_eval(parse_args())
