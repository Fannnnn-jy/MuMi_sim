from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from lumigrasp.training_imitation.dataset import (
    LumiGraspImitationDataset,
    discover_samples,
    split_samples,
)
from lumigrasp.training_imitation.model import SimpleImitationPolicy


@dataclass(frozen=True)
class TrainConfig:
    data_root: str = "/home/ubuntu/Junyi/LumiGrasp/data/pickcube-v1"
    output_dir: str = "/home/ubuntu/Junyi/LumiGrasp/runs/imitation"
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    image_size: int = 512
    val_fraction: float = 0.2
    num_workers: int = 16
    seed: int = 42
    dpos_clip_norm: float | None = 10.0 # 0.10
    quat_key: str = "delta_quat_xyzw"
    device: str = "cuda"

    log_interval: int = 10
    overfit_samples: int = 0
    position_loss_weight: float = 10.0
    quat_loss_weight: float = 1.0
    checkpoint_interval: int = 10

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weighted_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    position_weight: float,
    quat_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_loss = criterion(pred[:, :3], target[:, :3])
    quat_loss = criterion(pred[:, 3:7], target[:, 3:7])
    loss = position_weight * pos_loss + quat_weight * quat_loss
    return loss, pos_loss, quat_loss


def _serialize_sample(sample: object, index: int) -> dict[str, object]:
    return {
        "selection_index": index,
        "metadata_path": str(sample.metadata_path),
        "sample_dir": str(sample.metadata_path.parent),
        "image_paths": [str(path) for path in sample.image_paths],
        "qpos6": list(sample.qpos6),
        "target_action": list(sample.action),
        "target_delta_xyz": list(sample.action[:3]),
        "target_delta_quat": list(sample.action[3:7]),
    }


def _write_overfit_manifest(output_dir: Path, samples: list[object], config: TrainConfig) -> None:
    manifest = {
        "mode": "overfit",
        "n_samples": len(samples),
        "data_root": config.data_root,
        "image_size": config.image_size,
        "quat_key": config.quat_key,
        "dpos_clip_norm": config.dpos_clip_norm,
        "position_loss_weight": config.position_loss_weight,
        "quat_loss_weight": config.quat_loss_weight,
        "samples": [_serialize_sample(sample, index) for index, sample in enumerate(samples)],
    }
    (output_dir / "overfit_samples.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _log_overfit_predictions(
    path: Path,
    model: nn.Module,
    dataset: LumiGraspImitationDataset,
    samples: list[object],
    device: torch.device | str,
    epoch: int,
) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            batch = dataset[index]
            images = batch["images"].unsqueeze(0).to(device)
            qpos = batch["qpos"].unsqueeze(0).to(device)
            target = batch["action"].cpu()
            pred = model(images, qpos).squeeze(0).detach().cpu()
            error = pred - target
            rows.append({
                "epoch": epoch,
                "selection_index": index,
                "metadata_path": str(sample.metadata_path),
                "sample_dir": str(sample.metadata_path.parent),
                "qpos6": list(sample.qpos6),
                "target_action": target.tolist(),
                "pred_action": pred.tolist(),
                "error": error.tolist(),
                "target_delta_xyz": target[:3].tolist(),
                "pred_delta_xyz": pred[:3].tolist(),
                "delta_xyz_error_norm": float(torch.linalg.norm(error[:3])),
                "target_delta_quat": target[3:7].tolist(),
                "pred_delta_quat": pred[3:7].tolist(),
                "delta_quat_error_norm": float(torch.linalg.norm(error[3:7])),
            })
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    model.train()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int,
    split: str,
    log_interval: int,
    position_loss_weight: float,
    quat_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_pos_loss = 0.0
    total_quat_loss = 0.0
    total_count = 0
    epoch_start = time()
    last_log = epoch_start

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            qpos = batch["qpos"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)

            pred = model(images, qpos)
            loss, pos_loss, quat_loss = _weighted_action_loss(
                pred,
                target,
                criterion,
                position_loss_weight,
                quat_loss_weight,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_pos_loss += float(pos_loss.detach().cpu()) * batch_size
            total_quat_loss += float(quat_loss.detach().cpu()) * batch_size
            total_count += batch_size

            if log_interval > 0 and (step == 1 or step % log_interval == 0 or step == len(loader)):
                now = time()
                avg_loss = total_loss / max(total_count, 1)
                avg_pos_loss = total_pos_loss / max(total_count, 1)
                avg_quat_loss = total_quat_loss / max(total_count, 1)
                examples_per_sec = total_count / max(now - epoch_start, 1e-6)
                delta = now - last_log
                print(
                    f"epoch {epoch:03d} {split}: step {step}/{len(loader)} "
                    f"loss={avg_loss:.6f} pos={avg_pos_loss:.6f} quat={avg_quat_loss:.6f} "
                    f"throughput={examples_per_sec:.1f}/s dt={delta:.1f}s",
                    flush=True,
                )
                last_log = now

    denom = max(total_count, 1)
    return {
        "loss": total_loss / denom,
        "pos_loss": total_pos_loss / denom,
        "quat_loss": total_quat_loss / denom,
    }


def train(config: TrainConfig) -> dict[str, object]:
    _seed_everything(config.seed)
    device = config.device
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(
        config.data_root,
        quat_key=config.quat_key,
        dpos_clip_norm=config.dpos_clip_norm,
    )
    if config.overfit_samples > 0:
        train_samples = samples[: config.overfit_samples]
        val_samples = []
        shuffle_train = False
        _write_overfit_manifest(output_dir, train_samples, config)
        predictions_path = output_dir / "overfit_predictions.jsonl"
        predictions_path.write_text("", encoding="utf-8")
        print("Overfit mode: training only on " f"{len(train_samples)} fixed sample(s).", flush=True)
        print(f"Overfit sample manifest: {output_dir / 'overfit_samples.json'}", flush=True)
        print(f"Overfit prediction log: {predictions_path}", flush=True)
    else:
        train_samples, val_samples = split_samples(samples, val_fraction=config.val_fraction, seed=config.seed)
        shuffle_train = True

    train_dataset = LumiGraspImitationDataset(train_samples, image_size=config.image_size)
    val_dataset = LumiGraspImitationDataset(val_samples, image_size=config.image_size) if val_samples else None
    predictions_path = output_dir / "overfit_predictions.jsonl" if config.overfit_samples > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=shuffle_train,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device == "cuda",
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    model = SimpleImitationPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss()

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    start = time()
    print(f"Loaded {len(samples)} samples: train={len(train_samples)}, val={len(val_samples)}, device={device}")
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch=epoch,
            split="train",
            log_interval=config.log_interval,
            position_loss_weight=config.position_loss_weight,
            quat_loss_weight=config.quat_loss_weight,
        )
        train_loss = train_metrics["loss"]
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pos_loss": train_metrics["pos_loss"],
            "train_quat_loss": train_metrics["quat_loss"],
        }
        if val_loader is not None:
            val_metrics = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                split="val",
                log_interval=config.log_interval,
                position_loss_weight=config.position_loss_weight,
                quat_loss_weight=config.quat_loss_weight,
            )
            val_loss = val_metrics["loss"]
            row["val_loss"] = val_metrics["loss"]
            row["val_pos_loss"] = val_metrics["pos_loss"]
            row["val_quat_loss"] = val_metrics["quat_loss"]
            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(output_dir / "best.pt", model, optimizer, config, epoch, train_loss, val_loss)
        else:
            _save_checkpoint(output_dir / "best.pt", model, optimizer, config, epoch, train_loss, None)
        _save_checkpoint(output_dir / "last.pt", model, optimizer, config, epoch, train_loss, row.get("val_loss"))
        if config.checkpoint_interval > 0 and epoch % config.checkpoint_interval == 0:
            _save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                config,
                epoch,
                train_loss,
                row.get("val_loss"),
            )
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if predictions_path is not None:
            _log_overfit_predictions(predictions_path, model, train_dataset, train_samples, device, epoch)

        val_text = f", val_loss={row['val_loss']:.6f}" if "val_loss" in row else ""
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"pos={row['train_pos_loss']:.6f} quat={row['train_quat_loss']:.6f}{val_text}"
        )

    return {
        "output_dir": str(output_dir),
        "samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "seconds": time() - start,
        "best_val_loss": best_val if val_loader is not None else None,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    train_loss: float,
    val_loss: float | int | None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple LumiGrasp imitation policy.")
    parser.add_argument("--data-root", default=TrainConfig.data_root)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--dpos-clip-norm", type=float, default=TrainConfig.dpos_clip_norm)
    parser.add_argument("--disable-dpos-clip", action="store_true")
    parser.add_argument("--quat-key", default=TrainConfig.quat_key, choices=("delta_quat_xyzw", "delta_quat_wxyz"))
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    parser.add_argument("--overfit-samples", type=int, default=TrainConfig.overfit_samples)
    parser.add_argument("--position-loss-weight", type=float, default=TrainConfig.position_loss_weight)
    parser.add_argument("--quat-loss-weight", type=float, default=TrainConfig.quat_loss_weight)
    parser.add_argument("--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        dpos_clip_norm=None if args.disable_dpos_clip else args.dpos_clip_norm,
        quat_key=args.quat_key,
        device=args.device,
        log_interval=args.log_interval,
        overfit_samples=args.overfit_samples,
        position_loss_weight=args.position_loss_weight,
        quat_loss_weight=args.quat_loss_weight,
        checkpoint_interval=args.checkpoint_interval,
    )
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
