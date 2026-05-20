"""Joint Adaptation Network baseline for unsupervised domain adaptation."""

from __future__ import annotations

import argparse
import math
import sys
import time
from itertools import cycle
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from data import DATASET_SPECS, build_data, normalize_dataset_name
    from erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import DATASET_SPECS, build_data, normalize_dataset_name
    from .erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class JAN(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, _ = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.classifier(features)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.feature_extractor(source_images)
        target_features = self.feature_extractor(target_images)
        source_logits = self.classifier(source_features)
        target_logits = self.classifier(target_features)
        return source_logits, target_logits, source_features, target_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAN baseline for UDA datasets.")
    parser.add_argument("--data-root", default="data", help="Dataset root or parent directory.")
    parser.add_argument("--dataset", default="officehome", help="UDA dataset name.")
    parser.add_argument("--source", default="Art", help="Source domain name.")
    parser.add_argument("--target", default="Clipart", help="Target domain name.")
    parser.add_argument("--source-list", default=None, help="Optional source list file with image path and label.")
    parser.add_argument("--target-list", default=None, help="Optional target list file. Labels are optional.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes.")
    parser.add_argument("--arch", default="resnet50", help="Torchvision model name or small_cnn.")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights if available.")
    parser.add_argument("--image-size", type=int, default=None, help="Input image size.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--jan-loss-weight", type=float, default=1.0)
    parser.add_argument("--kernel-sigma", type=float, default=None, help="Base RBF bandwidth. Uses batch estimate by default.")
    parser.add_argument("--kernel-mul", type=float, default=2.0, help="Bandwidth multiplier for multi-kernel JMMD.")
    parser.add_argument("--kernel-num", type=int, default=5, help="Number of RBF kernels per joint layer.")
    parser.add_argument("--no-prediction-kernel", action="store_true", help="Align only features instead of feature-prediction joint.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Limit batches per epoch for quick runs.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoints when > 0.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-fake-data", action="store_true", help="Run without real images for quick checks.")
    parser.add_argument("--fake-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    set_seed(args.seed)

    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    if bundle.target_train is None:
        raise ValueError("JAN requires a target training loader.")

    device = torch.device(args.device)
    model = JAN(
        arch=args.arch,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "jan_loss",
            "total_loss",
            "source_acc",
            "target_acc",
            "lr",
            "elapsed_sec",
        ],
    )

    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(
        f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) "
        f"| target: {args.target} ({bundle.target_train_size})"
    )
    print(
        f"Method: JAN | joint_prediction_kernel: {not args.no_prediction_kernel} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, args)
        if scheduler is not None:
            scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]

        elapsed = time.time() - started_at
        lr = optimizer.param_groups[0]["lr"]
        logger.log(
            {
                "epoch": epoch,
                "class_loss": train_metrics["class_loss"],
                "jan_loss": train_metrics["jan_loss"],
                "total_loss": train_metrics["total_loss"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"jan_loss={train_metrics['jan_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    jan_losses = AverageMeter()
    total_losses = AverageMeter()
    target_iter = cycle(target_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            source_logits, target_logits, source_features, target_features = model(source_images, target_images)
            class_loss = F.cross_entropy(source_logits, source_labels)

            source_layers = [source_features]
            target_layers = [target_features]
            if not args.no_prediction_kernel:
                source_layers.append(F.softmax(source_logits, dim=1))
                target_layers.append(F.softmax(target_logits, dim=1))

            # JAN aligns the joint distribution of task-specific representations across domains.
            jan_loss = compute_jmmd_loss(source_layers, target_layers, args)
            loss = class_loss + args.jan_loss_weight * jan_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        jan_losses.update(jan_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "jan_loss": jan_losses.avg,
        "total_loss": total_losses.avg,
    }


def compute_jmmd_loss(source_layers: list[torch.Tensor], target_layers: list[torch.Tensor], args) -> torch.Tensor:
    if len(source_layers) != len(target_layers):
        raise ValueError("source_layers and target_layers must have the same length.")

    batch_size = min(source_layers[0].size(0), target_layers[0].size(0))
    if batch_size <= 0:
        raise ValueError("JMMD requires non-empty source and target batches.")

    joint_kernel = None
    for source, target in zip(source_layers, target_layers):
        source = source[:batch_size].flatten(1).float()
        target = target[:batch_size].flatten(1).float()
        kernel = gaussian_kernel_matrix(
            source,
            target,
            kernel_mul=args.kernel_mul,
            kernel_num=args.kernel_num,
            fixed_sigma=args.kernel_sigma,
        )
        joint_kernel = kernel if joint_kernel is None else joint_kernel * kernel

    if joint_kernel is None:
        raise ValueError("JMMD requires at least one joint layer.")

    source_source = joint_kernel[:batch_size, :batch_size].mean()
    target_target = joint_kernel[batch_size:, batch_size:].mean()
    source_target = joint_kernel[:batch_size, batch_size:].mean()
    target_source = joint_kernel[batch_size:, :batch_size].mean()
    return (source_source + target_target - source_target - target_source).clamp_min(0.0)


def gaussian_kernel_matrix(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    fixed_sigma: float | None,
) -> torch.Tensor:
    if kernel_num <= 0:
        raise ValueError("kernel_num must be positive.")
    if kernel_mul <= 0:
        raise ValueError("kernel_mul must be positive.")

    features = torch.cat([source, target], dim=0)
    distances = torch.cdist(features, features, p=2).pow(2)
    if fixed_sigma is None:
        count = features.size(0)
        bandwidth = distances.detach().sum() / max(count * count - count, 1)
    else:
        bandwidth = torch.as_tensor(float(fixed_sigma), device=features.device, dtype=features.dtype)
    bandwidth = bandwidth.clamp_min(1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))

    kernels = []
    for idx in range(kernel_num):
        kernels.append(torch.exp(-distances / (bandwidth * (kernel_mul ** idx))))
    return sum(kernels)


@torch.no_grad()
def evaluate(model: JAN, loader, device) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict(images)
        valid = labels >= 0
        if valid.sum().item() > 0:
            loss = F.cross_entropy(logits[valid], labels[valid])
            batch_acc, count = accuracy(logits, labels)
            losses.update(loss.item(), count)
            acc_meter.update(batch_acc, count)

    if acc_meter.count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": losses.avg, "acc": acc_meter.avg}


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"jan_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, target_acc: float) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "target_acc": target_acc,
        "args": to_serializable_args(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
