"""ICDA-style implicit class-conditioned domain alignment for UDA."""

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
    from data import build_data, normalize_dataset_name
    from dann import compute_grl_lambda, gradient_reverse
    from erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import build_data, normalize_dataset_name
    from .dann import compute_grl_lambda, gradient_reverse
    from .erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class ICDA(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.classifier(features)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_features = self.feature_extractor(source_images)
        target_features = self.feature_extractor(target_images)
        source_logits = self.classifier(source_features)
        target_logits = self.classifier(target_features)
        return source_logits, target_logits, source_features, target_features

    def domain_logits(self, features: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        return self.domain_discriminator(gradient_reverse(features, grl_lambda))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICDA baseline for UDA datasets.")
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
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.01)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-aligned-targets", type=int, default=1)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-schedule", choices=("dann", "none"), default="dann")
    parser.add_argument("--domain-hidden-dim", type=int, default=1024)
    parser.add_argument("--domain-dropout", type=float, default=0.5)
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
    validate_args(args)
    set_seed(args.seed)

    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    if bundle.target_train is None:
        raise ValueError("ICDA requires a target training loader.")

    device = torch.device(args.device)
    model = ICDA(args.arch, bundle.num_classes, args.pretrained, args.domain_hidden_dim, args.domain_dropout).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "domain_loss",
            "target_entropy_loss",
            "total_loss",
            "source_acc",
            "target_acc",
            "aligned_source_ratio",
            "aligned_target_ratio",
            "aligned_class_count",
            "grl_lambda",
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
        f"Method: ICDA | threshold: {args.target_confidence_threshold:g} | model: {args.arch} "
        f"| classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, epoch, args)
        if scheduler is not None:
            scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]

        elapsed = time.time() - started_at
        lr = optimizer.param_groups[0]["lr"]
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"domain_loss={train_metrics['domain_loss']:.4f} "
            f"target_entropy={train_metrics['target_entropy_loss']:.4f} "
            f"aligned_t={train_metrics['aligned_target_ratio']:.3f} "
            f"classes={train_metrics['aligned_class_count']:.2f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} "
            f"lambda={train_metrics['grl_lambda']:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: ICDA, source_loader, target_loader, optimizer, scaler, device, epoch: int, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    domain_losses = AverageMeter()
    entropy_losses = AverageMeter()
    total_losses = AverageMeter()
    source_ratios = AverageMeter()
    target_ratios = AverageMeter()
    class_counts = AverageMeter()
    lambda_meter = AverageMeter()
    target_iter = cycle(target_loader)
    total_steps = args.steps_per_epoch or len(source_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        grl_lambda = compute_grl_lambda(epoch, step, total_steps, args)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            source_logits, target_logits, source_features, target_features = model(source_images, target_images)
            class_loss = F.cross_entropy(source_logits, source_labels)
            target_entropy_loss = entropy_loss(target_logits)

            target_pseudo, target_confidence = target_logits.detach().softmax(dim=1).max(dim=1)
            source_mask, target_mask, aligned_class_count = class_conditioned_masks(
                source_labels,
                target_pseudo,
                target_confidence,
                args.target_confidence_threshold,
                args.min_aligned_targets,
            )
            domain_loss = class_conditioned_domain_loss(model, source_features, target_features, source_mask, target_mask, grl_lambda)
            loss = class_loss + args.domain_loss_weight * domain_loss + args.entropy_loss_weight * target_entropy_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        domain_losses.update(domain_loss.item(), batch_size)
        entropy_losses.update(target_entropy_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        source_ratios.update(source_mask.float().mean().item(), batch_size)
        target_ratios.update(target_mask.float().mean().item(), batch_size)
        class_counts.update(float(aligned_class_count), batch_size)
        lambda_meter.update(grl_lambda, batch_size)

    return {
        "class_loss": class_losses.avg,
        "domain_loss": domain_losses.avg,
        "target_entropy_loss": entropy_losses.avg,
        "total_loss": total_losses.avg,
        "aligned_source_ratio": source_ratios.avg,
        "aligned_target_ratio": target_ratios.avg,
        "aligned_class_count": class_counts.avg,
        "grl_lambda": lambda_meter.avg,
    }


def class_conditioned_masks(
    source_labels: torch.Tensor,
    target_pseudo: torch.Tensor,
    target_confidence: torch.Tensor,
    confidence_threshold: float,
    min_aligned_targets: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    confident_target = target_confidence >= confidence_threshold
    source_classes = torch.unique(source_labels)
    target_classes = torch.unique(target_pseudo[confident_target]) if confident_target.any() else target_pseudo.new_empty(0)
    aligned_classes = source_classes[torch.isin(source_classes, target_classes)]

    if aligned_classes.numel() == 0 or confident_target.sum().item() < min_aligned_targets:
        source_mask = torch.ones_like(source_labels, dtype=torch.bool)
        target_mask = torch.ones_like(target_pseudo, dtype=torch.bool)
        return source_mask, target_mask, 0

    source_mask = torch.isin(source_labels, aligned_classes)
    target_mask = confident_target & torch.isin(target_pseudo, aligned_classes)
    return source_mask, target_mask, int(aligned_classes.numel())


def class_conditioned_domain_loss(
    model: ICDA,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    grl_lambda: float,
) -> torch.Tensor:
    selected_source = source_features[source_mask]
    selected_target = target_features[target_mask]
    if selected_source.numel() == 0 or selected_target.numel() == 0:
        return source_features.sum() * 0.0

    features = torch.cat([selected_source, selected_target], dim=0)
    domain_logits = model.domain_logits(features, grl_lambda)
    domain_labels = torch.cat(
        [
            torch.zeros(selected_source.size(0), dtype=torch.long, device=features.device),
            torch.ones(selected_target.size(0), dtype=torch.long, device=features.device),
        ],
        dim=0,
    )
    return F.cross_entropy(domain_logits, domain_labels)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: ICDA, loader, device) -> dict[str, float]:
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


def validate_args(args) -> None:
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    if args.min_aligned_targets < 1:
        raise ValueError("--min-aligned-targets must be at least 1.")
    for name in ("domain_loss_weight", "entropy_loss_weight", "grl_lambda"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"icda_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
