"""ACTIR-style adaptive-invariant representation baseline for UDA."""

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
    from erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import build_data, normalize_dataset_name
    from .erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class ACTIR(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        layers: list[nn.Module] = [
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.projector = nn.Sequential(*layers)
        self.invariant_classifier = nn.Linear(bottleneck_dim, num_classes)
        self.adaptive_classifier = nn.Linear(bottleneck_dim, num_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.feature_extractor(images))

    def logits_from_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        invariant_logits = self.invariant_classifier(features)
        adaptive_logits = self.adaptive_classifier(features)
        return invariant_logits, adaptive_logits, invariant_logits + adaptive_logits

    def predict(self, images: torch.Tensor, use_adaptive: bool = True) -> torch.Tensor:
        features = self.encode(images)
        invariant_logits, adaptive_logits, combined_logits = self.logits_from_features(features)
        return combined_logits if use_adaptive else invariant_logits

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.encode(source_images)
        target_features = self.encode(target_images)
        source_logits = self.logits_from_features(source_features)
        target_logits = self.logits_from_features(target_features)
        return source_features, target_features, source_logits, target_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACTIR baseline for UDA datasets.")
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
    parser.add_argument("--bottleneck-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9, help="Blend source combined-head and invariant-head losses.")
    parser.add_argument("--decorrelation-weight", type=float, default=1.0)
    parser.add_argument("--gradient-penalty-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-decorrelation-weight", type=float, default=0.5)
    parser.add_argument("--target-entropy-weight", type=float, default=0.01)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.8)
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
        raise ValueError("ACTIR requires a target training loader.")

    device = torch.device(args.device)
    model = ACTIR(args.arch, bundle.num_classes, args.pretrained, args.bottleneck_dim, args.dropout).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "source_invariant_loss",
            "source_combined_loss",
            "decorrelation_loss",
            "gradient_penalty",
            "target_loss",
            "target_decorrelation_loss",
            "target_entropy_loss",
            "selected_ratio",
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
        f"Method: ACTIR | bottleneck: {args.bottleneck_dim} | gamma: {args.gamma} "
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
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"src_inv={train_metrics['source_invariant_loss']:.4f} "
            f"src_all={train_metrics['source_combined_loss']:.4f} "
            f"decor={train_metrics['decorrelation_loss']:.4f} "
            f"gp={train_metrics['gradient_penalty']:.4f} "
            f"tar={train_metrics['target_loss']:.4f} "
            f"selected={train_metrics['selected_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: ACTIR, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    source_invariant_losses = AverageMeter()
    source_combined_losses = AverageMeter()
    decorrelation_losses = AverageMeter()
    gradient_penalties = AverageMeter()
    target_losses = AverageMeter()
    target_decorrelation_losses = AverageMeter()
    target_entropy_losses = AverageMeter()
    selected_ratios = AverageMeter()
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
            _, _, source_logits, target_logits = model(source_images, target_images)
            source_invariant, source_adaptive, source_combined = source_logits
            target_invariant, target_adaptive, target_combined = target_logits

            source_invariant_loss = F.cross_entropy(source_invariant, source_labels)
            source_combined_loss = F.cross_entropy(source_combined, source_labels)
            decorrelation_loss = conditional_decorrelation_loss(source_invariant, source_adaptive, source_labels)
            constraint_loss = source_combined_loss + args.decorrelation_weight * decorrelation_loss
            gradient_penalty = adaptive_gradient_penalty(constraint_loss, model)

            pseudo_labels, target_mask = target_pseudo_labels(target_invariant.detach(), args.target_confidence_threshold)
            if target_mask.any():
                target_loss = F.cross_entropy(target_combined[target_mask], pseudo_labels[target_mask])
                target_decorrelation_loss = conditional_decorrelation_loss(
                    target_invariant[target_mask],
                    target_adaptive[target_mask],
                    pseudo_labels[target_mask],
                )
            else:
                target_loss = target_combined.sum() * 0.0
                target_decorrelation_loss = target_combined.sum() * 0.0
            target_entropy_loss = entropy_loss(target_combined)

            source_loss = args.gamma * source_combined_loss + (1.0 - args.gamma) * source_invariant_loss
            loss = (
                source_loss
                + args.decorrelation_weight * decorrelation_loss
                + args.gradient_penalty_weight * gradient_penalty
                + args.target_loss_weight * target_loss
                + args.target_decorrelation_weight * target_decorrelation_loss
                + args.target_entropy_weight * target_entropy_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        source_invariant_losses.update(source_invariant_loss.item(), batch_size)
        source_combined_losses.update(source_combined_loss.item(), batch_size)
        decorrelation_losses.update(decorrelation_loss.item(), batch_size)
        gradient_penalties.update(gradient_penalty.item(), batch_size)
        target_losses.update(target_loss.item(), batch_size)
        target_decorrelation_losses.update(target_decorrelation_loss.item(), batch_size)
        target_entropy_losses.update(target_entropy_loss.item(), batch_size)
        selected_ratios.update(target_mask.float().mean().item(), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "source_invariant_loss": source_invariant_losses.avg,
        "source_combined_loss": source_combined_losses.avg,
        "decorrelation_loss": decorrelation_losses.avg,
        "gradient_penalty": gradient_penalties.avg,
        "target_loss": target_losses.avg,
        "target_decorrelation_loss": target_decorrelation_losses.avg,
        "target_entropy_loss": target_entropy_losses.avg,
        "selected_ratio": selected_ratios.avg,
        "total_loss": total_losses.avg,
    }


def conditional_decorrelation_loss(invariant_logits: torch.Tensor, adaptive_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    zero = invariant_logits.sum() * 0.0
    terms = []
    for class_id in labels.unique().tolist():
        mask = labels == class_id
        if mask.sum().item() < 2:
            continue
        invariant = invariant_logits[mask] - invariant_logits[mask].mean(dim=0, keepdim=True)
        adaptive = adaptive_logits[mask] - adaptive_logits[mask].mean(dim=0, keepdim=True)
        covariance = invariant.t() @ adaptive / max(mask.sum().item() - 1, 1)
        terms.append(covariance.abs().mean())
    if not terms:
        return zero
    return torch.stack(terms).mean()


def adaptive_gradient_penalty(constraint_loss: torch.Tensor, model: ACTIR) -> torch.Tensor:
    params = [param for param in model.adaptive_classifier.parameters() if param.requires_grad]
    grads = torch.autograd.grad(constraint_loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    penalties = [grad.pow(2).mean() for grad in grads if grad is not None]
    if not penalties:
        return constraint_loss * 0.0
    return torch.stack(penalties).mean()


@torch.no_grad()
def target_pseudo_labels(logits: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=1)
    confidence, labels = probabilities.max(dim=1)
    return labels, confidence >= threshold


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: ACTIR, loader, device) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict(images, use_adaptive=True)
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
    if args.bottleneck_dim <= 0:
        raise ValueError("--bottleneck-dim must be positive.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be in [0, 1].")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    if args.dropout < 0:
        raise ValueError("--dropout must be non-negative.")
    for name in (
        "decorrelation_weight",
        "gradient_penalty_weight",
        "target_loss_weight",
        "target_decorrelation_weight",
        "target_entropy_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"actir_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
