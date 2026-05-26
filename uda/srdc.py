"""Structurally Regularized Deep Clustering baseline for UDA."""

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


class SRDC(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, bottleneck_dim: int) -> None:
        super().__init__()
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.cluster_centers = nn.Parameter(torch.empty(num_classes, bottleneck_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.bottleneck(features)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(images))

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.encode(source_images)
        target_features = self.encode(target_images)
        source_logits = self.classify(source_features)
        target_logits = self.classify(target_features)
        return source_features, target_features, source_logits, target_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SRDC baseline for UDA datasets.")
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
    parser.add_argument("--target-cluster-weight", type=float, default=1.0)
    parser.add_argument("--target-feature-weight", type=float, default=1.0)
    parser.add_argument("--source-regularization-weight", type=float, default=1.0)
    parser.add_argument("--source-feature-weight", type=float, default=1.0)
    parser.add_argument("--source-soft-select", action="store_true", help="Weight source examples by similarity to target batch clusters.")
    parser.add_argument("--source-mix-weight", action="store_true", help="Anneal soft source weights from 1 to similarity weights.")
    parser.add_argument("--cluster-beta", type=float, default=1.0, help="Blend hard pseudo labels with auxiliary target distribution.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Student-t clustering degrees of freedom.")
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
        raise ValueError("SRDC requires a target training loader.")

    device = torch.device(args.device)
    model = SRDC(args.arch, bundle.num_classes, args.pretrained, args.bottleneck_dim).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "source_class_loss",
            "source_feature_loss",
            "target_class_cluster_loss",
            "target_feature_cluster_loss",
            "total_loss",
            "source_weight",
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
        f"Method: SRDC | bottleneck: {args.bottleneck_dim} | beta: {args.cluster_beta} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(
            model,
            bundle.source_train,
            bundle.target_train,
            optimizer,
            scaler,
            device,
            bundle.num_classes,
            epoch,
            args,
        )
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
                **train_metrics,
                "source_acc": source_acc,
                "target_acc": target_acc,
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"src_cls={train_metrics['source_class_loss']:.4f} "
            f"src_feat={train_metrics['source_feature_loss']:.4f} "
            f"tar_cls={train_metrics['target_class_cluster_loss']:.4f} "
            f"tar_feat={train_metrics['target_feature_cluster_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(
    model: SRDC,
    source_loader,
    target_loader,
    optimizer,
    scaler,
    device,
    num_classes: int,
    epoch: int,
    args,
) -> dict[str, float]:
    model.train()
    source_class_losses = AverageMeter()
    source_feature_losses = AverageMeter()
    target_class_losses = AverageMeter()
    target_feature_losses = AverageMeter()
    total_losses = AverageMeter()
    source_weight_meter = AverageMeter()
    target_iter = cycle(target_loader)
    lam = srdc_schedule(epoch, args)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            source_features, target_features, source_logits, target_logits = model(source_images, target_images)
            source_weights = source_selection_weights(
                source_features,
                source_labels,
                target_features,
                target_logits,
                num_classes,
                lam,
                args,
            )
            source_class_loss = weighted_cross_entropy(source_logits, source_labels, source_weights)
            source_assignments = student_t_assignments(source_features, model.cluster_centers, args.alpha)
            source_feature_loss = weighted_nll(source_assignments, source_labels, source_weights)

            target_class_loss = target_deep_cluster_loss(target_logits, args.cluster_beta)
            target_assignments = student_t_assignments(target_features, model.cluster_centers, args.alpha)
            target_feature_loss = target_distribution_loss(target_assignments, args.cluster_beta)
            source_loss = source_class_loss + args.source_feature_weight * source_feature_loss
            target_loss = args.target_cluster_weight * target_class_loss + args.target_feature_weight * target_feature_loss
            loss = args.source_regularization_weight * source_loss + lam * target_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        source_class_losses.update(source_class_loss.item(), batch_size)
        source_feature_losses.update(source_feature_loss.item(), batch_size)
        target_class_losses.update(target_class_loss.item(), batch_size)
        target_feature_losses.update(target_feature_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        source_weight_meter.update(source_weights.detach().mean().item(), batch_size)

    return {
        "source_class_loss": source_class_losses.avg,
        "source_feature_loss": source_feature_losses.avg,
        "target_class_cluster_loss": target_class_losses.avg,
        "target_feature_cluster_loss": target_feature_losses.avg,
        "total_loss": total_losses.avg,
        "source_weight": source_weight_meter.avg,
    }


def target_deep_cluster_loss(logits: torch.Tensor, beta: float) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    auxiliary = auxiliary_target_distribution(probabilities)
    hard = F.one_hot(probabilities.argmax(dim=1), num_classes=probabilities.size(1)).to(probabilities.dtype)
    target = ((1.0 - beta) * hard + beta * auxiliary).detach()
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def target_distribution_loss(probabilities: torch.Tensor, beta: float) -> torch.Tensor:
    auxiliary = auxiliary_target_distribution(probabilities)
    hard = F.one_hot(probabilities.argmax(dim=1), num_classes=probabilities.size(1)).to(probabilities.dtype)
    target = ((1.0 - beta) * hard + beta * auxiliary).detach()
    return -(target * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()


def auxiliary_target_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    class_balance = probabilities.sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    target = probabilities / class_balance
    return target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)


def student_t_assignments(features: torch.Tensor, centers: torch.Tensor, alpha: float) -> torch.Tensor:
    distances = torch.cdist(features.float(), centers.float(), p=2).pow(2)
    logits = (1.0 + distances / alpha).pow(-0.5 * (alpha + 1.0))
    return logits / logits.sum(dim=1, keepdim=True).clamp_min(1e-8)


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (weights * losses).mean()


def weighted_nll(probabilities: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = -probabilities.clamp_min(1e-8).log().gather(1, labels.unsqueeze(1)).squeeze(1)
    return (weights * losses).mean()


def source_selection_weights(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
    lam: float,
    args,
) -> torch.Tensor:
    if not args.source_soft_select:
        return source_features.new_ones(source_features.size(0))

    target_labels = target_logits.detach().argmax(dim=1)
    centers = class_centers(target_features.detach(), target_labels, num_classes)
    source_norm = F.normalize(source_features.float(), dim=1)
    center_norm = F.normalize(centers.float(), dim=1)
    similarities = (source_norm * center_norm[source_labels]).sum(dim=1).clamp(min=-1.0, max=1.0)
    weights = 0.5 * (similarities + 1.0)
    if args.source_mix_weight:
        weights = lam * weights + (1.0 - lam)
    return weights.detach()


def class_centers(features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    centers = features.new_zeros(num_classes, features.size(1))
    global_center = features.mean(dim=0)
    for class_id in range(num_classes):
        mask = labels == class_id
        centers[class_id] = features[mask].mean(dim=0) if mask.any() else global_center
    return centers


@torch.no_grad()
def evaluate(model: SRDC, loader, device) -> dict[str, float]:
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


def srdc_schedule(epoch: int, args) -> float:
    progress = epoch / max(args.epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def validate_args(args) -> None:
    if args.bottleneck_dim <= 0:
        raise ValueError("--bottleneck-dim must be positive.")
    if not 0.0 <= args.cluster_beta <= 1.0:
        raise ValueError("--cluster-beta must be in [0, 1].")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")
    for name in (
        "target_cluster_weight",
        "target_feature_weight",
        "source_regularization_weight",
        "source_feature_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"srdc_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
