"""Contrastive Adaptation Network baseline for unsupervised domain adaptation."""

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


class CAN(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, _ = build_feature_model(arch, num_classes, pretrained)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify(self.extract(images))

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.extract(source_images)
        target_features = self.extract(target_images)
        return source_features, target_features, self.classify(source_features), self.classify(target_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAN baseline for UDA datasets.")
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
    parser.add_argument("--can-loss-weight", type=float, default=1.0)
    parser.add_argument("--inter-class-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-label-mode", choices=("cluster", "prediction"), default="cluster")
    parser.add_argument("--cluster-steps", type=int, default=3)
    parser.add_argument("--target-distance-threshold", type=float, default=2.0)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-samples-per-class", type=int, default=1)
    parser.add_argument("--kernel-sigma", type=float, default=None, help="Base RBF bandwidth. Uses batch estimate by default.")
    parser.add_argument("--kernel-mul", type=float, default=2.0, help="Bandwidth multiplier for multi-kernel CDD.")
    parser.add_argument("--kernel-num", type=int, default=5, help="Number of RBF kernels for CDD.")
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
        raise ValueError("CAN requires a target training loader.")

    device = torch.device(args.device)
    model = CAN(args.arch, bundle.num_classes, args.pretrained).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "can_loss",
            "intra_loss",
            "inter_loss",
            "selected_ratio",
            "valid_classes",
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
        f"Method: CAN | pseudo_labels: {args.pseudo_label_mode} | model: {args.arch} "
        f"| classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, bundle.num_classes, args)
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
                "can_loss": train_metrics["can_loss"],
                "intra_loss": train_metrics["intra_loss"],
                "inter_loss": train_metrics["inter_loss"],
                "selected_ratio": train_metrics["selected_ratio"],
                "valid_classes": train_metrics["valid_classes"],
                "total_loss": train_metrics["total_loss"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} class_loss={train_metrics['class_loss']:.4f} "
            f"can_loss={train_metrics['can_loss']:.4f} intra={train_metrics['intra_loss']:.4f} "
            f"inter={train_metrics['inter_loss']:.4f} selected={train_metrics['selected_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: CAN, source_loader, target_loader, optimizer, scaler, device, num_classes: int, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    can_losses = AverageMeter()
    intra_losses = AverageMeter()
    inter_losses = AverageMeter()
    selected_ratios = AverageMeter()
    valid_class_meter = AverageMeter()
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
            source_features, target_features, source_logits, target_logits = model(source_images, target_images)
            class_loss = F.cross_entropy(source_logits, source_labels)
            target_labels, target_mask = assign_target_labels(
                source_features.detach(),
                source_labels,
                target_features.detach(),
                target_logits.detach(),
                num_classes,
                args,
            )
            can_loss, intra_loss, inter_loss, valid_classes = compute_cdd_loss(
                source_features,
                source_labels,
                target_features,
                target_labels,
                target_mask,
                args,
            )
            loss = class_loss + args.can_loss_weight * can_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        selected_ratio = target_mask.float().mean().item() if target_mask.numel() else 0.0
        class_losses.update(class_loss.item(), batch_size)
        can_losses.update(can_loss.item(), batch_size)
        intra_losses.update(intra_loss.item(), batch_size)
        inter_losses.update(inter_loss.item(), batch_size)
        selected_ratios.update(selected_ratio, batch_size)
        valid_class_meter.update(float(valid_classes), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "can_loss": can_losses.avg,
        "intra_loss": intra_losses.avg,
        "inter_loss": inter_losses.avg,
        "selected_ratio": selected_ratios.avg,
        "valid_classes": valid_class_meter.avg,
        "total_loss": total_losses.avg,
    }


@torch.no_grad()
def assign_target_labels(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    num_classes: int,
    args,
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.pseudo_label_mode == "prediction":
        probabilities = F.softmax(target_logits, dim=1)
        confidence, labels = probabilities.max(dim=1)
        mask = confidence >= args.target_confidence_threshold
        return labels, filter_small_classes(labels, mask, args.min_samples_per_class)

    source_norm = F.normalize(source_features.float(), dim=1)
    target_norm = F.normalize(target_features.float(), dim=1)
    centers = initialize_centers(source_norm, source_labels, target_norm, target_logits, num_classes)
    source_classes = source_labels.unique().long()
    for _ in range(args.cluster_steps):
        label_indices = (target_norm @ centers[source_classes].t()).argmax(dim=1)
        labels = source_classes[label_indices]
        for class_id in labels.unique().tolist():
            class_mask = labels == class_id
            centers[class_id] = F.normalize(target_norm[class_mask].mean(dim=0), dim=0)

    similarities = target_norm @ centers[source_classes].t()
    max_similarity, label_indices = similarities.max(dim=1)
    labels = source_classes[label_indices]
    distances = 1.0 - max_similarity
    mask = distances <= args.target_distance_threshold
    return labels, filter_small_classes(labels, mask, args.min_samples_per_class)


def initialize_centers(source_features, source_labels, target_features, target_logits, num_classes: int) -> torch.Tensor:
    feature_dim = source_features.size(1)
    centers = source_features.new_zeros(num_classes, feature_dim)
    filled = torch.zeros(num_classes, dtype=torch.bool, device=source_features.device)

    for class_id in source_labels.unique().tolist():
        class_mask = source_labels == class_id
        centers[class_id] = F.normalize(source_features[class_mask].mean(dim=0), dim=0)
        filled[class_id] = True

    target_predictions = target_logits.argmax(dim=1)
    for class_id in target_predictions.unique().tolist():
        if filled[class_id]:
            continue
        class_mask = target_predictions == class_id
        centers[class_id] = F.normalize(target_features[class_mask].mean(dim=0), dim=0)
        filled[class_id] = True

    if (~filled).any():
        fallback = target_features if target_features.size(0) > 0 else source_features
        repeats = math.ceil(num_classes / max(fallback.size(0), 1))
        fallback = fallback.repeat(repeats, 1)[:num_classes]
        centers[~filled] = fallback[~filled]
        centers = F.normalize(centers, dim=1)
    return centers


def filter_small_classes(labels: torch.Tensor, mask: torch.Tensor, min_samples: int) -> torch.Tensor:
    if min_samples <= 1:
        return mask
    filtered = mask.clone()
    for class_id in labels[mask].unique().tolist():
        class_mask = mask & (labels == class_id)
        if class_mask.sum().item() < min_samples:
            filtered[class_mask] = False
    return filtered


def compute_cdd_loss(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    target_mask: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    zero = source_features.sum() * 0.0
    valid_classes: list[int] = []
    source_by_class: dict[int, torch.Tensor] = {}
    target_by_class: dict[int, torch.Tensor] = {}

    for class_id in torch.cat([source_labels.detach(), target_labels.detach()]).unique().tolist():
        source_mask = source_labels == class_id
        target_class_mask = target_mask & (target_labels == class_id)
        if source_mask.any() and target_class_mask.any():
            valid_classes.append(int(class_id))
            source_by_class[int(class_id)] = source_features[source_mask]
            target_by_class[int(class_id)] = target_features[target_class_mask]

    if not valid_classes:
        return zero, zero.detach(), zero.detach(), 0

    intra_terms = [mmd_pair(source_by_class[class_id], target_by_class[class_id], args) for class_id in valid_classes]
    inter_terms = [
        mmd_pair(source_by_class[source_class], target_by_class[target_class], args)
        for source_class in valid_classes
        for target_class in valid_classes
        if source_class != target_class
    ]
    intra_loss = torch.stack(intra_terms).mean()
    inter_loss = torch.stack(inter_terms).mean() if inter_terms else zero
    can_loss = intra_loss - args.inter_class_weight * inter_loss
    return can_loss, intra_loss.detach(), inter_loss.detach(), len(valid_classes)


def mmd_pair(source: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    source = source.flatten(1).float()
    target = target.flatten(1).float()
    source_count = source.size(0)
    kernels = gaussian_kernel_matrix(torch.cat([source, target], dim=0), args)
    source_source = kernels[:source_count, :source_count].mean()
    target_target = kernels[source_count:, source_count:].mean()
    source_target = kernels[:source_count, source_count:].mean()
    return source_source + target_target - 2.0 * source_target


def gaussian_kernel_matrix(features: torch.Tensor, args) -> torch.Tensor:
    diff = features[:, None, :] - features[None, :, :]
    distances = diff.pow(2).sum(dim=2)
    if args.kernel_sigma is None:
        bandwidth = distances.detach().mean().clamp_min(1e-6)
    else:
        bandwidth = features.new_tensor(args.kernel_sigma).clamp_min(1e-6)
    bandwidth = bandwidth / (args.kernel_mul ** (args.kernel_num // 2))
    kernels = [torch.exp(-distances / (bandwidth * (args.kernel_mul ** i))) for i in range(args.kernel_num)]
    return sum(kernels)


@torch.no_grad()
def evaluate(model: CAN, loader, device) -> dict[str, float]:
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
    if args.can_loss_weight < 0:
        raise ValueError("--can-loss-weight must be non-negative.")
    if args.inter_class_weight < 0:
        raise ValueError("--inter-class-weight must be non-negative.")
    if args.cluster_steps < 0:
        raise ValueError("--cluster-steps must be non-negative.")
    if args.kernel_num <= 0 or args.kernel_mul <= 0:
        raise ValueError("--kernel-num and --kernel-mul must be positive.")
    if args.kernel_sigma is not None and args.kernel_sigma <= 0:
        raise ValueError("--kernel-sigma must be positive when set.")
    if args.target_distance_threshold < 0 or args.target_confidence_threshold < 0:
        raise ValueError("Target filtering thresholds must be non-negative.")
    if args.min_samples_per_class < 1:
        raise ValueError("--min-samples-per-class must be at least 1.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"can_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
