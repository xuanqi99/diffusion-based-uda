"""Probability-polarized optimal transport baseline for UDA."""

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


class PPOT(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, _ = build_feature_model(arch, num_classes, pretrained)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(images))

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.encode(source_images)
        target_features = self.encode(target_images)
        source_logits = self.classify(source_features)
        target_logits = self.classify(target_features)
        return source_logits, target_logits, source_features, target_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPOT baseline for UDA datasets.")
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
    parser.add_argument("--source-pretrain-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--ot-loss-weight", type=float, default=1.0)
    parser.add_argument("--transported-loss-weight", type=float, default=0.5)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.01)
    parser.add_argument("--semantic-cost-weight", type=float, default=0.5)
    parser.add_argument("--polarization-weight", type=float, default=0.2)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--polarization-iters", type=int, default=2)
    parser.add_argument("--positive-threshold-scale", type=float, default=1.5)
    parser.add_argument("--negative-threshold-scale", type=float, default=0.5)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.0)
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
        raise ValueError("PPOT requires a target training loader.")

    device = torch.device(args.device)
    model = PPOT(args.arch, bundle.num_classes, args.pretrained).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "ot_loss",
            "transported_loss",
            "entropy_loss",
            "polarization_loss",
            "total_loss",
            "source_acc",
            "target_acc",
            "intra_mass",
            "inter_mass",
            "lr",
            "elapsed_sec",
        ],
    )

    if args.source_pretrain_epochs > 0:
        for epoch in range(1, args.source_pretrain_epochs + 1):
            loss = train_source_epoch(model, bundle.source_train, optimizer, scaler, device, args)
            print(f"Pretrain {epoch:03d}/{args.source_pretrain_epochs:03d} class_loss={loss:.4f}")

    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(
        f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) "
        f"| target: {args.target} ({bundle.target_train_size})"
    )
    print(
        f"Method: PPOT | epsilon: {args.sinkhorn_epsilon:g} | polarization: {args.polarization_weight:g} "
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
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"ot={train_metrics['ot_loss']:.4f} "
            f"transported={train_metrics['transported_loss']:.4f} "
            f"intra={train_metrics['intra_mass']:.4f} inter={train_metrics['inter_mass']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: PPOT, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    ot_losses = AverageMeter()
    transported_losses = AverageMeter()
    entropy_losses = AverageMeter()
    polarization_losses = AverageMeter()
    total_losses = AverageMeter()
    intra_masses = AverageMeter()
    inter_masses = AverageMeter()
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
            target_entropy_loss = entropy_loss(target_logits)
            ppot_metrics = ppot_losses(source_features, target_features, target_logits, source_labels, model, args)
            loss = (
                class_loss
                + args.ot_loss_weight * ppot_metrics["ot_loss"]
                + args.transported_loss_weight * ppot_metrics["transported_loss"]
                + args.entropy_loss_weight * target_entropy_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        ot_losses.update(ppot_metrics["ot_loss"].item(), batch_size)
        transported_losses.update(ppot_metrics["transported_loss"].item(), batch_size)
        entropy_losses.update(target_entropy_loss.item(), batch_size)
        polarization_losses.update(ppot_metrics["polarization_loss"].item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        intra_masses.update(ppot_metrics["intra_mass"].item(), batch_size)
        inter_masses.update(ppot_metrics["inter_mass"].item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "ot_loss": ot_losses.avg,
        "transported_loss": transported_losses.avg,
        "entropy_loss": entropy_losses.avg,
        "polarization_loss": polarization_losses.avg,
        "total_loss": total_losses.avg,
        "intra_mass": intra_masses.avg,
        "inter_mass": inter_masses.avg,
    }


def train_source_epoch(model: PPOT, loader, optimizer, scaler, device, args) -> float:
    model.train()
    losses = AverageMeter()
    for step, (images, labels) in enumerate(loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            logits = model.predict(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))
    return losses.avg


def ppot_losses(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    source_labels: torch.Tensor,
    model: PPOT,
    args,
) -> dict[str, torch.Tensor]:
    target_probabilities = F.softmax(target_logits, dim=1)
    target_confidence, target_pseudo = target_probabilities.detach().max(dim=1)
    confident = target_confidence >= args.target_confidence_threshold
    same_class = (source_labels.view(-1, 1) == target_pseudo.view(1, -1)) & confident.view(1, -1)

    feature_cost = cosine_cost(source_features, target_features)
    semantic_cost = -torch.log(target_probabilities[:, source_labels].t().clamp_min(1e-8))
    cost = feature_cost + args.semantic_cost_weight * semantic_cost

    plan, lower, upper = probability_polarized_plan(cost, same_class, args)
    polarization_loss = polarization_regularizer(plan, same_class, lower, upper)
    ot_loss = (plan * cost).sum() + args.polarization_weight * polarization_loss

    transported_features = barycentric_map(plan, target_features)
    transported_logits = model.classify(transported_features)
    transported_loss = F.cross_entropy(transported_logits, source_labels)

    intra_mass = plan.masked_select(same_class).sum() if same_class.any() else plan.sum() * 0.0
    inter_mass = plan.masked_select(~same_class).sum()
    return {
        "ot_loss": ot_loss,
        "transported_loss": transported_loss,
        "polarization_loss": polarization_loss,
        "intra_mass": intra_mass.detach(),
        "inter_mass": inter_mass.detach(),
    }


def probability_polarized_plan(cost: torch.Tensor, same_class: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = cost.shape
    row_marginal = torch.full((rows,), 1.0 / rows, dtype=cost.dtype, device=cost.device)
    col_marginal = torch.full((cols,), 1.0 / cols, dtype=cost.dtype, device=cost.device)
    base_mass = 1.0 / (rows * cols)
    lower = torch.as_tensor(args.negative_threshold_scale * base_mass, dtype=cost.dtype, device=cost.device)
    upper = torch.as_tensor(args.positive_threshold_scale * base_mass, dtype=cost.dtype, device=cost.device)

    adjusted_cost = cost
    plan = sinkhorn(adjusted_cost, row_marginal, col_marginal, args.sinkhorn_epsilon, args.sinkhorn_iters)
    for _ in range(args.polarization_iters):
        force = polarization_cost_force(plan.detach(), same_class, lower, upper)
        adjusted_cost = cost + args.polarization_weight * force
        plan = sinkhorn(adjusted_cost, row_marginal, col_marginal, args.sinkhorn_epsilon, args.sinkhorn_iters)
    return plan, lower, upper


def polarization_cost_force(plan: torch.Tensor, same_class: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    same_force = -torch.relu(upper - plan) / upper.clamp_min(1e-8)
    different_force = torch.relu(plan - lower) / lower.clamp_min(1e-8)
    return torch.where(same_class, same_force, different_force)


def polarization_regularizer(plan: torch.Tensor, same_class: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    same_loss = torch.relu(upper - plan).masked_select(same_class).mean() if same_class.any() else plan.sum() * 0.0
    different_mask = ~same_class
    different_loss = torch.relu(plan - lower).masked_select(different_mask).mean() if different_mask.any() else plan.sum() * 0.0
    return same_loss + different_loss


def sinkhorn(cost: torch.Tensor, row_marginal: torch.Tensor, col_marginal: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    log_k = -cost.float() / epsilon
    log_u = torch.zeros_like(row_marginal, dtype=torch.float32)
    log_v = torch.zeros_like(col_marginal, dtype=torch.float32)
    log_row = torch.log(row_marginal.float().clamp_min(1e-8))
    log_col = torch.log(col_marginal.float().clamp_min(1e-8))
    for _ in range(iterations):
        log_u = log_row - torch.logsumexp(log_k + log_v.view(1, -1), dim=1)
        log_v = log_col - torch.logsumexp(log_k + log_u.view(-1, 1), dim=0)
    return torch.exp(log_k + log_u.view(-1, 1) + log_v.view(1, -1)).to(cost.dtype)


def barycentric_map(plan: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    row_mass = plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (plan @ target_features) / row_mass


def cosine_cost(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    source_features = F.normalize(source_features, dim=1)
    target_features = F.normalize(target_features, dim=1)
    return 1.0 - source_features @ target_features.t()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: PPOT, loader, device) -> dict[str, float]:
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
    if args.sinkhorn_epsilon <= 0:
        raise ValueError("--sinkhorn-epsilon must be positive.")
    if args.sinkhorn_iters <= 0:
        raise ValueError("--sinkhorn-iters must be positive.")
    if args.polarization_iters < 0:
        raise ValueError("--polarization-iters must be non-negative.")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    if args.positive_threshold_scale <= args.negative_threshold_scale:
        raise ValueError("--positive-threshold-scale must be greater than --negative-threshold-scale.")
    for name in (
        "ot_loss_weight",
        "transported_loss_weight",
        "entropy_loss_weight",
        "semantic_cost_weight",
        "polarization_weight",
        "negative_threshold_scale",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"ppot_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
