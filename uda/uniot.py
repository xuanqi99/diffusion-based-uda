"""UniOT-style unified optimal transport baseline for universal UDA."""

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


class UniOT(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, num_private_prototypes: int) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.private_prototypes = nn.Parameter(torch.randn(num_private_prototypes, feature_dim) * 0.02)

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
        return source_features, target_features, source_logits, target_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniOT baseline for UDA datasets.")
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
    parser.add_argument("--num-private-prototypes", type=int, default=4)
    parser.add_argument("--ot-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-common-loss-weight", type=float, default=0.5)
    parser.add_argument("--private-cluster-weight", type=float, default=0.1)
    parser.add_argument("--prototype-compactness-weight", type=float, default=0.1)
    parser.add_argument("--prediction-cost-weight", type=float, default=0.5)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--min-common-mass", type=float, default=0.2)
    parser.add_argument("--max-common-mass", type=float, default=0.95)
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
        raise ValueError("UniOT requires a target training loader.")

    device = torch.device(args.device)
    model = UniOT(args.arch, bundle.num_classes, args.pretrained, args.num_private_prototypes).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "ot_loss",
            "target_common_loss",
            "private_cluster_loss",
            "prototype_compactness_loss",
            "total_loss",
            "source_acc",
            "target_acc",
            "common_mass",
            "private_mass",
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
        f"Method: UniOT | private_prototypes: {args.num_private_prototypes} | model: {args.arch} "
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
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"ot={train_metrics['ot_loss']:.4f} "
            f"target_common={train_metrics['target_common_loss']:.4f} "
            f"common_mass={train_metrics['common_mass']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: UniOT, source_loader, target_loader, optimizer, scaler, device, num_classes: int, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    ot_losses = AverageMeter()
    target_common_losses = AverageMeter()
    private_cluster_losses = AverageMeter()
    compactness_losses = AverageMeter()
    total_losses = AverageMeter()
    common_masses = AverageMeter()
    private_masses = AverageMeter()
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
            prototypes = source_class_prototypes(source_features, source_labels, model.classifier.weight, num_classes)
            ot_metrics = uniot_losses(
                target_features=target_features,
                target_logits=target_logits,
                class_prototypes=prototypes,
                private_prototypes=model.private_prototypes,
                source_features=source_features,
                source_labels=source_labels,
                args=args,
            )
            loss = (
                class_loss
                + args.ot_loss_weight * ot_metrics["ot_loss"]
                + args.target_common_loss_weight * ot_metrics["target_common_loss"]
                + args.private_cluster_weight * ot_metrics["private_cluster_loss"]
                + args.prototype_compactness_weight * ot_metrics["prototype_compactness_loss"]
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        ot_losses.update(ot_metrics["ot_loss"].item(), batch_size)
        target_common_losses.update(ot_metrics["target_common_loss"].item(), batch_size)
        private_cluster_losses.update(ot_metrics["private_cluster_loss"].item(), batch_size)
        compactness_losses.update(ot_metrics["prototype_compactness_loss"].item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        common_masses.update(ot_metrics["common_mass"].item(), batch_size)
        private_masses.update(ot_metrics["private_mass"].item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "ot_loss": ot_losses.avg,
        "target_common_loss": target_common_losses.avg,
        "private_cluster_loss": private_cluster_losses.avg,
        "prototype_compactness_loss": compactness_losses.avg,
        "total_loss": total_losses.avg,
        "common_mass": common_masses.avg,
        "private_mass": private_masses.avg,
    }


def source_class_prototypes(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = classifier_weight.detach().clone()
    for class_id in range(num_classes):
        class_mask = source_labels == class_id
        if class_mask.any():
            prototypes[class_id] = source_features[class_mask].mean(dim=0)
    return prototypes


def uniot_losses(
    target_features: torch.Tensor,
    target_logits: torch.Tensor,
    class_prototypes: torch.Tensor,
    private_prototypes: torch.Tensor,
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    target_probabilities = F.softmax(target_logits, dim=1)
    class_cost = cosine_cost(target_features, class_prototypes)
    prediction_cost = -torch.log(target_probabilities.clamp_min(1e-8))
    common_cost = class_cost + args.prediction_cost_weight * prediction_cost
    private_cost = cosine_cost(target_features, private_prototypes)
    cost = torch.cat([common_cost, private_cost], dim=1)

    target_marginal = torch.full((target_features.size(0),), 1.0 / target_features.size(0), device=target_features.device)
    anchor_marginal, common_budget = adaptive_anchor_marginal(target_probabilities, private_prototypes.size(0), args)
    plan = sinkhorn(cost, target_marginal, anchor_marginal, args.sinkhorn_epsilon, args.sinkhorn_iters)

    common_plan = plan[:, : target_logits.size(1)]
    private_plan = plan[:, target_logits.size(1) :]
    ot_loss = (plan * cost).sum()
    target_common_loss = soft_cross_entropy_from_transport(target_logits, common_plan)
    private_cluster_loss = target_private_clustering_loss(private_plan)
    prototype_compactness_loss = source_compactness_loss(source_features, source_labels, class_prototypes)
    return {
        "ot_loss": ot_loss,
        "target_common_loss": target_common_loss,
        "private_cluster_loss": private_cluster_loss,
        "prototype_compactness_loss": prototype_compactness_loss,
        "common_mass": common_plan.sum().detach(),
        "private_mass": private_plan.sum().detach(),
        "common_budget": common_budget.detach(),
    }


def adaptive_anchor_marginal(target_probabilities: torch.Tensor, num_private_prototypes: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    class_mass = target_probabilities.mean(dim=0)
    class_mass = class_mass / class_mass.sum().clamp_min(1e-8)
    confidence = target_probabilities.max(dim=1).values.mean()
    common_budget = confidence.clamp(args.min_common_mass, args.max_common_mass)
    private_budget = 1.0 - common_budget
    common_marginal = common_budget * class_mass
    private_marginal = torch.ones(
        num_private_prototypes,
        dtype=target_probabilities.dtype,
        device=target_probabilities.device,
    ) * (private_budget / num_private_prototypes)
    return torch.cat([common_marginal, private_marginal], dim=0), common_budget


def sinkhorn(cost: torch.Tensor, row_marginal: torch.Tensor, col_marginal: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    cost = cost.float()
    log_k = -cost / epsilon
    log_u = torch.zeros_like(row_marginal)
    log_v = torch.zeros_like(col_marginal)
    log_row = torch.log(row_marginal.clamp_min(1e-8))
    log_col = torch.log(col_marginal.clamp_min(1e-8))
    for _ in range(iterations):
        log_u = log_row - torch.logsumexp(log_k + log_v.view(1, -1), dim=1)
        log_v = log_col - torch.logsumexp(log_k + log_u.view(-1, 1), dim=0)
    return torch.exp(log_k + log_u.view(-1, 1) + log_v.view(1, -1)).to(cost.dtype)


def cosine_cost(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    return 1.0 - features @ prototypes.t()


def soft_cross_entropy_from_transport(logits: torch.Tensor, common_plan: torch.Tensor) -> torch.Tensor:
    mass = common_plan.sum().clamp_min(1e-8)
    soft_targets = common_plan / common_plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
    row_weights = common_plan.sum(dim=1).detach()
    return -((soft_targets.detach() * F.log_softmax(logits, dim=1)).sum(dim=1) * row_weights).sum() / mass


def target_private_clustering_loss(private_plan: torch.Tensor) -> torch.Tensor:
    row_mass = private_plan.sum(dim=1, keepdim=True)
    if row_mass.sum().item() <= 1e-8:
        return private_plan.sum() * 0.0
    assignments = private_plan / row_mass.clamp_min(1e-8)
    local_entropy = -(assignments * assignments.clamp_min(1e-8).log()).sum(dim=1)
    weighted_local = (local_entropy * row_mass.squeeze(1)).sum() / row_mass.sum().clamp_min(1e-8)
    global_assignments = private_plan.sum(dim=0)
    global_assignments = global_assignments / global_assignments.sum().clamp_min(1e-8)
    global_entropy = -(global_assignments * global_assignments.clamp_min(1e-8).log()).sum()
    return weighted_local - global_entropy


def source_compactness_loss(source_features: torch.Tensor, source_labels: torch.Tensor, class_prototypes: torch.Tensor) -> torch.Tensor:
    selected = class_prototypes[source_labels]
    return cosine_cost(source_features, selected).diag().mean()


@torch.no_grad()
def evaluate(model: UniOT, loader, device) -> dict[str, float]:
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
    if args.num_private_prototypes <= 0:
        raise ValueError("--num-private-prototypes must be positive.")
    if args.sinkhorn_epsilon <= 0:
        raise ValueError("--sinkhorn-epsilon must be positive.")
    if args.sinkhorn_iters <= 0:
        raise ValueError("--sinkhorn-iters must be positive.")
    if not 0.0 <= args.min_common_mass <= args.max_common_mass <= 1.0:
        raise ValueError("--min-common-mass and --max-common-mass must satisfy 0 <= min <= max <= 1.")
    for name in (
        "ot_loss_weight",
        "target_common_loss_weight",
        "private_cluster_weight",
        "prototype_compactness_weight",
        "prediction_cost_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"uniot_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
