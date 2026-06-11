"""TCRL-style triplet contrastive representation learning baseline for UDA."""

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


class TCRL(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool,
        embedding_dim: int,
        num_parts: int,
    ) -> None:
        super().__init__()
        self.num_parts = num_parts
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.global_projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.part_projector = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim * num_parts),
            nn.BatchNorm1d(embedding_dim * num_parts),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.cluster_prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images)
        global_features = F.normalize(self.global_projector(features), dim=1)
        part_features = self.part_projector(features).view(images.size(0), self.num_parts, -1)
        part_features = F.normalize(part_features, dim=2)
        return global_features, part_features

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        global_features, _ = self.encode(images)
        return self.classifier(global_features)

    def prototype_logits(self, embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
        prototypes = F.normalize(self.cluster_prototypes, dim=1)
        return embeddings @ prototypes.t() / temperature

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, temperature: float):
        source_global, source_parts = self.encode(source_images)
        target_global, target_parts = self.encode(target_images)
        source_logits = self.classifier(source_global)
        target_logits = self.classifier(target_global)
        source_proto_logits = self.prototype_logits(source_global, temperature)
        target_proto_logits = self.prototype_logits(target_global, temperature)
        source_part_logits = self.prototype_logits(source_parts.flatten(0, 1), temperature).view(source_images.size(0), self.num_parts, -1)
        target_part_logits = self.prototype_logits(target_parts.flatten(0, 1), temperature).view(target_images.size(0), self.num_parts, -1)
        return {
            "source_global": source_global,
            "target_global": target_global,
            "source_parts": source_parts,
            "target_parts": target_parts,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "source_proto_logits": source_proto_logits,
            "target_proto_logits": target_proto_logits,
            "source_part_logits": source_part_logits,
            "target_part_logits": target_part_logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCRL baseline for UDA datasets.")
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
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-parts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--prototype-loss-weight", type=float, default=1.0)
    parser.add_argument("--part-prototype-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-prototype-loss-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.2)
    parser.add_argument("--triplet-loss-weight", type=float, default=0.2)
    parser.add_argument("--part-global-loss-weight", type=float, default=0.1)
    parser.add_argument("--prototype-separation-weight", type=float, default=0.1)
    parser.add_argument("--target-entropy-weight", type=float, default=0.01)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.7)
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
        raise ValueError("TCRL requires a target training loader.")

    device = torch.device(args.device)
    model = TCRL(args.arch, bundle.num_classes, args.pretrained, args.embedding_dim, args.num_parts).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "prototype_loss",
            "part_prototype_loss",
            "target_prototype_loss",
            "contrastive_loss",
            "triplet_loss",
            "part_global_loss",
            "prototype_separation_loss",
            "target_entropy_loss",
            "selected_target_ratio",
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
        f"Method: TCRL | embedding_dim: {args.embedding_dim} | parts: {args.num_parts} "
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
            f"proto={train_metrics['prototype_loss']:.4f} "
            f"triplet={train_metrics['triplet_loss']:.4f} "
            f"contrast={train_metrics['contrastive_loss']:.4f} "
            f"selected={train_metrics['selected_target_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: TCRL, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    meters = {name: AverageMeter() for name in (
        "class_loss",
        "prototype_loss",
        "part_prototype_loss",
        "target_prototype_loss",
        "contrastive_loss",
        "triplet_loss",
        "part_global_loss",
        "prototype_separation_loss",
        "target_entropy_loss",
        "selected_target_ratio",
        "total_loss",
    )}
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
            output = model(source_images, target_images, args.temperature)
            class_loss = F.cross_entropy(output["source_logits"], source_labels)
            prototype_loss = F.cross_entropy(output["source_proto_logits"], source_labels)
            part_prototype_loss = part_proxy_loss(output["source_part_logits"], source_labels)
            target_pseudo, target_mask = target_pseudo_labels(output["target_logits"].detach(), args.target_confidence_threshold)
            if target_mask.any():
                target_prototype_loss = F.cross_entropy(output["target_proto_logits"][target_mask], target_pseudo[target_mask])
                target_part_loss = part_proxy_loss(output["target_part_logits"][target_mask], target_pseudo[target_mask])
                target_prototype_loss = 0.5 * (target_prototype_loss + target_part_loss)
            else:
                target_prototype_loss = output["target_logits"].sum() * 0.0

            features, labels = labeled_contrastive_batch(output, source_labels, target_pseudo, target_mask)
            contrastive_loss = supervised_contrastive_loss(features, labels, args.temperature)
            triplet_loss = batch_hard_triplet_loss(features, labels, args.triplet_margin)
            part_global_loss = part_global_consistency(output["source_global"], output["source_parts"], output["target_global"], output["target_parts"])
            prototype_separation_loss = prototype_separation(model.cluster_prototypes)
            target_entropy_loss = entropy_loss(output["target_logits"])

            loss = (
                class_loss
                + args.prototype_loss_weight * prototype_loss
                + args.part_prototype_loss_weight * part_prototype_loss
                + args.target_prototype_loss_weight * target_prototype_loss
                + args.contrastive_loss_weight * contrastive_loss
                + args.triplet_loss_weight * triplet_loss
                + args.part_global_loss_weight * part_global_loss
                + args.prototype_separation_weight * prototype_separation_loss
                + args.target_entropy_weight * target_entropy_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        values = {
            "class_loss": class_loss,
            "prototype_loss": prototype_loss,
            "part_prototype_loss": part_prototype_loss,
            "target_prototype_loss": target_prototype_loss,
            "contrastive_loss": contrastive_loss,
            "triplet_loss": triplet_loss,
            "part_global_loss": part_global_loss,
            "prototype_separation_loss": prototype_separation_loss,
            "target_entropy_loss": target_entropy_loss,
            "selected_target_ratio": target_mask.float().mean(),
            "total_loss": loss,
        }
        for name, value in values.items():
            meters[name].update(value.item(), batch_size)

    return {name: meter.avg for name, meter in meters.items()}


def part_proxy_loss(part_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    repeated_labels = labels.view(-1, 1).expand(-1, part_logits.size(1)).reshape(-1)
    return F.cross_entropy(part_logits.reshape(-1, part_logits.size(-1)), repeated_labels)


@torch.no_grad()
def target_pseudo_labels(logits: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=1)
    confidence, labels = probabilities.max(dim=1)
    return labels, confidence >= threshold


def labeled_contrastive_batch(output: dict[str, torch.Tensor], source_labels: torch.Tensor, target_pseudo: torch.Tensor, target_mask: torch.Tensor):
    if target_mask.any():
        features = torch.cat([output["source_global"], output["target_global"][target_mask]], dim=0)
        labels = torch.cat([source_labels, target_pseudo[target_mask]], dim=0)
    else:
        features = output["source_global"]
        labels = source_labels
    return features, labels


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if features.size(0) < 2:
        return features.sum() * 0.0
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positive_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    mean_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob[valid].mean()


def batch_hard_triplet_loss(features: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    if features.size(0) < 2:
        return features.sum() * 0.0
    distances = torch.cdist(features, features, p=2)
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    positive_mask = same & ~eye
    negative_mask = ~same
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    hardest_positive = distances.masked_fill(~positive_mask, -1.0).max(dim=1).values
    hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def part_global_consistency(
    source_global: torch.Tensor,
    source_parts: torch.Tensor,
    target_global: torch.Tensor,
    target_parts: torch.Tensor,
) -> torch.Tensor:
    source_part_mean = F.normalize(source_parts.mean(dim=1), dim=1)
    target_part_mean = F.normalize(target_parts.mean(dim=1), dim=1)
    return 0.5 * (F.mse_loss(source_part_mean, source_global) + F.mse_loss(target_part_mean, target_global))


def prototype_separation(prototypes: torch.Tensor) -> torch.Tensor:
    prototypes = F.normalize(prototypes, dim=1)
    similarity = prototypes @ prototypes.t()
    eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    return similarity.masked_select(~eye).pow(2).mean()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: TCRL, loader, device) -> dict[str, float]:
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
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive.")
    if args.num_parts <= 0:
        raise ValueError("--num-parts must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.triplet_margin < 0:
        raise ValueError("--triplet-margin must be non-negative.")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    for name in (
        "prototype_loss_weight",
        "part_prototype_loss_weight",
        "target_prototype_loss_weight",
        "contrastive_loss_weight",
        "triplet_loss_weight",
        "part_global_loss_weight",
        "prototype_separation_weight",
        "target_entropy_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"tcrl_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
