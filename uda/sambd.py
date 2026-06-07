"""SAMB-D-style semantic-aware message broadcasting baseline for UDA."""

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


class MessageBroadcaster(nn.Module):
    def __init__(self, feature_dim: int, num_groups: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.group_tokens = nn.Parameter(torch.randn(num_groups, feature_dim) * 0.02)
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = F.normalize(self.group_tokens, dim=1)
        token_scores = F.normalize(features, dim=1) @ tokens.t()
        token_weights = F.softmax(token_scores, dim=1)

        queries = self.query(tokens)
        keys = self.key(features)
        values = self.value(features)
        attention = F.softmax(queries @ keys.t() / math.sqrt(keys.size(1)), dim=1)
        group_messages = attention @ values
        sample_messages = token_weights @ group_messages
        broadcast_features = F.normalize(features + self.refine(torch.cat([features, sample_messages], dim=1)), dim=1)
        return broadcast_features, token_weights, group_messages


class SAMBD(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool,
        num_groups: int,
        broadcast_hidden_dim: int,
        domain_hidden_dim: int,
        domain_dropout: float,
    ) -> None:
        super().__init__()
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.broadcaster = MessageBroadcaster(feature_dim, num_groups, broadcast_hidden_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(num_groups * feature_dim, domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, domain_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(domain_dropout),
            nn.Linear(domain_hidden_dim, 2),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images)
        return self.broadcaster(features)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        broadcast_features, _, _ = self.encode(images)
        return self.classifier(broadcast_features)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, grl_lambda: float):
        source_features, source_weights, source_messages = self.encode(source_images)
        target_features, target_weights, target_messages = self.encode(target_images)
        source_logits = self.classifier(source_features)
        target_logits = self.classifier(target_features)
        domain_features = torch.cat([source_messages.flatten(0), target_messages.flatten(0)], dim=0)
        domain_features = domain_features.view(2, -1)
        domain_logits = self.domain_discriminator(gradient_reverse(domain_features, grl_lambda))
        return {
            "source_features": source_features,
            "target_features": target_features,
            "source_weights": source_weights,
            "target_weights": target_weights,
            "source_messages": source_messages,
            "target_messages": target_messages,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "domain_logits": domain_logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAMB-D baseline for UDA datasets.")
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
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--broadcast-hidden-dim", type=int, default=256)
    parser.add_argument("--domain-hidden-dim", type=int, default=512)
    parser.add_argument("--domain-dropout", type=float, default=0.5)
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-entropy-weight", type=float, default=0.01)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.1)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-schedule", choices=("dann", "none"), default="dann")
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
        raise ValueError("SAMB-D requires a target training loader.")

    device = torch.device(args.device)
    model = SAMBD(
        args.arch,
        bundle.num_classes,
        args.pretrained,
        args.num_groups,
        args.broadcast_hidden_dim,
        args.domain_hidden_dim,
        args.domain_dropout,
    ).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "domain_loss",
            "target_loss",
            "target_entropy_loss",
            "diversity_loss",
            "consistency_loss",
            "selected_ratio",
            "total_loss",
            "source_acc",
            "target_acc",
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
        f"Method: SAMB-D | groups: {args.num_groups} | model: {args.arch} "
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
            f"domain={train_metrics['domain_loss']:.4f} "
            f"target={train_metrics['target_loss']:.4f} "
            f"selected={train_metrics['selected_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} "
            f"lambda={train_metrics['grl_lambda']:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: SAMBD, source_loader, target_loader, optimizer, scaler, device, epoch: int, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    domain_losses = AverageMeter()
    target_losses = AverageMeter()
    entropy_losses = AverageMeter()
    diversity_losses = AverageMeter()
    consistency_losses = AverageMeter()
    selected_ratios = AverageMeter()
    total_losses = AverageMeter()
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
            output = model(source_images, target_images, grl_lambda)
            class_loss = F.cross_entropy(output["source_logits"], source_labels)
            domain_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
            domain_loss = F.cross_entropy(output["domain_logits"], domain_labels)
            pseudo_labels, target_mask = target_pseudo_labels(output["target_logits"].detach(), args.target_confidence_threshold)
            if target_mask.any():
                target_loss = F.cross_entropy(output["target_logits"][target_mask], pseudo_labels[target_mask])
            else:
                target_loss = output["target_logits"].sum() * 0.0
            target_entropy_loss = entropy_loss(output["target_logits"])
            diversity_loss = group_diversity_loss(model.broadcaster.group_tokens)
            consistency_loss = message_consistency_loss(output["source_messages"], output["target_messages"])
            loss = (
                class_loss
                + args.domain_loss_weight * domain_loss
                + args.target_loss_weight * target_loss
                + args.target_entropy_weight * target_entropy_loss
                + args.diversity_loss_weight * diversity_loss
                + args.consistency_loss_weight * consistency_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        domain_losses.update(domain_loss.item(), batch_size)
        target_losses.update(target_loss.item(), batch_size)
        entropy_losses.update(target_entropy_loss.item(), batch_size)
        diversity_losses.update(diversity_loss.item(), batch_size)
        consistency_losses.update(consistency_loss.item(), batch_size)
        selected_ratios.update(target_mask.float().mean().item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        lambda_meter.update(grl_lambda, batch_size)

    return {
        "class_loss": class_losses.avg,
        "domain_loss": domain_losses.avg,
        "target_loss": target_losses.avg,
        "target_entropy_loss": entropy_losses.avg,
        "diversity_loss": diversity_losses.avg,
        "consistency_loss": consistency_losses.avg,
        "selected_ratio": selected_ratios.avg,
        "total_loss": total_losses.avg,
        "grl_lambda": lambda_meter.avg,
    }


@torch.no_grad()
def target_pseudo_labels(logits: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=1)
    confidence, labels = probabilities.max(dim=1)
    return labels, confidence >= threshold


def group_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(tokens, dim=1)
    similarity = normalized @ normalized.t()
    eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    return similarity.masked_select(~eye).pow(2).mean()


def message_consistency_loss(source_messages: torch.Tensor, target_messages: torch.Tensor) -> torch.Tensor:
    source_messages = F.normalize(source_messages, dim=1)
    target_messages = F.normalize(target_messages, dim=1)
    return F.mse_loss(source_messages, target_messages)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: SAMBD, loader, device) -> dict[str, float]:
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
    if args.num_groups < 2:
        raise ValueError("--num-groups must be at least 2.")
    if args.broadcast_hidden_dim <= 0:
        raise ValueError("--broadcast-hidden-dim must be positive.")
    if args.domain_hidden_dim <= 0:
        raise ValueError("--domain-hidden-dim must be positive.")
    if not 0.0 <= args.domain_dropout < 1.0:
        raise ValueError("--domain-dropout must be in [0, 1).")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    for name in (
        "domain_loss_weight",
        "target_loss_weight",
        "target_entropy_weight",
        "diversity_loss_weight",
        "consistency_loss_weight",
        "grl_lambda",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"sambd_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
