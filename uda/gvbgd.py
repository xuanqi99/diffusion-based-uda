"""GVB-GD baseline from Gradually Vanishing Bridge for UDA."""

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
    from dann import compute_grl_lambda, evaluate, gradient_reverse, save_checkpoint
    from data import build_data, normalize_dataset_name
    from erm import autocast_context, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, save_json, set_seed, to_serializable_args
else:
    from .dann import compute_grl_lambda, evaluate, gradient_reverse, save_checkpoint
    from .data import build_data, normalize_dataset_name
    from .erm import autocast_context, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, save_json, set_seed, to_serializable_args


class BridgeClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.features, self.classifier, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.bridge = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor, use_bridge: bool = True):
        features = self.features(images)
        logits = self.classifier(features)
        bridge_logits = self.bridge(features)
        return logits - bridge_logits if use_bridge else logits, bridge_logits


class BridgeDomainDiscriminator(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.bridge = nn.Linear(hidden_dim, 1)

    def forward(self, probabilities: torch.Tensor, grl_lambda: float, use_bridge: bool = True):
        features = self.layers(gradient_reverse(probabilities, grl_lambda))
        logits = self.head(features).squeeze(1)
        bridge_logits = self.bridge(features).squeeze(1)
        return logits - bridge_logits if use_bridge else logits, bridge_logits


class GVBGD(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.generator = BridgeClassifier(arch, num_classes, pretrained)
        self.domain_discriminator = BridgeDomainDiscriminator(num_classes, hidden_dim, dropout)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.generator(images, use_bridge=True)
        return logits

    def forward(self, source_images, target_images, grl_lambda: float, generator_bridge: bool, discriminator_bridge: bool):
        source_logits, source_bridge = self.generator(source_images, generator_bridge)
        target_logits, target_bridge = self.generator(target_images, generator_bridge)
        probabilities = F.softmax(torch.cat([source_logits, target_logits], dim=0), dim=1)
        domain_logits, domain_bridge = self.domain_discriminator(probabilities, grl_lambda, discriminator_bridge)
        return source_logits, domain_logits, probabilities, torch.cat([source_bridge, target_bridge]), domain_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GVB-GD baseline for UDA datasets.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="officehome")
    parser.add_argument("--source", default="Art")
    parser.add_argument("--target", default="Clipart")
    parser.add_argument("--source-list", default=None)
    parser.add_argument("--target-list", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--bridge-lr", type=float, default=None)
    parser.add_argument("--discriminator-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--transfer-loss-weight", type=float, default=1.0)
    parser.add_argument("--generator-bridge-weight", type=float, default=1.0)
    parser.add_argument("--discriminator-bridge-weight", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-schedule", choices=("dann", "none"), default="dann")
    parser.add_argument("--entropy-eps", type=float, default=1e-5)
    parser.add_argument("--domain-hidden-dim", type=int, default=1024)
    parser.add_argument("--domain-dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-fake-data", action="store_true")
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
        raise ValueError("GVB-GD requires a target training loader.")
    device = torch.device(args.device)
    model = GVBGD(args.arch, bundle.num_classes, args.pretrained, args.domain_hidden_dim, args.domain_dropout).to(device)
    optimizer = build_gvbgd_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(output_dir / "metrics.csv", [
        "epoch", "class_loss", "transfer_loss", "generator_bridge_loss",
        "discriminator_bridge_loss", "prediction_entropy", "total_loss",
        "source_acc", "target_acc", "grl_lambda", "lr", "elapsed_sec",
    ])

    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) | target: {args.target} ({bundle.target_train_size})")
    print(f"Method: GVB-GD | model: {args.arch} | classes: {bundle.num_classes} | device: {device}")
    best_target_acc = -math.inf
    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, epoch, args)
        if scheduler is not None:
            scheduler.step()
        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": time.time() - started_at}
        logger.log(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} class_loss={train_metrics['class_loss']:.4f} "
            f"transfer_loss={train_metrics['transfer_loss']:.4f} gvbg={train_metrics['generator_bridge_loss']:.4f} "
            f"gvbd={train_metrics['discriminator_bridge_loss']:.4f} entropy={train_metrics['prediction_entropy']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lambda={train_metrics['grl_lambda']:.4f} lr={lr:.6g}"
        )
        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model, source_loader, target_loader, optimizer, scaler, device, epoch: int, args):
    model.train()
    meters = {name: AverageMeter() for name in [
        "class_loss", "transfer_loss", "generator_bridge_loss",
        "discriminator_bridge_loss", "prediction_entropy", "total_loss", "grl_lambda",
    ]}
    target_iter = cycle(target_loader)
    total_steps = args.steps_per_epoch or len(source_loader)
    use_gvb = args.generator_bridge_weight != 0.0
    use_dvb = args.discriminator_bridge_weight != 0.0

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
            source_logits, domain_logits, probabilities, generator_bridge, discriminator_bridge = model(
                source_images, target_images, grl_lambda, use_gvb, use_dvb
            )
            class_loss = F.cross_entropy(source_logits, source_labels)
            transfer_loss = compute_transfer_loss(probabilities, domain_logits, source_images.size(0), grl_lambda, args.entropy_eps)
            generator_bridge_loss = generator_bridge.abs().mean()
            discriminator_bridge_loss = discriminator_bridge.abs().mean()
            prediction_entropy = entropy(probabilities, args.entropy_eps).mean()
            loss = (
                class_loss
                + args.transfer_loss_weight * transfer_loss
                + args.generator_bridge_weight * generator_bridge_loss
                + args.discriminator_bridge_weight * discriminator_bridge_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = source_images.size(0)
        for name, value in [
            ("class_loss", class_loss), ("transfer_loss", transfer_loss),
            ("generator_bridge_loss", generator_bridge_loss), ("discriminator_bridge_loss", discriminator_bridge_loss),
            ("prediction_entropy", prediction_entropy), ("total_loss", loss),
        ]:
            meters[name].update(value.item(), n)
        meters["grl_lambda"].update(grl_lambda, n)
    return {name: meter.avg for name, meter in meters.items()}


def compute_transfer_loss(probabilities, domain_logits, source_count: int, grl_lambda: float, eps: float):
    labels = torch.cat([
        torch.ones(source_count, dtype=probabilities.dtype, device=probabilities.device),
        torch.zeros(probabilities.size(0) - source_count, dtype=probabilities.dtype, device=probabilities.device),
    ])
    weights = entropy_weights(probabilities, source_count, grl_lambda, eps)
    losses = F.binary_cross_entropy_with_logits(domain_logits, labels, reduction="none")
    return (losses * weights).sum() / 2.0


def entropy_weights(probabilities, source_count: int, grl_lambda: float, eps: float):
    weights = torch.exp(-gradient_reverse(entropy(probabilities, eps), grl_lambda))
    source_weights = weights[:source_count]
    target_weights = weights[source_count:]
    source_weights = source_weights / source_weights.sum().clamp_min(eps).detach()
    target_weights = target_weights / target_weights.sum().clamp_min(eps).detach()
    return torch.cat([source_weights, target_weights])


def entropy(probabilities, eps: float):
    return -(probabilities * (probabilities + eps).log()).sum(dim=1)


def build_gvbgd_optimizer(args, model: GVBGD):
    param_groups = [
        {"params": model.generator.features.parameters(), "lr": args.feature_lr or args.lr},
        {"params": model.generator.classifier.parameters(), "lr": args.classifier_lr or args.lr},
        {"params": model.generator.bridge.parameters(), "lr": args.bridge_lr or args.classifier_lr or args.lr},
        {"params": model.domain_discriminator.parameters(), "lr": args.discriminator_lr or args.lr},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


def validate_args(args) -> None:
    for name in ("transfer_loss_weight", "generator_bridge_weight", "discriminator_bridge_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.entropy_eps <= 0:
        raise ValueError("--entropy-eps must be positive.")


def make_output_dir(args) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"gvbgd_{args.dataset}_{args.source}_to_{args.target}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    main()
