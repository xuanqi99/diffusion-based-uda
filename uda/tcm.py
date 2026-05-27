"""TCM-style transporting causal mechanisms baseline for UDA."""

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


class MechanismProxy(nn.Module):
    def __init__(self, feature_dim: int, proxy_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, proxy_dim),
            nn.BatchNorm1d(proxy_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proxy_dim, proxy_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TCM(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, num_mechanisms: int, proxy_dim: int) -> None:
        super().__init__()
        self.num_mechanisms = num_mechanisms
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.mechanism_selector = nn.Linear(feature_dim, num_mechanisms)
        self.proxies = nn.ModuleList([MechanismProxy(feature_dim, proxy_dim) for _ in range(num_mechanisms)])
        self.mechanism_classifiers = nn.ModuleList(
            [nn.Linear(feature_dim + proxy_dim, num_classes) for _ in range(num_mechanisms)]
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def mechanism_weights(self, features: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.softmax(self.mechanism_selector(features) / temperature, dim=1)

    def mechanism_logits(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proxies = []
        logits = []
        for proxy, classifier in zip(self.proxies, self.mechanism_classifiers):
            proxy_features = proxy(features)
            proxies.append(proxy_features)
            logits.append(classifier(torch.cat([features, proxy_features], dim=1)))
        return torch.stack(logits, dim=1), torch.stack(proxies, dim=1)

    def interventional_logits(
        self,
        images: torch.Tensor,
        prior: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(images)
        weights = self.mechanism_weights(features, temperature)
        logits, proxies = self.mechanism_logits(features)
        if prior is None:
            prior = weights.detach().mean(dim=0)
        prior = prior / prior.sum().clamp_min(1e-8)
        return (logits * prior.view(1, -1, 1)).sum(dim=1), weights, proxies

    def predict(self, images: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits, _, _ = self.interventional_logits(images, temperature=temperature)
        return logits

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, temperature: float):
        source_features = self.encode(source_images)
        target_features = self.encode(target_images)
        source_weights = self.mechanism_weights(source_features, temperature)
        target_weights = self.mechanism_weights(target_features, temperature)
        source_logits, source_proxies = self.mechanism_logits(source_features)
        target_logits, target_proxies = self.mechanism_logits(target_features)
        return source_logits, target_logits, source_weights, target_weights, source_proxies, target_proxies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCM baseline for UDA datasets.")
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
    parser.add_argument("--num-mechanisms", type=int, default=4)
    parser.add_argument("--proxy-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mechanism-alignment-weight", type=float, default=0.5)
    parser.add_argument("--prior-alignment-weight", type=float, default=0.1)
    parser.add_argument("--disentangle-weight", type=float, default=0.1)
    parser.add_argument("--target-loss-weight", type=float, default=0.5)
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
        raise ValueError("TCM requires a target training loader.")

    device = torch.device(args.device)
    model = TCM(args.arch, bundle.num_classes, args.pretrained, args.num_mechanisms, args.proxy_dim).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "source_loss",
            "target_loss",
            "mechanism_alignment_loss",
            "prior_alignment_loss",
            "disentangle_loss",
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
        f"Method: TCM | mechanisms: {args.num_mechanisms} | proxy_dim: {args.proxy_dim} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, args)
        if scheduler is not None:
            scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device, args)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device, args)["acc"]

        elapsed = time.time() - started_at
        lr = optimizer.param_groups[0]["lr"]
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"source={train_metrics['source_loss']:.4f} "
            f"target={train_metrics['target_loss']:.4f} "
            f"mech_align={train_metrics['mechanism_alignment_loss']:.4f} "
            f"prior_align={train_metrics['prior_alignment_loss']:.4f} "
            f"selected={train_metrics['selected_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: TCM, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    source_losses = AverageMeter()
    target_losses = AverageMeter()
    mechanism_alignment_losses = AverageMeter()
    prior_alignment_losses = AverageMeter()
    disentangle_losses = AverageMeter()
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
            (
                source_mech_logits,
                target_mech_logits,
                source_weights,
                target_weights,
                source_proxies,
                target_proxies,
            ) = model(source_images, target_images, args.temperature)

            source_prior = source_weights.mean(dim=0)
            target_prior = target_weights.mean(dim=0)
            source_logits = interventional_logits(source_mech_logits, source_prior)
            target_logits = interventional_logits(target_mech_logits, target_prior)

            source_loss = F.cross_entropy(source_logits, source_labels)
            pseudo_labels, target_mask = target_pseudo_labels(target_logits.detach(), args.target_confidence_threshold)
            if target_mask.any():
                target_loss = F.cross_entropy(target_logits[target_mask], pseudo_labels[target_mask])
            else:
                target_loss = target_logits.sum() * 0.0

            mechanism_alignment_loss = mechanism_proxy_alignment(source_proxies, target_proxies, source_weights, target_weights)
            prior_alignment_loss = F.mse_loss(source_prior, target_prior)
            disentangle_loss = mechanism_disentangle_loss(torch.cat([source_proxies, target_proxies], dim=0))
            target_entropy_loss = entropy_loss(target_logits)

            loss = (
                source_loss
                + args.target_loss_weight * target_loss
                + args.mechanism_alignment_weight * mechanism_alignment_loss
                + args.prior_alignment_weight * prior_alignment_loss
                + args.disentangle_weight * disentangle_loss
                + args.target_entropy_weight * target_entropy_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        source_losses.update(source_loss.item(), batch_size)
        target_losses.update(target_loss.item(), batch_size)
        mechanism_alignment_losses.update(mechanism_alignment_loss.item(), batch_size)
        prior_alignment_losses.update(prior_alignment_loss.item(), batch_size)
        disentangle_losses.update(disentangle_loss.item(), batch_size)
        target_entropy_losses.update(target_entropy_loss.item(), batch_size)
        selected_ratios.update(target_mask.float().mean().item(), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "source_loss": source_losses.avg,
        "target_loss": target_losses.avg,
        "mechanism_alignment_loss": mechanism_alignment_losses.avg,
        "prior_alignment_loss": prior_alignment_losses.avg,
        "disentangle_loss": disentangle_losses.avg,
        "target_entropy_loss": target_entropy_losses.avg,
        "selected_ratio": selected_ratios.avg,
        "total_loss": total_losses.avg,
    }


def interventional_logits(mechanism_logits: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    prior = prior / prior.sum().clamp_min(1e-8)
    return (mechanism_logits * prior.view(1, -1, 1)).sum(dim=1)


def mechanism_proxy_alignment(
    source_proxies: torch.Tensor,
    target_proxies: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for mechanism_id in range(source_proxies.size(1)):
        source_weight = source_weights[:, mechanism_id]
        target_weight = target_weights[:, mechanism_id]
        source_mean = weighted_mean(source_proxies[:, mechanism_id], source_weight)
        target_mean = weighted_mean(target_proxies[:, mechanism_id], target_weight)
        terms.append(F.mse_loss(source_mean, target_mean))
    return torch.stack(terms).mean()


def weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-8)
    return (features * weights.unsqueeze(1)).sum(dim=0)


def mechanism_disentangle_loss(proxies: torch.Tensor) -> torch.Tensor:
    pooled = F.normalize(proxies.mean(dim=0), dim=1)
    similarity = pooled @ pooled.t()
    eye = torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    return similarity.masked_select(~eye).pow(2).mean()


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
def evaluate(model: TCM, loader, device, args) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict(images, temperature=args.temperature)
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
    if args.num_mechanisms < 2:
        raise ValueError("--num-mechanisms must be at least 2.")
    if args.proxy_dim <= 0:
        raise ValueError("--proxy-dim must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    for name in (
        "mechanism_alignment_weight",
        "prior_alignment_weight",
        "disentangle_weight",
        "target_loss_weight",
        "target_entropy_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"tcm_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
