"""CPH-style comparative prototype hashing baseline for UDA."""

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


class CPH(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, embedding_dim: int, hash_bits: int) -> None:
        super().__init__()
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.hash_head = nn.Linear(embedding_dim, hash_bits)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images)
        embeddings = F.normalize(self.embedding_head(features), dim=1)
        return features, embeddings

    def hash_codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.hash_head(embeddings))

    def prototype_logits(self, embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=1)
        return embeddings @ prototypes.t() / temperature

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        _, embeddings = self.encode(images)
        return self.classifier(embeddings)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, temperature: float):
        _, source_embeddings = self.encode(source_images)
        _, target_embeddings = self.encode(target_images)
        source_hash = self.hash_codes(source_embeddings)
        target_hash = self.hash_codes(target_embeddings)
        source_logits = self.classifier(source_embeddings)
        target_logits = self.classifier(target_embeddings)
        source_proto_logits = self.prototype_logits(source_embeddings, temperature)
        target_proto_logits = self.prototype_logits(target_embeddings, temperature)
        return {
            "source_embeddings": source_embeddings,
            "target_embeddings": target_embeddings,
            "source_hash": source_hash,
            "target_hash": target_hash,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "source_proto_logits": source_proto_logits,
            "target_proto_logits": target_proto_logits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPH baseline for UDA datasets.")
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
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--prototype-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-prototype-loss-weight", type=float, default=0.5)
    parser.add_argument("--relation-loss-weight", type=float, default=0.2)
    parser.add_argument("--quantization-loss-weight", type=float, default=0.1)
    parser.add_argument("--balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--prototype-separation-weight", type=float, default=0.1)
    parser.add_argument("--target-entropy-weight", type=float, default=0.01)
    parser.add_argument("--target-confidence-threshold", type=float, default=0.6)
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
        raise ValueError("CPH requires a target training loader.")

    device = torch.device(args.device)
    model = CPH(args.arch, bundle.num_classes, args.pretrained, args.embedding_dim, args.hash_bits).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "source_prototype_loss",
            "target_prototype_loss",
            "relation_loss",
            "quantization_loss",
            "balance_loss",
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
        f"Method: CPH | embedding_dim: {args.embedding_dim} | hash_bits: {args.hash_bits} "
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
            f"proto={train_metrics['source_prototype_loss']:.4f} "
            f"target_proto={train_metrics['target_prototype_loss']:.4f} "
            f"relation={train_metrics['relation_loss']:.4f} "
            f"selected={train_metrics['selected_target_ratio']:.3f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: CPH, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    source_prototype_losses = AverageMeter()
    target_prototype_losses = AverageMeter()
    relation_losses = AverageMeter()
    quantization_losses = AverageMeter()
    balance_losses = AverageMeter()
    prototype_separation_losses = AverageMeter()
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
            output = model(source_images, target_images, args.temperature)
            class_loss = F.cross_entropy(output["source_logits"], source_labels)
            source_prototype_loss = F.cross_entropy(output["source_proto_logits"], source_labels)
            target_pseudo, target_mask = target_pseudo_labels(output["target_proto_logits"].detach(), args.target_confidence_threshold)
            if target_mask.any():
                target_prototype_loss = F.cross_entropy(output["target_proto_logits"][target_mask], target_pseudo[target_mask])
            else:
                target_prototype_loss = output["target_proto_logits"].sum() * 0.0

            relation_loss = dual_domain_relation_loss(
                output["source_hash"],
                output["target_hash"],
                source_labels,
                target_pseudo,
                target_mask,
            )
            quantization_loss = hash_quantization_loss(output["source_hash"], output["target_hash"])
            balance_loss = hash_balance_loss(output["source_hash"], output["target_hash"])
            prototype_separation_loss = prototype_separation(model.prototypes)
            target_entropy_loss = entropy_loss(output["target_logits"])

            loss = (
                class_loss
                + args.prototype_loss_weight * source_prototype_loss
                + args.target_prototype_loss_weight * target_prototype_loss
                + args.relation_loss_weight * relation_loss
                + args.quantization_loss_weight * quantization_loss
                + args.balance_loss_weight * balance_loss
                + args.prototype_separation_weight * prototype_separation_loss
                + args.target_entropy_weight * target_entropy_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        source_prototype_losses.update(source_prototype_loss.item(), batch_size)
        target_prototype_losses.update(target_prototype_loss.item(), batch_size)
        relation_losses.update(relation_loss.item(), batch_size)
        quantization_losses.update(quantization_loss.item(), batch_size)
        balance_losses.update(balance_loss.item(), batch_size)
        prototype_separation_losses.update(prototype_separation_loss.item(), batch_size)
        target_entropy_losses.update(target_entropy_loss.item(), batch_size)
        selected_ratios.update(target_mask.float().mean().item(), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "source_prototype_loss": source_prototype_losses.avg,
        "target_prototype_loss": target_prototype_losses.avg,
        "relation_loss": relation_losses.avg,
        "quantization_loss": quantization_losses.avg,
        "balance_loss": balance_losses.avg,
        "prototype_separation_loss": prototype_separation_losses.avg,
        "target_entropy_loss": target_entropy_losses.avg,
        "selected_target_ratio": selected_ratios.avg,
        "total_loss": total_losses.avg,
    }


@torch.no_grad()
def target_pseudo_labels(logits: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=1)
    confidence, labels = probabilities.max(dim=1)
    return labels, confidence >= threshold


def dual_domain_relation_loss(
    source_hash: torch.Tensor,
    target_hash: torch.Tensor,
    source_labels: torch.Tensor,
    target_pseudo: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    losses = [pairwise_relation_loss(source_hash, source_labels)]
    if target_mask.sum().item() >= 2:
        losses.append(pairwise_relation_loss(target_hash[target_mask], target_pseudo[target_mask]))
    if target_mask.any():
        cross_hash = torch.cat([source_hash, target_hash[target_mask]], dim=0)
        cross_labels = torch.cat([source_labels, target_pseudo[target_mask]], dim=0)
        losses.append(pairwise_relation_loss(cross_hash, cross_labels))
    return torch.stack(losses).mean()


def pairwise_relation_loss(hash_codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if hash_codes.size(0) < 2:
        return hash_codes.sum() * 0.0
    similarities = F.normalize(hash_codes, dim=1) @ F.normalize(hash_codes, dim=1).t()
    targets = torch.where(labels.view(-1, 1) == labels.view(1, -1), 1.0, -1.0).to(hash_codes.dtype)
    eye = torch.eye(hash_codes.size(0), dtype=torch.bool, device=hash_codes.device)
    return F.mse_loss(similarities.masked_select(~eye), targets.masked_select(~eye))


def hash_quantization_loss(source_hash: torch.Tensor, target_hash: torch.Tensor) -> torch.Tensor:
    codes = torch.cat([source_hash, target_hash], dim=0)
    return (codes.abs() - 1.0).pow(2).mean()


def hash_balance_loss(source_hash: torch.Tensor, target_hash: torch.Tensor) -> torch.Tensor:
    codes = torch.cat([source_hash, target_hash], dim=0)
    return codes.mean(dim=0).pow(2).mean()


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
def evaluate(model: CPH, loader, device) -> dict[str, float]:
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
    if args.hash_bits <= 0:
        raise ValueError("--hash-bits must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0.0 <= args.target_confidence_threshold <= 1.0:
        raise ValueError("--target-confidence-threshold must be in [0, 1].")
    for name in (
        "prototype_loss_weight",
        "target_prototype_loss_weight",
        "relation_loss_weight",
        "quantization_loss_weight",
        "balance_loss_weight",
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
        output_dir = Path("runs") / f"cph_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
