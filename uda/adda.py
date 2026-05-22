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
    from erm import autocast_context, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import build_data, normalize_dataset_name
    from .erm import autocast_context, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ADDA(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.source_encoder, self.classifier, feature_dim = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.target_encoder, _, _ = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.discriminator = DomainDiscriminator(feature_dim, hidden_dim, dropout)
        self.sync_target_encoder()

    def sync_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())

    def predict_source(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.source_encoder(images))

    def predict_target(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.target_encoder(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADDA baseline for UDA datasets.")
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
    parser.add_argument("--source-epochs", type=int, default=5, help="Supervised source pretraining epochs.")
    parser.add_argument("--epochs", type=int, default=20, help="Adversarial target adaptation epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-lr", type=float, default=None)
    parser.add_argument("--discriminator-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--domain-hidden-dim", type=int, default=1024)
    parser.add_argument("--domain-dropout", type=float, default=0.5)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
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
    set_seed(args.seed)

    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    if bundle.target_train is None:
        raise ValueError("ADDA requires a target training loader.")

    device = torch.device(args.device)
    model = ADDA(
        arch=args.arch,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
        hidden_dim=args.domain_hidden_dim,
        dropout=args.domain_dropout,
    ).to(device)

    source_optimizer = build_optimizer(
        args,
        list(model.source_encoder.parameters()) + list(model.classifier.parameters()),
        lr=args.lr,
    )
    target_optimizer = build_optimizer(args, model.target_encoder.parameters(), lr=args.target_lr or args.lr)
    discriminator_optimizer = build_optimizer(
        args,
        model.discriminator.parameters(),
        lr=args.discriminator_lr or args.lr,
    )
    source_scheduler = build_scheduler(args, source_optimizer, max(args.source_epochs, 1))
    target_scheduler = build_scheduler(args, target_optimizer, max(args.epochs, 1))
    discriminator_scheduler = build_scheduler(args, discriminator_optimizer, max(args.epochs, 1))
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "stage",
            "epoch",
            "class_loss",
            "discriminator_loss",
            "target_adv_loss",
            "domain_acc",
            "source_acc",
            "target_acc",
            "source_lr",
            "target_lr",
            "discriminator_lr",
            "elapsed_sec",
        ],
    )

    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(
        f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) "
        f"| target: {args.target} ({bundle.target_train_size})"
    )
    print(f"Method: ADDA | model: {args.arch} | classes: {bundle.num_classes} | device: {device}")

    for epoch in range(1, args.source_epochs + 1):
        started_at = time.time()
        class_loss = train_source_epoch(model, bundle.source_train, source_optimizer, scaler, device, args)
        if source_scheduler is not None:
            source_scheduler.step()
        model.source_encoder.eval()
        model.classifier.eval()
        source_acc = evaluate(model.predict_source, bundle.source_eval, device)["acc"]
        row = {
            "stage": "source",
            "epoch": epoch,
            "class_loss": class_loss,
            "source_acc": source_acc,
            "target_acc": float("nan"),
            "source_lr": source_optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started_at,
        }
        logger.log(row)
        print(f"[source] Epoch {epoch:03d}/{args.source_epochs:03d} class_loss={class_loss:.4f} source_acc={source_acc:.4f}")
        save_checkpoint(output_dir / "checkpoint_last.pt", model, epoch, "source", args, source_optimizer, target_optimizer, discriminator_optimizer, float("nan"))

    model.sync_target_encoder()
    freeze_module(model.source_encoder)
    freeze_module(model.classifier)

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        metrics = train_adapt_epoch(
            model,
            bundle.source_train,
            bundle.target_train,
            target_optimizer,
            discriminator_optimizer,
            scaler,
            device,
            args,
        )
        if target_scheduler is not None:
            target_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

        model.source_encoder.eval()
        model.target_encoder.eval()
        model.classifier.eval()
        source_acc = evaluate(model.predict_source, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model.predict_target, bundle.target_eval, device)["acc"]
        row = {
            "stage": "adapt",
            "epoch": epoch,
            **metrics,
            "source_acc": source_acc,
            "target_acc": target_acc,
            "target_lr": target_optimizer.param_groups[0]["lr"],
            "discriminator_lr": discriminator_optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - started_at,
        }
        logger.log(row)
        print(
            f"[adapt] Epoch {epoch:03d}/{args.epochs:03d} "
            f"d_loss={metrics['discriminator_loss']:.4f} adv_loss={metrics['target_adv_loss']:.4f} "
            f"domain_acc={metrics['domain_acc']:.4f} source_acc={source_acc:.4f} target_acc={target_acc:.4f}"
        )
        save_checkpoint(output_dir / "checkpoint_last.pt", model, epoch, "adapt", args, source_optimizer, target_optimizer, discriminator_optimizer, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, epoch, "adapt", args, source_optimizer, target_optimizer, discriminator_optimizer, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, epoch, "adapt", args, source_optimizer, target_optimizer, discriminator_optimizer, target_acc)


def train_source_epoch(model: ADDA, loader, optimizer, scaler, device, args) -> float:
    model.source_encoder.train()
    model.classifier.train()
    losses = AverageMeter()

    for step, (images, labels) in enumerate(loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            loss = F.cross_entropy(model.predict_source(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))
    return losses.avg


def train_adapt_epoch(model: ADDA, source_loader, target_loader, target_optimizer, discriminator_optimizer, scaler, device, args) -> dict[str, float]:
    model.source_encoder.eval()
    model.classifier.eval()
    model.target_encoder.train()
    model.discriminator.train()
    discriminator_losses = AverageMeter()
    target_losses = AverageMeter()
    domain_accs = AverageMeter()
    target_iter = cycle(target_loader)

    for step, (source_images, _) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break
        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        d_loss, d_acc = train_discriminator(model, source_images, target_images, discriminator_optimizer, scaler, device, args)
        target_loss = train_target_encoder(model, target_images, target_optimizer, scaler, device, args)
        batch_size = source_images.size(0) + target_images.size(0)
        discriminator_losses.update(d_loss, batch_size)
        target_losses.update(target_loss, target_images.size(0))
        domain_accs.update(d_acc, batch_size)

    return {
        "discriminator_loss": discriminator_losses.avg,
        "target_adv_loss": target_losses.avg,
        "domain_acc": domain_accs.avg,
    }


def train_discriminator(model: ADDA, source_images, target_images, optimizer, scaler, device, args) -> tuple[float, float]:
    model.discriminator.train()
    set_requires_grad(model.discriminator, True)
    set_requires_grad(model.target_encoder, False)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        with torch.no_grad():
            source_features = model.source_encoder(source_images)
            target_features = model.target_encoder(target_images)
        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat(
            [
                torch.zeros(source_features.size(0), dtype=torch.long, device=device),
                torch.ones(target_features.size(0), dtype=torch.long, device=device),
            ],
            dim=0,
        )
        logits = model.discriminator(features)
        loss = F.cross_entropy(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    batch_acc, _ = accuracy(logits.detach(), labels)
    return loss.item(), batch_acc


def train_target_encoder(model: ADDA, target_images, optimizer, scaler, device, args) -> float:
    model.discriminator.eval()
    set_requires_grad(model.discriminator, False)
    set_requires_grad(model.target_encoder, True)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        target_features = model.target_encoder(target_images)
        logits = model.discriminator(target_features)
        source_labels = torch.zeros(target_features.size(0), dtype=torch.long, device=device)
        loss = args.target_loss_weight * F.cross_entropy(logits, source_labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


@torch.no_grad()
def evaluate(predict_fn, loader, device) -> dict[str, float]:
    losses = AverageMeter()
    acc_meter = AverageMeter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = predict_fn(images)
        valid = labels >= 0
        if valid.sum().item() > 0:
            loss = F.cross_entropy(logits[valid], labels[valid])
            batch_acc, count = accuracy(logits, labels)
            losses.update(loss.item(), count)
            acc_meter.update(batch_acc, count)
    if acc_meter.count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": losses.avg, "acc": acc_meter.avg}


def build_optimizer(args, parameters, lr: float):
    params = list(parameters)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def freeze_module(module: nn.Module) -> None:
    module.eval()
    set_requires_grad(module, False)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"adda_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, model, epoch: int, stage: str, args, source_optimizer, target_optimizer, discriminator_optimizer, target_acc: float) -> None:
    payload = {
        "epoch": epoch,
        "stage": stage,
        "model": model.state_dict(),
        "optimizers": {
            "source": source_optimizer.state_dict(),
            "target": target_optimizer.state_dict(),
            "discriminator": discriminator_optimizer.state_dict(),
        },
        "target_acc": target_acc,
        "args": to_serializable_args(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
