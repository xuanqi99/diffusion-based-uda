"""Maximum Classifier Discrepancy baseline for unsupervised domain adaptation."""

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


class MCD(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.feature_extractor, self.classifier1, feature_dim = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.classifier2 = nn.Linear(feature_dim, num_classes)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def classify(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier1(features), self.classifier2(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classify(self.extract(images))

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits1, logits2 = self.forward(images)
        return 0.5 * (logits1 + logits2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCD baseline for UDA datasets.")
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
    parser.add_argument("--feature-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--mcd-loss-weight", type=float, default=1.0)
    parser.add_argument("--generator-steps", type=int, default=4, help="Feature-generator discrepancy minimization steps per batch.")
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
        raise ValueError("MCD requires a target training loader.")

    device = torch.device(args.device)
    model = MCD(
        arch=args.arch,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    feature_optimizer = build_optimizer(args, model.feature_extractor.parameters(), lr=args.feature_lr or args.lr)
    classifier_optimizer = build_optimizer(
        args,
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
        lr=args.classifier_lr or args.lr,
    )
    feature_scheduler = build_scheduler(args, feature_optimizer, args.epochs)
    classifier_scheduler = build_scheduler(args, classifier_optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "source_loss",
            "classifier_loss",
            "target_discrepancy_max",
            "generator_loss",
            "source_acc",
            "target_acc",
            "feature_lr",
            "classifier_lr",
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
        f"Method: MCD | generator_steps: {args.generator_steps} | model: {args.arch} "
        f"| classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(
            model,
            bundle.source_train,
            bundle.target_train,
            feature_optimizer,
            classifier_optimizer,
            scaler,
            device,
            args,
        )
        if feature_scheduler is not None:
            feature_scheduler.step()
        if classifier_scheduler is not None:
            classifier_scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]

        elapsed = time.time() - started_at
        feature_lr = feature_optimizer.param_groups[0]["lr"]
        classifier_lr = classifier_optimizer.param_groups[0]["lr"]
        logger.log(
            {
                "epoch": epoch,
                "source_loss": train_metrics["source_loss"],
                "classifier_loss": train_metrics["classifier_loss"],
                "target_discrepancy_max": train_metrics["target_discrepancy_max"],
                "generator_loss": train_metrics["generator_loss"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "feature_lr": feature_lr,
                "classifier_lr": classifier_lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"source_loss={train_metrics['source_loss']:.4f} "
            f"disc_max={train_metrics['target_discrepancy_max']:.4f} "
            f"gen_loss={train_metrics['generator_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} "
            f"feature_lr={feature_lr:.6g} classifier_lr={classifier_lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)


def train_one_epoch(
    model: MCD,
    source_loader,
    target_loader,
    feature_optimizer,
    classifier_optimizer,
    scaler,
    device,
    args,
) -> dict[str, float]:
    source_losses = AverageMeter()
    classifier_losses = AverageMeter()
    target_discrepancies = AverageMeter()
    generator_losses = AverageMeter()
    target_iter = cycle(target_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        source_loss = train_source_step(model, source_images, source_labels, feature_optimizer, classifier_optimizer, scaler, device, args)
        classifier_loss, discrepancy_max = train_classifier_discrepancy_step(
            model,
            source_images,
            source_labels,
            target_images,
            classifier_optimizer,
            scaler,
            device,
            args,
        )
        generator_loss = train_generator_discrepancy_steps(model, target_images, feature_optimizer, scaler, device, args)

        batch_size = source_images.size(0)
        source_losses.update(source_loss, batch_size)
        classifier_losses.update(classifier_loss, batch_size)
        target_discrepancies.update(discrepancy_max, batch_size)
        generator_losses.update(generator_loss, batch_size)

    set_requires_grad(model.feature_extractor, True)
    set_requires_grad(model.classifier1, True)
    set_requires_grad(model.classifier2, True)
    return {
        "source_loss": source_losses.avg,
        "classifier_loss": classifier_losses.avg,
        "target_discrepancy_max": target_discrepancies.avg,
        "generator_loss": generator_losses.avg,
    }


def train_source_step(model: MCD, source_images, source_labels, feature_optimizer, classifier_optimizer, scaler, device, args) -> float:
    model.train()
    set_requires_grad(model.feature_extractor, True)
    set_requires_grad(model.classifier1, True)
    set_requires_grad(model.classifier2, True)
    feature_optimizer.zero_grad(set_to_none=True)
    classifier_optimizer.zero_grad(set_to_none=True)

    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        logits1, logits2 = model(source_images)
        loss = source_classification_loss(logits1, logits2, source_labels)

    scaler.scale(loss).backward()
    scaler.step(feature_optimizer)
    scaler.step(classifier_optimizer)
    scaler.update()
    return loss.item()


def train_classifier_discrepancy_step(
    model: MCD,
    source_images,
    source_labels,
    target_images,
    optimizer,
    scaler,
    device,
    args,
) -> tuple[float, float]:
    model.feature_extractor.eval()
    model.classifier1.train()
    model.classifier2.train()
    set_requires_grad(model.feature_extractor, False)
    set_requires_grad(model.classifier1, True)
    set_requires_grad(model.classifier2, True)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        with torch.no_grad():
            source_features = model.extract(source_images)
            target_features = model.extract(target_images)
        source_logits1, source_logits2 = model.classify(source_features)
        target_logits1, target_logits2 = model.classify(target_features)
        class_loss = source_classification_loss(source_logits1, source_logits2, source_labels)
        discrepancy = classifier_discrepancy(target_logits1, target_logits2)
        loss = class_loss - args.mcd_loss_weight * discrepancy

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), discrepancy.item()


def train_generator_discrepancy_steps(model: MCD, target_images, optimizer, scaler, device, args) -> float:
    if args.generator_steps <= 0:
        return 0.0

    model.feature_extractor.train()
    model.classifier1.eval()
    model.classifier2.eval()
    set_requires_grad(model.feature_extractor, True)
    set_requires_grad(model.classifier1, False)
    set_requires_grad(model.classifier2, False)
    losses = AverageMeter()

    for _ in range(args.generator_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            target_features = model.extract(target_images)
            target_logits1, target_logits2 = model.classify(target_features)
            loss = args.mcd_loss_weight * classifier_discrepancy(target_logits1, target_logits2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), target_images.size(0))

    return losses.avg


def source_classification_loss(logits1: torch.Tensor, logits2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)


def classifier_discrepancy(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    prob1 = F.softmax(logits1, dim=1)
    prob2 = F.softmax(logits2, dim=1)
    return torch.mean(torch.abs(prob1 - prob2))


@torch.no_grad()
def evaluate(model: MCD, loader, device) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits1, logits2 = model(images)
        logits = 0.5 * (logits1 + logits2)
        valid = labels >= 0
        if valid.sum().item() > 0:
            loss = source_classification_loss(logits1[valid], logits2[valid], labels[valid])
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


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"mcd_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, model, feature_optimizer, classifier_optimizer, epoch: int, args, target_acc: float) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizers": {
            "feature": feature_optimizer.state_dict(),
            "classifier": classifier_optimizer.state_dict(),
        },
        "target_acc": target_acc,
        "args": to_serializable_args(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
