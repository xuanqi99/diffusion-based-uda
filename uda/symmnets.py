"""SymmNets baseline from "Domain-Symmetric Networks" for UDA."""

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


class SymmNets(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        self.feature_extractor, self.source_classifier, feature_dim = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.target_classifier = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def classify(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.source_classifier(features), self.target_classifier(features)

    def cst_logits(self, features: torch.Tensor) -> torch.Tensor:
        source_logits, target_logits = self.classify(features)
        return torch.cat([source_logits, target_logits], dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classify(self.extract(images))

    def predict(self, images: torch.Tensor, head: str = "target") -> torch.Tensor:
        source_logits, target_logits = self.forward(images)
        if head == "source":
            return source_logits
        if head == "mean":
            return 0.5 * (source_logits + target_logits)
        if head == "symm":
            cst_probs = F.softmax(torch.cat([source_logits, target_logits], dim=1), dim=1)
            class_probs = cst_probs[:, : self.num_classes] + cst_probs[:, self.num_classes :]
            return torch.log(class_probs.clamp_min(1e-8))
        if head != "target":
            raise ValueError(f"Unsupported prediction head: {head}")
        return target_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymmNets baseline for UDA datasets.")
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
    parser.add_argument("--classifier-domain-weight", type=float, default=1.0)
    parser.add_argument("--confusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-loss-weight", type=float, default=1.0)
    parser.add_argument("--confusion-schedule", choices=("dann", "none"), default="dann")
    parser.add_argument("--eval-head", choices=("source", "target", "mean", "symm"), default="target")
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
        raise ValueError("SymmNets requires a target training loader.")

    device = torch.device(args.device)
    model = SymmNets(
        arch=args.arch,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
    ).to(device)
    feature_optimizer = build_optimizer(args, model.feature_extractor.parameters(), lr=args.feature_lr or args.lr)
    classifier_optimizer = build_optimizer(
        args,
        list(model.source_classifier.parameters()) + list(model.target_classifier.parameters()),
        lr=args.classifier_lr or args.lr,
    )
    feature_scheduler = build_scheduler(args, feature_optimizer, args.epochs)
    classifier_scheduler = build_scheduler(args, classifier_optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "source_task_loss",
            "domain_discrimination_loss",
            "classifier_loss",
            "category_confusion_loss",
            "domain_confusion_loss",
            "target_entropy_loss",
            "feature_loss",
            "source_acc",
            "target_acc",
            "confusion_weight",
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
        f"Method: SymmNets | eval_head: {args.eval_head} | model: {args.arch} "
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
            epoch,
            args,
        )
        if feature_scheduler is not None:
            feature_scheduler.step()
        if classifier_scheduler is not None:
            classifier_scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device, head="source")["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device, head=args.eval_head)["acc"]

        elapsed = time.time() - started_at
        feature_lr = feature_optimizer.param_groups[0]["lr"]
        classifier_lr = classifier_optimizer.param_groups[0]["lr"]
        logger.log(
            {
                "epoch": epoch,
                "source_task_loss": train_metrics["source_task_loss"],
                "domain_discrimination_loss": train_metrics["domain_discrimination_loss"],
                "classifier_loss": train_metrics["classifier_loss"],
                "category_confusion_loss": train_metrics["category_confusion_loss"],
                "domain_confusion_loss": train_metrics["domain_confusion_loss"],
                "target_entropy_loss": train_metrics["target_entropy_loss"],
                "feature_loss": train_metrics["feature_loss"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "confusion_weight": train_metrics["confusion_weight"],
                "feature_lr": feature_lr,
                "classifier_lr": classifier_lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"task={train_metrics['source_task_loss']:.4f} "
            f"disc={train_metrics['domain_discrimination_loss']:.4f} "
            f"cat_conf={train_metrics['category_confusion_loss']:.4f} "
            f"dom_conf={train_metrics['domain_confusion_loss']:.4f} "
            f"ent={train_metrics['target_entropy_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} "
            f"lambda={train_metrics['confusion_weight']:.4f} "
            f"feature_lr={feature_lr:.6g} classifier_lr={classifier_lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, feature_optimizer, classifier_optimizer, epoch, args, target_acc)


def train_one_epoch(
    model: SymmNets,
    source_loader,
    target_loader,
    feature_optimizer,
    classifier_optimizer,
    scaler,
    device,
    epoch: int,
    args,
) -> dict[str, float]:
    source_task_losses = AverageMeter()
    domain_discrimination_losses = AverageMeter()
    classifier_losses = AverageMeter()
    category_confusion_losses = AverageMeter()
    domain_confusion_losses = AverageMeter()
    target_entropy_losses = AverageMeter()
    feature_losses = AverageMeter()
    confusion_weights = AverageMeter()
    target_iter = cycle(target_loader)
    total_steps = args.steps_per_epoch or len(source_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)
        confusion_weight = compute_confusion_weight(epoch, step, total_steps, args)

        classifier_metrics = train_classifier_step(
            model,
            source_images,
            source_labels,
            target_images,
            classifier_optimizer,
            scaler,
            device,
            args,
        )
        feature_metrics = train_feature_step(
            model,
            source_images,
            source_labels,
            target_images,
            feature_optimizer,
            scaler,
            device,
            args,
            confusion_weight,
        )

        batch_size = source_images.size(0)
        source_task_losses.update(classifier_metrics["source_task_loss"], batch_size)
        domain_discrimination_losses.update(classifier_metrics["domain_discrimination_loss"], batch_size)
        classifier_losses.update(classifier_metrics["classifier_loss"], batch_size)
        category_confusion_losses.update(feature_metrics["category_confusion_loss"], batch_size)
        domain_confusion_losses.update(feature_metrics["domain_confusion_loss"], batch_size)
        target_entropy_losses.update(feature_metrics["target_entropy_loss"], batch_size)
        feature_losses.update(feature_metrics["feature_loss"], batch_size)
        confusion_weights.update(confusion_weight, batch_size)

    set_requires_grad(model.feature_extractor, True)
    set_requires_grad(model.source_classifier, True)
    set_requires_grad(model.target_classifier, True)
    return {
        "source_task_loss": source_task_losses.avg,
        "domain_discrimination_loss": domain_discrimination_losses.avg,
        "classifier_loss": classifier_losses.avg,
        "category_confusion_loss": category_confusion_losses.avg,
        "domain_confusion_loss": domain_confusion_losses.avg,
        "target_entropy_loss": target_entropy_losses.avg,
        "feature_loss": feature_losses.avg,
        "confusion_weight": confusion_weights.avg,
    }


def train_classifier_step(
    model: SymmNets,
    source_images,
    source_labels,
    target_images,
    optimizer,
    scaler,
    device,
    args,
) -> dict[str, float]:
    model.feature_extractor.eval()
    model.source_classifier.train()
    model.target_classifier.train()
    set_requires_grad(model.feature_extractor, False)
    set_requires_grad(model.source_classifier, True)
    set_requires_grad(model.target_classifier, True)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        with torch.no_grad():
            source_features = model.extract(source_images)
            target_features = model.extract(target_images)
        source_logits_s, source_logits_t = model.classify(source_features)
        target_logits_s, target_logits_t = model.classify(target_features)
        task_loss = source_task_loss(source_logits_s, source_logits_t, source_labels)
        domain_loss = domain_discrimination_loss(
            torch.cat([source_logits_s, source_logits_t], dim=1),
            torch.cat([target_logits_s, target_logits_t], dim=1),
            model.num_classes,
        )
        loss = task_loss + args.classifier_domain_weight * domain_loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "source_task_loss": task_loss.item(),
        "domain_discrimination_loss": domain_loss.item(),
        "classifier_loss": loss.item(),
    }


def train_feature_step(
    model: SymmNets,
    source_images,
    source_labels,
    target_images,
    optimizer,
    scaler,
    device,
    args,
    confusion_weight: float,
) -> dict[str, float]:
    model.feature_extractor.train()
    model.source_classifier.eval()
    model.target_classifier.eval()
    set_requires_grad(model.feature_extractor, True)
    set_requires_grad(model.source_classifier, False)
    set_requires_grad(model.target_classifier, False)
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        source_cst_logits = model.cst_logits(model.extract(source_images))
        target_cst_logits = model.cst_logits(model.extract(target_images))
        category_loss = category_confusion_loss(source_cst_logits, source_labels, model.num_classes)
        domain_loss = domain_confusion_loss(target_cst_logits, model.num_classes)
        entropy_loss = target_entropy_loss(target_cst_logits, model.num_classes)
        loss = category_loss + confusion_weight * (
            domain_loss + args.entropy_loss_weight * entropy_loss
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "category_confusion_loss": category_loss.item(),
        "domain_confusion_loss": domain_loss.item(),
        "target_entropy_loss": entropy_loss.item(),
        "feature_loss": loss.item(),
    }


def source_task_loss(source_logits: torch.Tensor, target_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(source_logits, labels) + F.cross_entropy(target_logits, labels)


def domain_discrimination_loss(source_cst_logits: torch.Tensor, target_cst_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    source_log_probs = F.log_softmax(source_cst_logits, dim=1)
    target_log_probs = F.log_softmax(target_cst_logits, dim=1)
    source_domain_log_prob = torch.logsumexp(source_log_probs[:, :num_classes], dim=1)
    target_domain_log_prob = torch.logsumexp(target_log_probs[:, num_classes:], dim=1)
    return -source_domain_log_prob.mean() - target_domain_log_prob.mean()


def category_confusion_loss(source_cst_logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    log_probs = F.log_softmax(source_cst_logits, dim=1)
    source_label_log_prob = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    target_label_log_prob = log_probs.gather(1, (labels + num_classes).view(-1, 1)).squeeze(1)
    return -0.5 * (source_label_log_prob.mean() + target_label_log_prob.mean())


def domain_confusion_loss(target_cst_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    log_probs = F.log_softmax(target_cst_logits, dim=1)
    source_domain_log_prob = torch.logsumexp(log_probs[:, :num_classes], dim=1)
    target_domain_log_prob = torch.logsumexp(log_probs[:, num_classes:], dim=1)
    return -0.5 * (source_domain_log_prob.mean() + target_domain_log_prob.mean())


def target_entropy_loss(target_cst_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    cst_probs = F.softmax(target_cst_logits, dim=1)
    class_probs = cst_probs[:, :num_classes] + cst_probs[:, num_classes:]
    return -(class_probs * class_probs.clamp_min(1e-8).log()).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: SymmNets, loader, device, head: str) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict(images, head=head)
        valid = labels >= 0
        if valid.sum().item() > 0:
            loss = F.cross_entropy(logits[valid], labels[valid])
            batch_acc, count = accuracy(logits, labels)
            losses.update(loss.item(), count)
            acc_meter.update(batch_acc, count)

    if acc_meter.count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": losses.avg, "acc": acc_meter.avg}


def compute_confusion_weight(epoch: int, step: int, total_steps: int, args) -> float:
    if args.confusion_schedule == "none":
        return args.confusion_loss_weight
    progress = ((epoch - 1) * total_steps + step) / max(args.epochs * total_steps, 1)
    schedule_value = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return args.confusion_loss_weight * schedule_value


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
        output_dir = Path("runs") / f"symmnets_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
