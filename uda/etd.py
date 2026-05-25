"""Enhanced Transport Distance baseline for unsupervised domain adaptation."""

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
    from data import DATASET_SPECS, build_data, normalize_dataset_name
    from erm import autocast_context, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import DATASET_SPECS, build_data, normalize_dataset_name
    from .erm import autocast_context, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class PotentialNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class AttentionWeight(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.query = nn.Linear(num_classes, num_classes)
        self.key = nn.Linear(num_classes, num_classes)

    def forward(self, source_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        source_probs = F.softmax(source_logits, dim=1)
        target_probs = F.softmax(target_logits, dim=1)
        scores = self.query(source_probs) @ self.key(target_probs).t()
        return torch.sigmoid(scores)


class ETD(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, potential_hidden_dim: int) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.source_potential = PotentialNetwork(feature_dim, potential_hidden_dim)
        self.target_potential = PotentialNetwork(feature_dim, potential_hidden_dim)
        self.attention = AttentionWeight(num_classes)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.classifier(features)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.feature_extractor(source_images)
        target_features = self.feature_extractor(target_images)
        source_logits = self.classifier(source_features)
        target_logits = self.classifier(target_features)
        return source_logits, target_logits, source_features, target_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETD baseline for UDA datasets.")
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
    parser.add_argument("--potential-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--etd-loss-weight", type=float, default=1.0)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.1, help="Entropic smoothing for the dual transport objective.")
    parser.add_argument("--potential-steps", type=int, default=5, help="U/V potential updates per mini-batch.")
    parser.add_argument("--potential-hidden-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA for the main update.")
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
        raise ValueError("ETD requires a target training loader.")

    device = torch.device(args.device)
    model = ETD(args.arch, bundle.num_classes, args.pretrained, args.potential_hidden_dim).to(device)
    main_optimizer = build_etd_optimizer(args, main_parameters(model), args.lr)
    potential_optimizer = build_etd_optimizer(args, potential_parameters(model), args.potential_lr or args.lr)
    scheduler = build_scheduler(args, main_optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "etd_loss",
            "entropy_loss",
            "potential_loss",
            "total_loss",
            "source_acc",
            "target_acc",
            "lr",
            "elapsed_sec",
        ],
    )

    if args.source_pretrain_epochs > 0:
        for epoch in range(1, args.source_pretrain_epochs + 1):
            loss = train_source_epoch(model, bundle.source_train, main_optimizer, scaler, device, args)
            print(f"Pretrain {epoch:03d}/{args.source_pretrain_epochs:03d} class_loss={loss:.4f}")

    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(
        f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) "
        f"| target: {args.target} ({bundle.target_train_size})"
    )
    print(
        f"Method: ETD | epsilon: {args.epsilon} | potential_steps: {args.potential_steps} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(
            model,
            bundle.source_train,
            bundle.target_train,
            main_optimizer,
            potential_optimizer,
            scaler,
            device,
            args,
        )
        if scheduler is not None:
            scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]

        elapsed = time.time() - started_at
        lr = main_optimizer.param_groups[0]["lr"]
        logger.log(
            {
                "epoch": epoch,
                "class_loss": train_metrics["class_loss"],
                "etd_loss": train_metrics["etd_loss"],
                "entropy_loss": train_metrics["entropy_loss"],
                "potential_loss": train_metrics["potential_loss"],
                "total_loss": train_metrics["total_loss"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"etd_loss={train_metrics['etd_loss']:.4f} "
            f"entropy={train_metrics['entropy_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, main_optimizer, potential_optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, main_optimizer, potential_optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model,
                main_optimizer,
                potential_optimizer,
                epoch,
                args,
                target_acc,
            )


def train_source_epoch(model, loader, optimizer, scaler, device, args) -> float:
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


def train_one_epoch(
    model: ETD,
    source_loader,
    target_loader,
    main_optimizer,
    potential_optimizer,
    scaler,
    device,
    args,
) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    etd_losses = AverageMeter()
    entropy_losses = AverageMeter()
    potential_losses = AverageMeter()
    total_losses = AverageMeter()
    target_iter = cycle(target_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        with torch.no_grad():
            source_logits, target_logits, source_features, target_features = model(source_images, target_images)
            detached_cost = attention_transport_cost(source_features, target_features, source_logits, target_logits, model).detach()

        potential_loss_value = train_potentials(
            model,
            source_features.detach().float(),
            target_features.detach().float(),
            detached_cost.float(),
            potential_optimizer,
            args,
        )

        main_optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            source_logits, target_logits, source_features, target_features = model(source_images, target_images)
            class_loss = F.cross_entropy(source_logits, source_labels)
            entropy_loss = target_entropy_loss(target_logits)

        cost = attention_transport_cost(source_features, target_features, source_logits, target_logits, model)
        etd_loss = enhanced_transport_distance(model, source_features.float(), target_features.float(), cost.float(), args.epsilon)
        loss = class_loss + args.etd_loss_weight * etd_loss + args.entropy_loss_weight * entropy_loss

        scaler.scale(loss).backward()
        scaler.step(main_optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        etd_losses.update(etd_loss.item(), batch_size)
        entropy_losses.update(entropy_loss.item(), batch_size)
        potential_losses.update(potential_loss_value, batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "etd_loss": etd_losses.avg,
        "entropy_loss": entropy_losses.avg,
        "potential_loss": potential_losses.avg,
        "total_loss": total_losses.avg,
    }


def train_potentials(
    model: ETD,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    cost: torch.Tensor,
    optimizer,
    args,
) -> float:
    steps = max(args.potential_steps, 0)
    if steps == 0:
        return 0.0

    last_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        distance = enhanced_transport_distance(model, source_features, target_features, cost, args.epsilon)
        loss = -distance
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def attention_transport_cost(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    model: ETD,
) -> torch.Tensor:
    distances = torch.cdist(source_features.float(), target_features.float(), p=2)
    weights = model.attention(source_logits.float(), target_logits.float())
    return distances * weights


def enhanced_transport_distance(
    model: ETD,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    source_potential = model.source_potential(source_features)
    target_potential = model.target_potential(target_features).t()
    logits = ((source_potential + target_potential - cost) / epsilon).clamp(min=-50.0, max=50.0)
    penalty = torch.exp(logits).mean()
    return source_potential.mean() + target_potential.mean() - epsilon * penalty


def target_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: ETD, loader, device) -> dict[str, float]:
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


def main_parameters(model: ETD):
    yield from model.feature_extractor.parameters()
    yield from model.classifier.parameters()
    yield from model.attention.parameters()


def potential_parameters(model: ETD):
    yield from model.source_potential.parameters()
    yield from model.target_potential.parameters()


def build_etd_optimizer(args, parameters, lr: float):
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


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"etd_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, model, main_optimizer, potential_optimizer, epoch: int, args, target_acc: float) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": main_optimizer.state_dict(),
        "potential_optimizer": potential_optimizer.state_dict(),
        "target_acc": target_acc,
        "args": to_serializable_args(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
