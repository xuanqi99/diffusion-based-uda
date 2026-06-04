"""WDGRL baseline for unsupervised domain adaptation."""

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


class DomainCritic(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1)


class WDGRL(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool, critic_hidden_dim: int, critic_dropout: float) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.critic = DomainCritic(feature_dim, critic_hidden_dim, critic_dropout)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(images)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor):
        source_features = self.encode(source_images)
        target_features = self.encode(target_images)
        source_logits = self.classifier(source_features)
        return source_logits, source_features, target_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WDGRL baseline for UDA datasets.")
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
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--wd-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--critic-hidden-dim", type=int, default=1024)
    parser.add_argument("--critic-dropout", type=float, default=0.0)
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
    validate_args(args)
    set_seed(args.seed)

    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    if bundle.target_train is None:
        raise ValueError("WDGRL requires a target training loader.")

    device = torch.device(args.device)
    model = WDGRL(args.arch, bundle.num_classes, args.pretrained, args.critic_hidden_dim, args.critic_dropout).to(device)
    main_optimizer = build_wdgrl_optimizer(args, main_parameters(model), args.lr)
    critic_optimizer = build_wdgrl_optimizer(args, model.critic.parameters(), args.critic_lr or args.lr)
    scheduler = build_scheduler(args, main_optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "wasserstein_loss",
            "critic_loss",
            "gradient_penalty",
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
        f"Method: WDGRL | critic_steps: {args.critic_steps} | gp: {args.gradient_penalty_weight:g} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(
            model,
            bundle.source_train,
            bundle.target_train,
            main_optimizer,
            critic_optimizer,
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
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"wd={train_metrics['wasserstein_loss']:.4f} "
            f"critic={train_metrics['critic_loss']:.4f} "
            f"gp={train_metrics['gradient_penalty']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, main_optimizer, critic_optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, main_optimizer, critic_optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, main_optimizer, critic_optimizer, epoch, args, target_acc)


def train_one_epoch(model: WDGRL, source_loader, target_loader, main_optimizer, critic_optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    wasserstein_losses = AverageMeter()
    critic_losses = AverageMeter()
    gradient_penalties = AverageMeter()
    total_losses = AverageMeter()
    target_iter = cycle(target_loader)

    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        critic_loss_value, gp_value = update_critic(model, source_images, target_images, critic_optimizer, args)

        set_requires_grad(model.critic, False)
        main_optimizer.zero_grad(set_to_none=True)
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            source_logits, source_features, target_features = model(source_images, target_images)
            class_loss = F.cross_entropy(source_logits, source_labels)
            wasserstein_loss = wasserstein_estimate(model.critic, source_features, target_features)
            loss = class_loss + args.wd_loss_weight * wasserstein_loss

        scaler.scale(loss).backward()
        scaler.step(main_optimizer)
        scaler.update()
        set_requires_grad(model.critic, True)

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        wasserstein_losses.update(wasserstein_loss.item(), batch_size)
        critic_losses.update(critic_loss_value, batch_size)
        gradient_penalties.update(gp_value, batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "wasserstein_loss": wasserstein_losses.avg,
        "critic_loss": critic_losses.avg,
        "gradient_penalty": gradient_penalties.avg,
        "total_loss": total_losses.avg,
    }


def update_critic(model: WDGRL, source_images: torch.Tensor, target_images: torch.Tensor, optimizer, args) -> tuple[float, float]:
    set_requires_grad(model.critic, True)
    critic_loss_meter = AverageMeter()
    gradient_penalty_meter = AverageMeter()
    for _ in range(args.critic_steps):
        with torch.no_grad():
            source_features = model.encode(source_images).detach()
            target_features = model.encode(target_images).detach()

        optimizer.zero_grad(set_to_none=True)
        wasserstein = wasserstein_estimate(model.critic, source_features, target_features)
        gradient_penalty = compute_gradient_penalty(model.critic, source_features, target_features)
        critic_loss = -wasserstein + args.gradient_penalty_weight * gradient_penalty
        critic_loss.backward()
        optimizer.step()

        critic_loss_meter.update(critic_loss.item(), source_images.size(0))
        gradient_penalty_meter.update(gradient_penalty.item(), source_images.size(0))
    return critic_loss_meter.avg, gradient_penalty_meter.avg


def wasserstein_estimate(critic: nn.Module, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    return critic(source_features).mean() - critic(target_features).mean()


def compute_gradient_penalty(critic: nn.Module, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    alpha = torch.rand(batch_size, 1, device=source_features.device, dtype=source_features.dtype)
    interpolated = alpha * source_features + (1.0 - alpha) * target_features
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    slopes = gradients.norm(2, dim=1)
    return (slopes - 1.0).pow(2).mean()


@torch.no_grad()
def evaluate(model: WDGRL, loader, device) -> dict[str, float]:
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


def build_wdgrl_optimizer(args, parameters, lr: float):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


def main_parameters(model: WDGRL):
    for name, parameter in model.named_parameters():
        if not name.startswith("critic."):
            yield parameter


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def validate_args(args) -> None:
    if args.critic_steps <= 0:
        raise ValueError("--critic-steps must be positive.")
    if args.critic_hidden_dim <= 0:
        raise ValueError("--critic-hidden-dim must be positive.")
    if not 0.0 <= args.critic_dropout < 1.0:
        raise ValueError("--critic-dropout must be in [0, 1).")
    for name in ("wd_loss_weight", "gradient_penalty_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"wdgrl_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, model, main_optimizer, critic_optimizer, epoch: int, args, target_acc: float) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "main_optimizer": main_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "target_acc": target_acc,
        "args": to_serializable_args(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
