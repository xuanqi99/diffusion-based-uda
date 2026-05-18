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
    from dann import compute_grl_lambda, gradient_reverse
    from erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import DATASET_SPECS, build_data, normalize_dataset_name
    from .dann import compute_grl_lambda, gradient_reverse
    from .erm import autocast_context, build_optimizer, build_scheduler, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


class MultiLinearMap(nn.Module):
    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        conditioned = torch.bmm(probabilities.unsqueeze(2), features.unsqueeze(1))
        return conditioned.view(batch_size, -1)


class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.output_dim = output_dim
        self.register_buffer("feature_map", torch.randn(feature_dim, output_dim))
        self.register_buffer("probability_map", torch.randn(num_classes, output_dim))

    def forward(self, features: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        projected_features = features @ self.feature_map
        projected_probabilities = probabilities @ self.probability_map
        return (projected_features * projected_probabilities) / math.sqrt(self.output_dim)


class CDAN(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool,
        hidden_dim: int,
        dropout: float,
        conditioning: str,
        randomized_dim: int,
    ) -> None:
        super().__init__()
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(
            arch=arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        if conditioning == "multilinear":
            self.conditioning_layer = MultiLinearMap()
            domain_input_dim = feature_dim * num_classes
        elif conditioning == "randomized":
            self.conditioning_layer = RandomizedMultiLinearMap(feature_dim, num_classes, randomized_dim)
            domain_input_dim = randomized_dim
        else:
            raise ValueError(f"Unsupported CDAN conditioning mode: {conditioning}")

        self.domain_discriminator = nn.Sequential(
            nn.Linear(domain_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.classifier(features)

    def forward(self, source_images: torch.Tensor, target_images: torch.Tensor, grl_lambda: float):
        source_features = self.feature_extractor(source_images)
        target_features = self.feature_extractor(target_images)

        source_logits = self.classifier(source_features)
        target_logits = self.classifier(target_features)
        features = torch.cat([source_features, target_features], dim=0)
        logits = torch.cat([source_logits, target_logits], dim=0)
        probabilities = F.softmax(logits, dim=1)

        # CDAN conditions domain confusion on both visual features and class predictions.
        conditional_features = self.conditioning_layer(features, probabilities)
        domain_logits = self.domain_discriminator(gradient_reverse(conditional_features, grl_lambda))
        return source_logits, domain_logits, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDAN baseline for UDA datasets.")
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
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-schedule", choices=("dann", "none"), default="dann")
    parser.add_argument("--conditioning", choices=("randomized", "multilinear"), default="randomized")
    parser.add_argument("--randomized-dim", type=int, default=1024)
    parser.add_argument("--entropy-conditioning", action="store_true", help="Use CDAN+E entropy-conditioned weights.")
    parser.add_argument("--entropy-eps", type=float, default=1e-5)
    parser.add_argument("--domain-hidden-dim", type=int, default=1024)
    parser.add_argument("--domain-dropout", type=float, default=0.5)
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
        raise ValueError("CDAN requires a target training loader.")

    device = torch.device(args.device)
    model = CDAN(
        arch=args.arch,
        num_classes=bundle.num_classes,
        pretrained=args.pretrained,
        hidden_dim=args.domain_hidden_dim,
        dropout=args.domain_dropout,
        conditioning=args.conditioning,
        randomized_dim=args.randomized_dim,
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
            "total_loss",
            "prediction_entropy",
            "source_acc",
            "target_acc",
            "grl_lambda",
            "lr",
            "elapsed_sec",
        ],
    )

    best_target_acc = -math.inf
    method_name = "CDAN+E" if args.entropy_conditioning else "CDAN"
    print(f"Output directory: {output_dir}")
    print(
        f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) "
        f"| target: {args.target} ({bundle.target_train_size})"
    )
    print(
        f"Method: {method_name} | conditioning: {args.conditioning} | model: {args.arch} "
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
        logger.log(
            {
                "epoch": epoch,
                "class_loss": train_metrics["class_loss"],
                "domain_loss": train_metrics["domain_loss"],
                "total_loss": train_metrics["total_loss"],
                "prediction_entropy": train_metrics["prediction_entropy"],
                "source_acc": source_acc,
                "target_acc": target_acc,
                "grl_lambda": train_metrics["grl_lambda"],
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"domain_loss={train_metrics['domain_loss']:.4f} "
            f"entropy={train_metrics['prediction_entropy']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} "
            f"lambda={train_metrics['grl_lambda']:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model, source_loader, target_loader, optimizer, scaler, device, epoch: int, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    domain_losses = AverageMeter()
    total_losses = AverageMeter()
    entropy_meter = AverageMeter()
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
            class_logits, domain_logits, probabilities = model(source_images, target_images, grl_lambda)
            class_loss = F.cross_entropy(class_logits, source_labels)

            domain_labels = torch.cat(
                [
                    torch.zeros(source_images.size(0), dtype=torch.long, device=device),
                    torch.ones(target_images.size(0), dtype=torch.long, device=device),
                ],
                dim=0,
            )
            per_sample_domain_loss = F.cross_entropy(domain_logits, domain_labels, reduction="none")
            if args.entropy_conditioning:
                weights = compute_entropy_weights(probabilities, source_images.size(0), args.entropy_eps)
                domain_loss = (per_sample_domain_loss * weights).sum() / weights.sum().clamp_min(args.entropy_eps)
            else:
                domain_loss = per_sample_domain_loss.mean()

            loss = class_loss + args.domain_loss_weight * domain_loss
            entropy = prediction_entropy(probabilities, args.entropy_eps).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        domain_losses.update(domain_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)
        entropy_meter.update(entropy.item(), batch_size)
        lambda_meter.update(grl_lambda, batch_size)

    return {
        "class_loss": class_losses.avg,
        "domain_loss": domain_losses.avg,
        "total_loss": total_losses.avg,
        "prediction_entropy": entropy_meter.avg,
        "grl_lambda": lambda_meter.avg,
    }


def prediction_entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    return -(probabilities * (probabilities + eps).log()).sum(dim=1)


def compute_entropy_weights(probabilities: torch.Tensor, source_count: int, eps: float) -> torch.Tensor:
    entropy = prediction_entropy(probabilities, eps)
    weights = 1.0 + torch.exp(-entropy)
    source_weights = weights[:source_count]
    target_weights = weights[source_count:]
    source_weights = source_weights / source_weights.mean().clamp_min(eps).detach()
    target_weights = target_weights / target_weights.mean().clamp_min(eps).detach()
    return torch.cat([source_weights, target_weights], dim=0).detach()


@torch.no_grad()
def evaluate(model: CDAN, loader, device) -> dict[str, float]:
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


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "cdane" if args.entropy_conditioning else "cdan"
        output_dir = Path("runs") / f"{suffix}_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
