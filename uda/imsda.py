"""iMSDA-style partially identifiable latent adaptation for UDA."""

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


class DomainAffineFlow(nn.Module):
    """Component-wise invertible domain transform for the changing latent part."""

    def __init__(self, num_domains: int, changing_dim: int) -> None:
        super().__init__()
        self.shift = nn.Embedding(num_domains, changing_dim)
        self.log_scale = nn.Embedding(num_domains, changing_dim)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.log_scale.weight)

    def inverse(self, changing: torch.Tensor, domain_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift = self.shift(domain_ids)
        log_scale = self.log_scale(domain_ids).clamp(-5.0, 5.0)
        normalized = (changing - shift) * torch.exp(-log_scale)
        log_abs_det = -log_scale.sum(dim=1)
        return normalized, log_abs_det


class iMSDA(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int,
        pretrained: bool,
        latent_dim: int,
        changing_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.feature_extractor, _, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.latent_dim = latent_dim
        self.changing_dim = changing_dim
        self.invariant_dim = latent_dim - changing_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.flow = DomainAffineFlow(num_domains=2, changing_dim=changing_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_features(self, features: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-10.0, 10.0)
        if sample:
            eps = torch.randn_like(mu)
            latent = mu + eps * torch.exp(0.5 * logvar)
        else:
            latent = mu
        return latent, mu, logvar

    def shared_latent(self, latent: torch.Tensor, domain_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        invariant = latent[:, : self.invariant_dim]
        changing = latent[:, self.invariant_dim :]
        normalized_changing, log_abs_det = self.flow.inverse(changing, domain_ids)
        return torch.cat([invariant, normalized_changing], dim=1), normalized_changing, log_abs_det

    def forward_domain(self, images: torch.Tensor, domain_id: int, sample: bool = True) -> dict[str, torch.Tensor]:
        features = self.feature_extractor(images)
        latent, mu, logvar = self.encode_features(features, sample=sample)
        domain_ids = torch.full((images.size(0),), domain_id, dtype=torch.long, device=images.device)
        shared, normalized_changing, log_abs_det = self.shared_latent(latent, domain_ids)
        logits = self.classifier(shared)
        reconstructed_features = self.decoder(latent)
        return {
            "features": features,
            "latent": latent,
            "shared": shared,
            "normalized_changing": normalized_changing,
            "mu": mu,
            "logvar": logvar,
            "log_abs_det": log_abs_det,
            "reconstructed_features": reconstructed_features,
            "logits": logits,
        }

    def predict(self, images: torch.Tensor, domain_id: int) -> torch.Tensor:
        return self.forward_domain(images, domain_id=domain_id, sample=False)["logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="iMSDA baseline for UDA datasets.")
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
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--changing-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--entropy-loss-weight", type=float, default=0.1)
    parser.add_argument("--vae-loss-weight", type=float, default=1e-4)
    parser.add_argument("--kl-beta", type=float, default=1.0)
    parser.add_argument("--flow-kl-weight", type=float, default=0.01)
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
        raise ValueError("iMSDA requires a target training loader.")

    device = torch.device(args.device)
    model = iMSDA(args.arch, bundle.num_classes, args.pretrained, args.latent_dim, args.changing_dim, args.hidden_dim).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "epoch",
            "class_loss",
            "target_entropy_loss",
            "vae_loss",
            "reconstruction_loss",
            "kl_loss",
            "flow_kl_loss",
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
        f"Method: iMSDA | latent_dim: {args.latent_dim} | changing_dim: {args.changing_dim} "
        f"| model: {args.arch} | classes: {bundle.num_classes} | device: {device}"
    )

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizer, scaler, device, args)
        if scheduler is not None:
            scheduler.step()

        source_acc = evaluate(model, bundle.source_eval, device, domain_id=0)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device, domain_id=1)["acc"]

        elapsed = time.time() - started_at
        lr = optimizer.param_groups[0]["lr"]
        logger.log({"epoch": epoch, **train_metrics, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": elapsed})
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"class_loss={train_metrics['class_loss']:.4f} "
            f"target_entropy={train_metrics['target_entropy_loss']:.4f} "
            f"vae={train_metrics['vae_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model: iMSDA, source_loader, target_loader, optimizer, scaler, device, args) -> dict[str, float]:
    model.train()
    class_losses = AverageMeter()
    entropy_losses = AverageMeter()
    vae_losses = AverageMeter()
    reconstruction_losses = AverageMeter()
    kl_losses = AverageMeter()
    flow_kl_losses = AverageMeter()
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
            source = model.forward_domain(source_images, domain_id=0, sample=True)
            target = model.forward_domain(target_images, domain_id=1, sample=True)

            class_loss = F.cross_entropy(source["logits"], source_labels)
            target_entropy_loss = entropy_loss(target["logits"])
            vae_loss, reconstruction_loss, kl_loss, flow_kl_loss = imsda_vae_loss(source, target, args.kl_beta, args.flow_kl_weight)
            loss = class_loss + args.entropy_loss_weight * target_entropy_loss + args.vae_loss_weight * vae_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = source_images.size(0)
        class_losses.update(class_loss.item(), batch_size)
        entropy_losses.update(target_entropy_loss.item(), batch_size)
        vae_losses.update(vae_loss.item(), batch_size)
        reconstruction_losses.update(reconstruction_loss.item(), batch_size)
        kl_losses.update(kl_loss.item(), batch_size)
        flow_kl_losses.update(flow_kl_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)

    return {
        "class_loss": class_losses.avg,
        "target_entropy_loss": entropy_losses.avg,
        "vae_loss": vae_losses.avg,
        "reconstruction_loss": reconstruction_losses.avg,
        "kl_loss": kl_losses.avg,
        "flow_kl_loss": flow_kl_losses.avg,
        "total_loss": total_losses.avg,
    }


def imsda_vae_loss(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor], kl_beta: float, flow_kl_weight: float):
    reconstruction_loss = 0.5 * (
        F.mse_loss(source["reconstructed_features"], source["features"])
        + F.mse_loss(target["reconstructed_features"], target["features"])
    )
    kl_loss = 0.5 * (gaussian_kl(source["mu"], source["logvar"]) + gaussian_kl(target["mu"], target["logvar"]))
    flow_kl_loss = 0.5 * (standard_normal_loss(source["shared"]) + standard_normal_loss(target["shared"]))
    vae_loss = reconstruction_loss + kl_beta * kl_loss + flow_kl_weight * flow_kl_loss
    return vae_loss, reconstruction_loss, kl_loss, flow_kl_loss


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def standard_normal_loss(latent: torch.Tensor) -> torch.Tensor:
    return 0.5 * latent.pow(2).sum(dim=1).mean()


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(probabilities * log_probabilities).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model: iMSDA, loader, device, domain_id: int) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.predict(images, domain_id=domain_id)
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
    if args.latent_dim <= 1:
        raise ValueError("--latent-dim must be greater than 1.")
    if not 1 <= args.changing_dim < args.latent_dim:
        raise ValueError("--changing-dim must be in [1, latent_dim).")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive.")
    for name in ("entropy_loss_weight", "vae_loss_weight", "kl_beta", "flow_kl_weight"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"imsda_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
