from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from data import DATASET_SPECS, build_data, normalize_dataset_name
    from models import build_model
    from utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args
else:
    from .data import DATASET_SPECS, build_data, normalize_dataset_name
    from .models import build_model
    from .utils import AverageMeter, CSVLogger, accuracy, save_json, set_seed, to_serializable_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERM baseline for UDA datasets.")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Limit batches per epoch for quick runs.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoints when > 0.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-fake-data", action="store_true", help="Run without real images for smoke tests.")
    parser.add_argument("--fake-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    set_seed(args.seed)

    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    device = torch.device(args.device)
    model = build_model(args.arch, bundle.num_classes, pretrained=args.pretrained).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, args.epochs)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)

    logger = CSVLogger(
        output_dir / "metrics.csv",
        ["epoch", "train_loss", "source_acc", "target_acc", "lr", "elapsed_sec"],
    )

    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) | target: {args.target} ({bundle.target_size})")
    print(f"Model: {args.arch} | classes: {bundle.num_classes} | device: {device}")

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        train_loss = train_one_epoch(model, bundle.source_train, optimizer, scaler, device, epoch, args)
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
                "train_loss": train_loss,
                "source_acc": source_acc,
                "target_acc": target_acc,
                "lr": lr,
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={train_loss:.4f} source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )

        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizer, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizer, epoch, args, target_acc)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int, args) -> float:
    model.train()
    losses = AverageMeter()

    for step, (images, labels) in enumerate(loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # ERM uses only labeled source-domain samples for the supervised objective.
        with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), images.size(0))

    return losses.avg


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)

        # Unlabeled target samples use label -1 and are skipped for metrics.
        valid = labels >= 0
        if valid.sum().item() > 0:
            loss = F.cross_entropy(logits[valid], labels[valid])
            batch_acc, count = accuracy(logits, labels)
            losses.update(loss.item(), count)
            acc_meter.update(batch_acc, count)

    if acc_meter.count == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {"loss": losses.avg, "acc": acc_meter.avg}


def build_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def build_scheduler(args, optimizer, epochs: int):
    if args.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))


def make_grad_scaler(enabled: bool, device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path("runs") / f"erm_{args.dataset}_{args.source}_to_{args.target}_{stamp}"
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
