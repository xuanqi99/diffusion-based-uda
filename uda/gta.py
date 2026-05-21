"""GTA baseline from "Generate To Adapt" for unsupervised domain adaptation."""

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
    from erm import autocast_context, build_scheduler, evaluate, make_grad_scaler
    from models import build_feature_model
    from utils import AverageMeter, CSVLogger, save_json, set_seed, to_serializable_args
else:
    from .data import DATASET_SPECS, build_data, normalize_dataset_name
    from .erm import autocast_context, build_scheduler, evaluate, make_grad_scaler
    from .models import build_feature_model
    from .utils import AverageMeter, CSVLogger, save_json, set_seed, to_serializable_args


class ConditioningImageGenerator(nn.Module):
    def __init__(self, feature_dim, num_classes, noise_dim, image_size, channels):
        super().__init__()
        if image_size <= 0 or noise_dim <= 0:
            raise ValueError("image_size and noise_dim must be positive.")
        self.num_classes = num_classes
        self.image_size = image_size
        self.init_size = max(4, image_size // 32)
        self.channels = channels
        input_dim = feature_dim + noise_dim + num_classes + 1
        self.project = nn.Sequential(
            nn.Linear(input_dim, channels * self.init_size * self.init_size),
            nn.ReLU(inplace=True),
        )
        blocks = []
        current_size, current_channels = self.init_size, channels
        while current_size < image_size:
            next_channels = max(16, current_channels // 2)
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1),
                _group_norm(next_channels),
                nn.ReLU(inplace=True),
            ]
            current_size *= 2
            current_channels = next_channels
        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(nn.Conv2d(current_channels, 3, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, features, noise, labels):
        label_input = F.one_hot(labels, num_classes=self.num_classes + 1).float()
        x = torch.cat([features, noise, label_input], dim=1)
        x = self.project(x).view(features.size(0), self.channels, self.init_size, self.init_size)
        x = self.to_rgb(self.blocks(x))
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x


class AuxiliaryImageDiscriminator(nn.Module):
    def __init__(self, num_classes, image_size, channels):
        super().__init__()
        layers = []
        in_channels, current_channels, current_size = 3, channels, image_size
        while current_size > 4:
            layers += [
                nn.Conv2d(in_channels, current_channels, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = current_channels
            current_channels = min(current_channels * 2, 512)
            current_size = max(1, current_size // 2)
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.data_head = nn.Linear(in_channels, 2)
        self.class_head = nn.Linear(in_channels, num_classes)

    def forward(self, images):
        features = self.features(images)
        return self.data_head(features), self.class_head(features)


class GTA(nn.Module):
    def __init__(self, arch, num_classes, pretrained, image_size, noise_dim, generator_channels, discriminator_channels):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.feature_extractor, self.classifier, feature_dim = build_feature_model(arch, num_classes, pretrained)
        self.generator = ConditioningImageGenerator(feature_dim, num_classes, noise_dim, image_size, generator_channels)
        self.discriminator = AuxiliaryImageDiscriminator(num_classes, image_size, discriminator_channels)

    def forward(self, images):
        return self.classifier(self.feature_extractor(images))

    def encode(self, images):
        return self.feature_extractor(images)


def parse_args():
    parser = argparse.ArgumentParser(description="GTA baseline for UDA datasets.")
    for name, default in [
        ("data-root", "data"),
        ("dataset", "officehome"),
        ("source", "Art"),
        ("target", "Clipart"),
        ("source-list", None),
        ("target-list", None),
        ("arch", "resnet50"),
        ("output-dir", None),
    ]:
        parser.add_argument(f"--{name}", default=default)
    for name, default in [
        ("num-classes", None),
        ("image-size", None),
        ("epochs", 20),
        ("batch-size", 32),
        ("eval-batch-size", 64),
        ("noise-dim", 128),
        ("generator-channels", 128),
        ("discriminator-channels", 64),
        ("num-workers", 4),
        ("steps-per-epoch", None),
        ("eval-every", 1),
        ("save-every", 0),
        ("fake-size", 64),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in [
        ("lr", 1e-3),
        ("generator-lr", None),
        ("discriminator-lr", None),
        ("weight-decay", 5e-4),
        ("momentum", 0.9),
        ("generator-adv-loss-weight", 1.0),
        ("generator-class-loss-weight", 1.0),
        ("discriminator-class-loss-weight", 1.0),
        ("source-gta-loss-weight", 0.1),
        ("target-gta-loss-weight", 0.1),
    ]:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    set_seed(args.seed)
    output_dir = make_output_dir(args)
    save_json(output_dir / "config.json", to_serializable_args(args))

    bundle = build_data(args)
    if bundle.target_train is None:
        raise ValueError("GTA requires a target training loader.")
    image_size = args.image_size or DATASET_SPECS[args.dataset].image_size
    device = torch.device(args.device)
    model = GTA(
        args.arch,
        bundle.num_classes,
        args.pretrained,
        image_size,
        args.noise_dim,
        args.generator_channels,
        args.discriminator_channels,
    ).to(device)

    optimizers = {
        "feature_classifier": build_optimizer(args, list(model.feature_extractor.parameters()) + list(model.classifier.parameters()), args.lr),
        "generator": build_optimizer(args, model.generator.parameters(), args.generator_lr or args.lr),
        "discriminator": build_optimizer(args, model.discriminator.parameters(), args.discriminator_lr or args.lr),
    }
    schedulers = {name: build_scheduler(args, optimizer, args.epochs) for name, optimizer in optimizers.items()}
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda", device_type=device.type)
    logger = CSVLogger(
        output_dir / "metrics.csv",
        ["epoch", "class_loss", "source_gta_loss", "target_gta_loss", "generator_loss", "discriminator_loss", "total_loss", "source_acc", "target_acc", "lr", "elapsed_sec"],
    )
    best_target_acc = -math.inf
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset} | source: {args.source} ({bundle.source_size}) | target: {args.target} ({bundle.target_train_size})")
    print(f"Method: GTA | model: {args.arch} | classes: {bundle.num_classes} | image_size: {image_size} | device: {device}")

    for epoch in range(1, args.epochs + 1):
        started_at = time.time()
        metrics = train_one_epoch(model, bundle.source_train, bundle.target_train, optimizers, scaler, device, args)
        for scheduler in schedulers.values():
            if scheduler is not None:
                scheduler.step()
        source_acc = evaluate(model, bundle.source_eval, device)["acc"]
        target_acc = float("nan")
        if bundle.target_eval is not None and epoch % args.eval_every == 0:
            target_acc = evaluate(model, bundle.target_eval, device)["acc"]
        lr = optimizers["feature_classifier"].param_groups[0]["lr"]
        row = {"epoch": epoch, "source_acc": source_acc, "target_acc": target_acc, "lr": lr, "elapsed_sec": time.time() - started_at, **metrics}
        logger.log(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} class_loss={metrics['class_loss']:.4f} "
            f"gta_src={metrics['source_gta_loss']:.4f} gta_tgt={metrics['target_gta_loss']:.4f} "
            f"g_loss={metrics['generator_loss']:.4f} d_loss={metrics['discriminator_loss']:.4f} "
            f"source_acc={source_acc:.4f} target_acc={target_acc:.4f} lr={lr:.6g}"
        )
        save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizers, epoch, args, target_acc)
        if not math.isnan(target_acc) and target_acc > best_target_acc:
            best_target_acc = target_acc
            save_checkpoint(output_dir / "best_target.pt", model, optimizers, epoch, args, target_acc)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, optimizers, epoch, args, target_acc)


def train_one_epoch(model, source_loader, target_loader, optimizers, scaler, device, args):
    meters = {name: AverageMeter() for name in ["class_loss", "source_gta_loss", "target_gta_loss", "generator_loss", "discriminator_loss", "total_loss"]}
    target_iter = cycle(target_loader)
    model.train()
    for step, (source_images, source_labels) in enumerate(source_loader, start=1):
        if args.steps_per_epoch and step > args.steps_per_epoch:
            break
        target_images, _ = next(target_iter)
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        d_loss = discriminator_step(model, source_images, source_labels, target_images, optimizers["discriminator"], scaler, device, args)
        g_loss = generator_step(model, source_images, source_labels, optimizers["generator"], scaler, device, args)
        f_metrics = feature_classifier_step(model, source_images, source_labels, target_images, optimizers["feature_classifier"], scaler, device, args)
        for key, value in {**f_metrics, "generator_loss": g_loss, "discriminator_loss": d_loss}.items():
            meters[key].update(value, source_images.size(0))
    set_requires_grad([model.feature_extractor, model.classifier, model.generator, model.discriminator], True)
    return {key: meter.avg for key, meter in meters.items()}


def discriminator_step(model, source_images, source_labels, target_images, optimizer, scaler, device, args):
    set_requires_grad(model.discriminator, True)
    set_requires_grad([model.feature_extractor, model.classifier, model.generator], False)
    optimizer.zero_grad(set_to_none=True)
    was_training = model.feature_extractor.training
    model.feature_extractor.eval()
    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        with torch.no_grad():
            source_features = model.encode(source_images)
            target_features = model.encode(target_images)
            fake_source = model.generator(source_features, sample_noise(source_images, args), source_labels).detach()
            fake_target_labels = fake_labels(target_images.size(0), model.num_classes, device)
            fake_target = model.generator(target_features, sample_noise(target_images, args), fake_target_labels).detach()
        real_data, real_class = model.discriminator(source_images)
        fake_source_data, _ = model.discriminator(fake_source)
        fake_target_data, _ = model.discriminator(fake_target)
        loss = (
            F.cross_entropy(real_data, domain_labels(source_images.size(0), True, device))
            + 0.5 * F.cross_entropy(fake_source_data, domain_labels(fake_source.size(0), False, device))
            + 0.5 * F.cross_entropy(fake_target_data, domain_labels(fake_target.size(0), False, device))
            + args.discriminator_class_loss_weight * F.cross_entropy(real_class, source_labels)
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.feature_extractor.train(was_training)
    return loss.item()


def generator_step(model, source_images, source_labels, optimizer, scaler, device, args):
    set_requires_grad(model.generator, True)
    set_requires_grad([model.feature_extractor, model.classifier, model.discriminator], False)
    optimizer.zero_grad(set_to_none=True)
    was_training = model.feature_extractor.training
    model.feature_extractor.eval()
    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        with torch.no_grad():
            features = model.encode(source_images)
        fake_source = model.generator(features, sample_noise(source_images, args), source_labels)
        data_logits, class_logits = model.discriminator(fake_source)
        loss = (
            args.generator_adv_loss_weight * F.cross_entropy(data_logits, domain_labels(source_images.size(0), True, device))
            + args.generator_class_loss_weight * F.cross_entropy(class_logits, source_labels)
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    model.feature_extractor.train(was_training)
    return loss.item()


def feature_classifier_step(model, source_images, source_labels, target_images, optimizer, scaler, device, args):
    set_requires_grad([model.feature_extractor, model.classifier], True)
    set_requires_grad([model.generator, model.discriminator], False)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(enabled=args.amp and device.type == "cuda", device_type=device.type):
        source_features = model.encode(source_images)
        target_features = model.encode(target_images)
        class_loss = F.cross_entropy(model.classifier(source_features), source_labels)
        fake_source = model.generator(source_features, sample_noise(source_images, args), source_labels)
        fake_target_labels = fake_labels(target_images.size(0), model.num_classes, device)
        fake_target = model.generator(target_features, sample_noise(target_images, args), fake_target_labels)
        _, source_class_logits = model.discriminator(fake_source)
        target_data_logits, _ = model.discriminator(fake_target)
        source_gta_loss = F.cross_entropy(source_class_logits, source_labels)
        target_gta_loss = F.cross_entropy(target_data_logits, domain_labels(target_images.size(0), True, device))
        total_loss = class_loss + args.source_gta_loss_weight * source_gta_loss + args.target_gta_loss_weight * target_gta_loss
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {
        "class_loss": class_loss.item(),
        "source_gta_loss": source_gta_loss.item(),
        "target_gta_loss": target_gta_loss.item(),
        "total_loss": total_loss.item(),
    }


def build_optimizer(args, parameters, lr):
    params = list(parameters)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


def sample_noise(images, args):
    return torch.randn(images.size(0), args.noise_dim, device=images.device)


def fake_labels(batch_size, num_classes, device):
    return torch.full((batch_size,), num_classes, dtype=torch.long, device=device)


def domain_labels(batch_size, real, device):
    return torch.full((batch_size,), 1 if real else 0, dtype=torch.long, device=device)


def set_requires_grad(modules, requires_grad):
    if isinstance(modules, nn.Module):
        modules = [modules]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)


def _group_norm(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def make_output_dir(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"gta_{args.dataset}_{args.source}_to_{args.target}_{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path, model, optimizers, epoch, args, target_acc):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "target_acc": target_acc,
            "args": to_serializable_args(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
