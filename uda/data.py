from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    default_num_classes: int
    domains: dict[str, tuple[str, ...]]
    image_size: int = 224


DATASET_SPECS: dict[str, DatasetSpec] = {
    "officehome": DatasetSpec(
        name="officehome",
        default_num_classes=65,
        domains={
            "art": ("Art", "art", "A", "a"),
            "clipart": ("Clipart", "clipart", "C", "c"),
            "product": ("Product", "product", "P", "p"),
            "real_world": ("Real_World", "RealWorld", "Real World", "real_world", "real", "R", "r"),
        },
    ),
    "office31": DatasetSpec(
        name="office31",
        default_num_classes=31,
        domains={
            "amazon": ("amazon", "Amazon", "A", "a"),
            "dslr": ("dslr", "DSLR", "D", "d"),
            "webcam": ("webcam", "Webcam", "W", "w"),
        },
    ),
    "visda2017": DatasetSpec(
        name="visda2017",
        default_num_classes=12,
        domains={
            "train": ("train", "Train", "synthetic", "Synthetic", "source", "Source"),
            "validation": ("validation", "Validation", "val", "Val", "real", "Real", "target", "Target"),
        },
    ),
    "domainnet": DatasetSpec(
        name="domainnet",
        default_num_classes=345,
        domains={
            "clipart": ("clipart", "Clipart"),
            "infograph": ("infograph", "Infograph"),
            "painting": ("painting", "Painting"),
            "quickdraw": ("quickdraw", "Quickdraw", "quick_draw", "Quick_Draw"),
            "real": ("real", "Real"),
            "sketch": ("sketch", "Sketch"),
        },
    ),
}


@dataclass
class DataBundle:
    source_train: DataLoader
    source_eval: DataLoader
    target_train: DataLoader | None
    target_eval: DataLoader | None
    num_classes: int
    source_size: int
    target_train_size: int
    target_size: int
    class_to_idx: dict[str, int] | None = None


class ImageListDataset(Dataset):
    """Dataset for common UDA list files: "relative/path.jpg label" per line."""

    def __init__(
        self,
        list_file: str | Path,
        root: str | Path,
        transform: Callable | None = None,
        require_labels: bool = False,
    ) -> None:
        self.list_file = Path(list_file)
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        if not self.list_file.is_file():
            raise FileNotFoundError(f"List file not found: {self.list_file}")

        for line_no, raw_line in enumerate(self.list_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            image_path, label = self._parse_line(line, line_no)
            if require_labels and label < 0:
                raise ValueError(f"Missing label in {self.list_file}:{line_no}")
            full_path = image_path if image_path.is_absolute() else self.root / image_path
            self.samples.append((full_path, label))

        if not self.samples:
            raise ValueError(f"No samples found in list file: {self.list_file}")

    @staticmethod
    def _parse_line(line: str, line_no: int) -> tuple[Path, int]:
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            image_name, maybe_label = parts
            try:
                return Path(image_name), int(maybe_label)
            except ValueError:
                pass

        suffix = Path(line).suffix.lower()
        if suffix not in IMG_EXTENSIONS:
            raise ValueError(f"Cannot parse list entry at line {line_no}: {line}")
        return Path(line), -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class RemappedImageFolder(Dataset):
    """ImageFolder wrapper that keeps target labels aligned with source classes."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        class_to_idx: dict[str, int] | None = None,
    ) -> None:
        from torchvision.datasets import ImageFolder

        self.root = Path(root)
        base = ImageFolder(str(self.root), transform=transform)
        if class_to_idx is None:
            self.dataset = base
            self.class_to_idx = dict(base.class_to_idx)
            self.classes = list(base.classes)
            return

        missing = sorted(set(base.class_to_idx) - set(class_to_idx))
        if missing:
            raise ValueError(
                f"Target domain {self.root} has classes not present in source: {missing[:10]}"
            )

        label_map = {old_idx: class_to_idx[name] for name, old_idx in base.class_to_idx.items()}
        base.samples = [(path, label_map[label]) for path, label in base.samples]
        base.imgs = base.samples
        self.dataset = base
        self.class_to_idx = dict(class_to_idx)
        self.classes = list(class_to_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.dataset[index]


class FakeDomainDataset(Dataset):
    """Small deterministic dataset used for quick checks without downloading data."""

    def __init__(self, size: int, num_classes: int, image_size: int, seed_offset: int = 0) -> None:
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(index + self.seed_offset)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        label = index % self.num_classes
        return image, label


def build_transforms(image_size: int):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def build_data(args) -> DataBundle:
    dataset_key = normalize_dataset_name(args.dataset)
    spec = DATASET_SPECS[dataset_key]
    image_size = args.image_size or spec.image_size

    if args.use_fake_data:
        return _build_fake_data(args, spec.default_num_classes, image_size)

    train_transform, eval_transform = build_transforms(image_size)
    data_root = Path(args.data_root)

    if args.source_list:
        source_train = ImageListDataset(args.source_list, data_root, train_transform, require_labels=True)
        source_eval = ImageListDataset(args.source_list, data_root, eval_transform, require_labels=True)
        class_to_idx = None
        inferred_classes = _infer_num_classes(source_train)
    else:
        source_root = resolve_domain_root(data_root, spec, args.source)
        source_train = RemappedImageFolder(source_root, transform=train_transform)
        source_eval = RemappedImageFolder(source_root, transform=eval_transform)
        class_to_idx = source_train.class_to_idx
        inferred_classes = len(class_to_idx)

    num_classes = args.num_classes or inferred_classes or spec.default_num_classes
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    if args.target_list:
        target_train: Dataset | None = ImageListDataset(args.target_list, data_root, train_transform, require_labels=False)
        target_eval: Dataset | None = ImageListDataset(args.target_list, data_root, eval_transform, require_labels=False)
    else:
        target_root = resolve_domain_root(data_root, spec, args.target)
        target_train = RemappedImageFolder(target_root, transform=train_transform, class_to_idx=class_to_idx)
        target_eval = RemappedImageFolder(target_root, transform=eval_transform, class_to_idx=class_to_idx)

    source_train_loader = _make_loader(
        source_train,
        args.batch_size,
        shuffle=True,
        drop_last=len(source_train) >= args.batch_size,
        args=args,
    )
    source_eval_loader = _make_loader(source_eval, args.eval_batch_size, shuffle=False, drop_last=False, args=args)
    target_train_loader = _make_loader(
        target_train,
        args.batch_size,
        shuffle=True,
        drop_last=len(target_train) >= args.batch_size,
        args=args,
    )
    target_eval_loader = _make_loader(target_eval, args.eval_batch_size, shuffle=False, drop_last=False, args=args)

    return DataBundle(
        source_train=source_train_loader,
        source_eval=source_eval_loader,
        target_train=target_train_loader,
        target_eval=target_eval_loader,
        num_classes=num_classes,
        source_size=len(source_train),
        target_train_size=len(target_train),
        target_size=len(target_eval),
        class_to_idx=class_to_idx,
    )


def normalize_dataset_name(name: str) -> str:
    normalized = _compact(name)
    aliases = {
        "officehome": "officehome",
        "office31": "office31",
        "visda": "visda2017",
        "visda2017": "visda2017",
        "domainnet": "domainnet",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: {', '.join(DATASET_SPECS)}")
    return aliases[normalized]


def resolve_domain_root(data_root: Path, spec: DatasetSpec, domain: str) -> Path:
    domain_names = _domain_candidates(spec, domain)
    dataset_names = _dataset_root_candidates(spec)
    candidates = [root / name for root in _root_candidates(data_root, dataset_names) for name in domain_names]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    preview = "\n".join(str(path) for path in candidates[:12])
    raise FileNotFoundError(
        f"Could not find domain '{domain}' for dataset '{spec.name}'. Tried:\n{preview}"
    )


def _build_fake_data(args, default_num_classes: int, image_size: int) -> DataBundle:
    num_classes = args.num_classes or min(default_num_classes, 10)
    source = FakeDomainDataset(size=args.fake_size, num_classes=num_classes, image_size=image_size, seed_offset=0)
    target = FakeDomainDataset(size=args.fake_size, num_classes=num_classes, image_size=image_size, seed_offset=10000)
    return DataBundle(
        source_train=_make_loader(source, args.batch_size, shuffle=True, drop_last=False, args=args),
        source_eval=_make_loader(source, args.eval_batch_size, shuffle=False, drop_last=False, args=args),
        target_train=_make_loader(target, args.batch_size, shuffle=True, drop_last=False, args=args),
        target_eval=_make_loader(target, args.eval_batch_size, shuffle=False, drop_last=False, args=args),
        num_classes=num_classes,
        source_size=len(source),
        target_train_size=len(target),
        target_size=len(target),
        class_to_idx=None,
    )


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool, args) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=drop_last,
        generator=generator if shuffle else None,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )


def _infer_num_classes(dataset: ImageListDataset) -> int | None:
    labels = [label for _, label in dataset.samples if label >= 0]
    if not labels:
        return None
    return max(labels) + 1


def _domain_candidates(spec: DatasetSpec, domain: str) -> list[str]:
    compact_domain = _compact(domain)
    selected: Iterable[str] = (domain,)
    for canonical, variants in spec.domains.items():
        compact_variants = {_compact(canonical), *(_compact(item) for item in variants)}
        if compact_domain in compact_variants:
            selected = variants
            break
    return _unique([domain, *selected])


def _dataset_root_candidates(spec: DatasetSpec) -> list[str]:
    names = {
        spec.name,
        spec.name.lower(),
        spec.name.upper(),
        spec.name.replace("2017", ""),
        "OfficeHome" if spec.name == "officehome" else spec.name,
        "Office31" if spec.name == "office31" else spec.name,
        "VisDA2017" if spec.name == "visda2017" else spec.name,
        "DomainNet" if spec.name == "domainnet" else spec.name,
    }
    return _unique(names)


def _root_candidates(data_root: Path, dataset_names: Iterable[str]) -> list[Path]:
    # The CLI accepts either a global data root or the dataset root itself.
    candidates = [data_root]
    candidates.extend(data_root / name for name in dataset_names)
    return _unique(candidates)


def _compact(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(" ", "")


def _unique(items: Iterable) -> list:
    seen = set()
    output = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
