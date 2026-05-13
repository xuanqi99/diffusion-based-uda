from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Lightweight classifier for smoke tests and small custom datasets."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    arch = arch.lower()
    if arch == "small_cnn":
        return SmallCNN(num_classes)

    from torchvision import models

    if not hasattr(models, arch):
        raise ValueError(f"Unknown model architecture '{arch}'.")

    model_fn = getattr(models, arch)
    model = _build_torchvision_model(model_fn, arch, pretrained)
    return _replace_classifier(model, num_classes)


def _build_torchvision_model(model_fn, arch: str, pretrained: bool) -> nn.Module:
    if not pretrained:
        try:
            return model_fn(weights=None)
        except TypeError:
            return model_fn(pretrained=False)

    from torchvision import models

    # Torchvision changed from pretrained=True to explicit weight enums.
    if hasattr(models, "get_model_weights"):
        try:
            weights = models.get_model_weights(arch).DEFAULT
            return model_fn(weights=weights)
        except Exception:
            pass
    try:
        return model_fn(pretrained=True)
    except TypeError:
        return model_fn(weights="DEFAULT")


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return model
        if isinstance(classifier, nn.Sequential):
            for idx in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[idx], nn.Linear):
                    classifier[idx] = nn.Linear(classifier[idx].in_features, num_classes)
                    return model
    raise ValueError("Could not locate a replaceable classifier layer.")
