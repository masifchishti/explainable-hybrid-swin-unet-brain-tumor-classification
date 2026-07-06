from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torchvision import models

from config import MODEL_PATH


class ResidualRefinement(nn.Module):
    """Exact residual refinement block used in the Kaggle notebook."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DecoderBlock(nn.Module):
    """Exact decoder block used in the Kaggle notebook."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                out_ch + skip_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = ResidualRefinement(out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.refine(x)
        return x


class HybridSwinUnet(nn.Module):
    """Exact Hybrid Swin-UNet architecture from the Kaggle notebook."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        # weights=None prevents an unnecessary internet download.
        # The complete trained weights are loaded from best_model.pth.
        swin = models.swin_t(weights=None)
        self.features = swin.features

        self.dec3 = DecoderBlock(768, 384, 384)
        self.dec2 = DecoderBlock(384, 192, 192)
        self.dec1 = DecoderBlock(192, 96, 96)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features

        x0 = f[0](x)
        s1 = f[1](x0).permute(0, 3, 1, 2).contiguous()

        x2 = f[2](f[1](x0))
        s2 = f[3](x2).permute(0, 3, 1, 2).contiguous()

        x4 = f[4](f[3](x2))
        s3 = f[5](x4).permute(0, 3, 1, 2).contiguous()

        x6 = f[6](f[5](x4))
        s4 = f[7](x6).permute(0, 3, 1, 2).contiguous()

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.gap(d1)
        out = self.classifier(out)
        return out


def _load_checkpoint_file(
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(checkpoint_path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "The checkpoint is not a dictionary or state_dict."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise RuntimeError("No valid model state_dict was found.")

    # Supports checkpoints saved through DataParallel.
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {
            str(key)[7:]: value
            for key, value in state_dict.items()
        }

    return state_dict


def _validate_checkpoint_shapes(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    expected = model.state_dict()

    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = []

    for key in set(expected).intersection(state_dict):
        if tuple(expected[key].shape) != tuple(state_dict[key].shape):
            mismatched.append(
                f"{key}: checkpoint={tuple(state_dict[key].shape)}, "
                f"model={tuple(expected[key].shape)}"
            )

    if missing or unexpected or mismatched:
        parts = [
            "The checkpoint does not exactly match the Kaggle model architecture."
        ]
        if missing:
            parts.append(f"Missing keys ({len(missing)}): {missing[:12]}")
        if unexpected:
            parts.append(
                f"Unexpected keys ({len(unexpected)}): {unexpected[:12]}"
            )
        if mismatched:
            parts.append(
                f"Shape mismatches ({len(mismatched)}): {mismatched[:12]}"
            )

        raise RuntimeError("\n".join(parts))


def load_model(
    checkpoint_path: str | Path = MODEL_PATH,
) -> tuple[HybridSwinUnet, torch.device]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place best_model.pth inside the models folder."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = HybridSwinUnet(num_classes=4)
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    state_dict = _extract_state_dict(checkpoint)

    # Never use strict=False for this research model.
    _validate_checkpoint_shapes(model, state_dict)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, device
