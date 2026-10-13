"""Pretrenowane backbone'y jako ekstraktory cech (CLIP przez open_clip, CNN przez timm).

Kazdy backbone opisuje slownik z parameters_feature_extraction.yml, np.
    {"library": "open_clip", "model": "ViT-B-32", "pretrained": "openai"}
    {"library": "timm", "model": "convnext_tiny.fb_in22k_ft_in1k"}
"""

from __future__ import annotations

import torch
from torch import nn

from .vision import PreprocessConfig


class ImageEncoder(nn.Module):
    """Obraz (B, 3, S, S) po `vision.preprocess` -> wektor cech (B, D).

    Dla CLIP opakowuje wieze wizualna (`model.visual`, z projekcja do przestrzeni
    wspolnej z tekstem), dla timm - model bez glowy klasyfikacyjnej.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _clip_preprocess_config(model) -> PreprocessConfig:
    cfg = model.visual.preprocess_cfg
    size = cfg["size"] if isinstance(cfg["size"], int) else cfg["size"][0]
    return PreprocessConfig(
        input_size=size,
        crop_pct=1.0,  # CLIP: resize krotszego boku do rozmiaru wejscia + center crop
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        interpolation=cfg.get("interpolation", "bicubic"),
    )


def load_backbone(spec: dict) -> tuple[ImageEncoder, PreprocessConfig]:
    if spec["library"] == "open_clip":
        import open_clip  # noqa: PLC0415 - ciezki import tylko gdy potrzebny

        model, _, _ = open_clip.create_model_and_transforms(
            spec["model"], pretrained=spec["pretrained"]
        )
        encoder = ImageEncoder(model.visual)
        preprocess_cfg = _clip_preprocess_config(model)
    elif spec["library"] == "timm":
        import timm  # noqa: PLC0415

        model = timm.create_model(spec["model"], pretrained=True, num_classes=0)
        encoder = ImageEncoder(model)
        pc = model.pretrained_cfg
        preprocess_cfg = PreprocessConfig(
            input_size=pc["input_size"][-1],
            crop_pct=pc["crop_pct"],
            mean=tuple(pc["mean"]),
            std=tuple(pc["std"]),
            interpolation=pc["interpolation"],
        )
    else:
        raise ValueError(f"Nieznana biblioteka backbone'u: {spec['library']}")
    return encoder.eval(), preprocess_cfg


def load_clip_text_encoder(spec: dict):
    """Zwraca (model CLIP, tokenizer) - do klasyfikacji zero-shot."""
    import open_clip  # noqa: PLC0415

    model, _, _ = open_clip.create_model_and_transforms(
        spec["model"], pretrained=spec["pretrained"]
    )
    return model.eval(), open_clip.get_tokenizer(spec["model"])
