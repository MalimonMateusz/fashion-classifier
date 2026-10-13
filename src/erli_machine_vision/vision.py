"""Wspolny preprocessing obrazow (OpenCV + numpy).

Ta sama funkcja jest uzywana przy ekstrakcji cech w PyTorch i przy inferencji
przez cv2.dnn, wiec model w produkcji dostaje dokladnie takie wejscie jak przy
treningu (bez rozjazdow torchvision vs OpenCV w resize/crop/normalizacji).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

INTERPOLATIONS = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


@dataclass(frozen=True)
class PreprocessConfig:
    input_size: int  # bok kwadratowego wejscia sieci
    crop_pct: float  # resize krotszego boku do input_size / crop_pct, potem center crop
    mean: tuple[float, float, float]  # RGB, skala 0-1
    std: tuple[float, float, float]
    interpolation: str = "bicubic"

    def to_dict(self) -> dict:
        return asdict(self)


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cv2 nie odczytal obrazu: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_center_crop(rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    scale_size = int(round(cfg.input_size / cfg.crop_pct))
    h, w = rgb.shape[:2]
    scale = scale_size / min(h, w)
    new_w, new_h = max(scale_size, round(w * scale)), max(scale_size, round(h * scale))
    resized = cv2.resize(
        rgb, (new_w, new_h), interpolation=INTERPOLATIONS[cfg.interpolation]
    )
    top = (new_h - cfg.input_size) // 2
    left = (new_w - cfg.input_size) // 2
    return resized[top : top + cfg.input_size, left : left + cfg.input_size]


def preprocess(rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """RGB uint8 HxWx3 -> float32 3xSxS znormalizowany tensor (bez wymiaru batcha)."""
    crop = resize_center_crop(rgb, cfg).astype(np.float32) / 255.0
    crop = (crop - np.array(cfg.mean, dtype=np.float32)) / np.array(
        cfg.std, dtype=np.float32
    )
    return crop.transpose(2, 0, 1)
