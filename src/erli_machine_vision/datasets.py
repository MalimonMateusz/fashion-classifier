"""Wlasne datasety Kedro."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class OnnxModelDataset(AbstractDataset[bytes, bytes]):
    """Model ONNX trzymany jako surowe bajty.

    Bajty mozna podac wprost do `cv2.dnn.readNetFromONNX` i `onnxruntime.InferenceSession`,
    wiec nie potrzebujemy tu zaleznosci od pakietu `onnx`.
    """

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> bytes:
        return self._filepath.read_bytes()

    def _save(self, data: bytes) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filepath.write_bytes(data)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}
