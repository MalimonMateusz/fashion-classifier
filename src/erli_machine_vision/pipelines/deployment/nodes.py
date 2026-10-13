"""Eksport najlepszego modelu do ONNX i weryfikacja inferencji przez OpenCV (cv2.dnn).

Eksportowany graf zawiera caly model end-to-end:
    obraz po `vision.preprocess` -> backbone -> L2-normalizacja -> warstwa liniowa -> softmax
wiec w produkcji wystarczy OpenCV + plik .onnx + model_card.json (klasy i preprocessing).
"""

from __future__ import annotations

import io
import logging
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from erli_machine_vision.backbones import load_backbone
from erli_machine_vision.model_zoo import MODELS
from erli_machine_vision.vision import PreprocessConfig, preprocess, read_rgb

log = logging.getLogger(__name__)


class EndToEndClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, weight: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.head.weight.copy_(
                torch.from_numpy(np.asarray(weight, dtype=np.float32))
            )
            self.head.bias.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return torch.softmax(self.head(features), dim=-1)


def export_onnx(
    best: dict, backbones: dict, params: dict, *models
) -> tuple[bytes, dict]:
    name = best["model"]
    model = models[list(MODELS).index(name)]
    if not hasattr(model, "linear_head"):
        raise ValueError(
            f"Model {name} nie ma glowy liniowej - nie da sie go wyeksportowac do ONNX"
        )
    backbone_name = MODELS[name]
    encoder, preprocess_cfg = load_backbone(backbones[backbone_name])
    weight, bias = model.linear_head()
    net = EndToEndClassifier(encoder, weight, bias).eval()

    dummy = torch.zeros(1, 3, preprocess_cfg.input_size, preprocess_cfg.input_size)
    buffer = io.BytesIO()
    torch.onnx.export(
        net,
        (dummy,),
        buffer,
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=params["opset"],
        dynamo=False,
    )
    onnx_bytes = buffer.getvalue()
    log.info("Wyeksportowano %s do ONNX (%.1f MB)", name, len(onnx_bytes) / 2**20)

    model_card = {
        "model": name,
        "backbone": backbones[backbone_name],
        "classes": list(model.classes_),
        "preprocess": preprocess_cfg.to_dict(),
        "input": "RGB, preprocess() z erli_machine_vision.vision, NCHW float32",
        "output": "prawdopodobienstwa klas (softmax), kolejnosc jak w `classes`",
        "metrics": {k: v for k, v in best.items() if k != "model"},
    }
    return onnx_bytes, model_card


def verify_cv2_dnn(
    onnx_bytes: bytes,
    model_card: dict,
    dataset: pd.DataFrame,
    params: dict,
    *all_predictions,
) -> dict:
    """Puszcza test przez cv2.dnn i porownuje z predykcjami z pipeline'u (sklearn na embeddingach)."""
    net = cv2.dnn.readNetFromONNX(np.frombuffer(onnx_bytes, dtype=np.uint8))
    cfg = PreprocessConfig(
        **{
            **model_card["preprocess"],
            "mean": tuple(model_card["preprocess"]["mean"]),
            "std": tuple(model_card["preprocess"]["std"]),
        }
    )
    classes = np.array(model_card["classes"])

    reference = next(
        p for p in all_predictions if p["model"].iloc[0] == model_card["model"]
    )
    reference = reference[reference["split"] == "test"].merge(
        dataset[["product_id", "image_path"]], on="product_id"
    )
    if params.get("max_images"):
        reference = reference.head(params["max_images"])

    preds, max_abs_diff, timings = [], 0.0, []
    for row in reference.itertuples():
        blob = preprocess(read_rgb(row.image_path), cfg)[None]
        start = time.perf_counter()
        net.setInput(blob)
        proba = net.forward()[0]
        timings.append(time.perf_counter() - start)
        preds.append(classes[proba.argmax()])
        max_abs_diff = max(max_abs_diff, abs(float(proba.max()) - row.confidence))
    preds = np.array(preds)

    result = {
        "opencv_version": cv2.__version__,
        "n_images": len(reference),
        "cv2_test_accuracy": float((preds == reference["label"].to_numpy()).mean()),
        "agreement_with_pipeline": float(
            (preds == reference["pred"].to_numpy()).mean()
        ),
        "max_abs_confidence_diff": max_abs_diff,
        "latency_ms_median": 1000 * float(np.median(timings)),
    }
    log.info("Weryfikacja cv2.dnn: %s", result)
    if result["agreement_with_pipeline"] < params["min_agreement"]:
        raise ValueError(f"cv2.dnn rozjezdza sie z pipeline'em: {result}")
    return result
