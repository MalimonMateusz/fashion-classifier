"""Ekstrakcja cech: klasyczne cechy OpenCV (HOG + histogram koloru) i embeddingi z backbone'ow.

Cechy liczymy raz i cache'ujemy w data/04_feature - trening klasyfikatorow
na gotowych wektorach trwa potem sekundy, nawet na CPU.
Format wyjscia: {"product_id": ndarray[str], "X": ndarray[float32] (N, D)}.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

from erli_machine_vision.backbones import load_backbone, load_clip_text_encoder
from erli_machine_vision.vision import preprocess, read_rgb

log = logging.getLogger(__name__)


def _handcrafted(path: str, params: dict) -> np.ndarray:
    size = params["image_size"]
    bgr = cv2.resize(
        cv2.imread(path, cv2.IMREAD_COLOR), (size, size), interpolation=cv2.INTER_AREA
    )

    # HOG - ksztalt/kontury (buty vs zegarek vs kurtka)
    hog = cv2.HOGDescriptor(
        _winSize=(size, size),
        _blockSize=(params["hog_block"], params["hog_block"]),
        _blockStride=(params["hog_cell"], params["hog_cell"]),
        _cellSize=(params["hog_cell"], params["hog_cell"]),
        _nbins=params["hog_bins"],
    )
    hog_vec = hog.compute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)).ravel()

    # histogram HSV - kolorystyka (np. zloto/srebro w bizuterii, roz w odziezy damskiej)
    bins = params["hsv_bins"]
    hist = cv2.calcHist(
        [cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).ravel()
    return np.concatenate([hog_vec, hist]).astype(np.float32)


def extract_handcrafted_features(dataset: pd.DataFrame, params: dict) -> dict:
    with ThreadPoolExecutor(params["num_workers"]) as pool:
        X = np.stack(
            list(pool.map(lambda p: _handcrafted(p, params), dataset["image_path"]))
        )
    log.info("Cechy HOG+HSV: %s", X.shape)
    return {"product_id": dataset["product_id"].to_numpy(), "X": X}


@torch.inference_mode()
def extract_embeddings(dataset: pd.DataFrame, backbone: dict, params: dict) -> dict:
    torch.set_num_threads(params["torch_threads"])
    encoder, cfg = load_backbone(backbone)
    paths = dataset["image_path"].tolist()
    batch_size = params["batch_size"]

    def load(path: str) -> np.ndarray:
        return preprocess(read_rgb(path), cfg)

    chunks = []
    start = time.perf_counter()
    with ThreadPoolExecutor(params["num_workers"]) as pool:
        for i in range(0, len(paths), batch_size):
            batch = np.stack(list(pool.map(load, paths[i : i + batch_size])))
            chunks.append(encoder(torch.from_numpy(batch)).float().numpy())
            if (i // batch_size) % 20 == 0:
                log.info("%s: %d/%d", backbone["model"], i + len(batch), len(paths))
    X = np.concatenate(chunks)
    elapsed = time.perf_counter() - start
    log.info(
        "%s: %s w %.0fs (%.1f img/s)",
        backbone["model"],
        X.shape,
        elapsed,
        len(X) / elapsed,
    )
    return {
        "product_id": dataset["product_id"].to_numpy(),
        "X": X,
        "preprocess": cfg.to_dict(),
    }


@torch.inference_mode()
def encode_class_prompts(backbone: dict, prompts: dict[str, list[str]]) -> dict:
    """Embeddingi opisow klas dla CLIP zero-shot (srednia po kilku sformulowaniach)."""
    model, tokenizer = load_clip_text_encoder(backbone)
    classes = sorted(prompts)
    embeddings = []
    for label in classes:
        emb = model.encode_text(tokenizer(prompts[label])).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.mean(dim=0).numpy())
    return {
        "classes": classes,
        "embeddings": np.stack(embeddings),
        "logit_scale": float(model.logit_scale.exp()),
    }
