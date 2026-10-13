"""Klasyfikacja zdjec produktow modowych - inferencja wylacznie przez OpenCV (cv2.dnn).

Wymaga tylko: opencv, numpy (+ requests dla URL-i). Bez PyTorcha.

    python scripts/predict.py zdjecie.jpg inne.webp
    python scripts/predict.py https://i.erli.pl/tt1wce.23aae9.l.webp --top 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from erli_machine_vision.vision import PreprocessConfig, preprocess  # noqa: E402


class FashionClassifier:
    def __init__(self, model_dir: Path = ROOT / "data" / "06_models"):
        card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        pre = card["preprocess"]
        self.cfg = PreprocessConfig(
            **{**pre, "mean": tuple(pre["mean"]), "std": tuple(pre["std"])}
        )
        self.classes = card["classes"]
        self.net = cv2.dnn.readNetFromONNX(str(model_dir / "fashion_classifier.onnx"))

    def predict(self, rgb: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        self.net.setInput(preprocess(rgb, self.cfg)[None])
        proba = self.net.forward()[0]
        order = np.argsort(proba)[::-1][:top]
        return [(self.classes[i], float(proba[i])) for i in order]


def load_rgb(source: str) -> np.ndarray:
    if source.startswith(("http://", "https://")):
        import requests  # noqa: PLC0415

        data = np.frombuffer(requests.get(source, timeout=30).content, dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        bgr = cv2.imread(source, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Nie udalo sie wczytac obrazu: {source}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+", help="sciezki do plikow lub URL-e")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "06_models")
    args = parser.parse_args()

    clf = FashionClassifier(args.model_dir)
    for source in args.images:
        preds = clf.predict(load_rgb(source), top=args.top)
        print(source)  # noqa: T201
        for label, p in preds:
            print(f"  {p:6.1%}  {label}")  # noqa: T201


if __name__ == "__main__":
    main()
