import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from erli_machine_vision.classifiers import (
    LinearProbe,
    ZeroShotClassifier,
    l2_normalize,
    softmax,
)
from erli_machine_vision.datasets import OnnxModelDataset
from erli_machine_vision.vision import PreprocessConfig, preprocess, resize_center_crop

CFG = PreprocessConfig(
    input_size=224, crop_pct=0.875, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
)


@pytest.mark.parametrize("shape", [(268, 268), (246, 268), (400, 150)])
def test_resize_center_crop_gives_square_input(shape):
    rgb = np.zeros((*shape, 3), dtype=np.uint8)
    assert resize_center_crop(rgb, CFG).shape == (224, 224, 3)


def test_preprocess_normalizes_to_chw():
    rgb = np.full((268, 268, 3), 255, dtype=np.uint8)
    out = preprocess(rgb, CFG)
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 2.0)  # (1.0 - 0.5) / 0.25


def test_zero_shot_linear_head_matches_predict_proba():
    rng = np.random.default_rng(0)
    clf = ZeroShotClassifier(
        rng.normal(size=(3, 8)), ["a", "b", "c"], logit_scale=100.0
    )
    X = rng.normal(size=(5, 8))
    W, b = clf.linear_head()
    np.testing.assert_allclose(
        softmax(l2_normalize(X) @ W.T + b), clf.predict_proba(X), rtol=1e-6
    )


def test_linear_probe_head_matches_predict_proba():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 8)), np.repeat(["a", "b", "c"], 20)
    est = LogisticRegression(max_iter=500).fit(l2_normalize(X), y)
    probe = LinearProbe(est, est.classes_)
    W, b = probe.linear_head()
    np.testing.assert_allclose(
        softmax(l2_normalize(X) @ W.T + b), probe.predict_proba(X), rtol=1e-6
    )


def test_onnx_dataset_roundtrip(tmp_path):
    ds = OnnxModelDataset(str(tmp_path / "m" / "model.onnx"))
    assert not ds.exists()
    ds.save(b"\x08\x01onnx")
    assert ds.load() == b"\x08\x01onnx"
