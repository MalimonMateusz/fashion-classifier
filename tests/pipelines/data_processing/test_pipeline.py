import numpy as np
import pandas as pd
import pytest
from PIL import Image

from erli_machine_vision.pipelines.data_processing.nodes import (
    _hamming_pairs,
    dedupe_and_group,
    normalize_title,
    split_dataset,
    validate_images,
)

DEDUP_PARAMS = {
    "duplicate_max_distance": 0,
    "group_max_distance": 6,
    "drop_gender_mismatch": True,
}


def _products(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (label, title, phash)"""
    return pd.DataFrame(
        [
            {"product_id": str(i), "label": label, "title": title, "phash": phash}
            for i, (label, title, phash) in enumerate(rows)
        ]
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Buty NIKE Air Max 90 czarne 44,5", "buty nike air max czarne"),
        ("Bluza Męska Rozpinana 4F roz. XL", "bluza meska rozpinana f"),
        ("KURTKA DAMSKA PIKOWANA r. 2XL", "kurtka damska pikowana"),
        ("ZŁOTY ŁAŃCUSZEK 585 14K", "zloty lancuszek k"),
    ],
)
def test_normalize_title_strips_sizes_and_accents(title, expected):
    assert normalize_title(title) == expected


def test_size_variants_share_normalized_title():
    assert normalize_title("Kapcie damskie r. 38") == normalize_title(
        "Kapcie damskie r. 41"
    )


def test_hamming_pairs():
    hashes = np.array([0b0000, 0b0001, 0b0111, 0b1111], dtype=np.uint64)
    assert _hamming_pairs(hashes, 0).tolist() == []
    assert _hamming_pairs(hashes, 1).tolist() == [[0, 1], [2, 3]]


def test_dedupe_drops_cross_class_duplicates_and_keeps_one_per_same_class_copy():
    df = _products(
        [
            ("buty_meskie", "Kapcie domowe 42", "ffffffffffffffff"),
            (
                "buty_damskie",
                "Kapcie domowe 39",
                "ffffffffffffffff",
            ),  # to samo zdjecie w innej klasie
            ("zegarki", "Zegarek Casio A", "0000000000000000"),
            ("zegarki", "Zegarek Casio A", "0000000000000000"),  # identyczna kopia
            (
                "zegarki",
                "Zegarek Casio A",
                "0000000000000000",
            ),  # i jeszcze jedna (lancuch)
            ("zegarki", "Zegarek Timex", "00000000ffffffff"),
        ]
    )
    out, stats = dedupe_and_group(df, DEDUP_PARAMS)
    assert stats["dropped_cross_class_duplicates"] == 2
    assert stats["dropped_same_class_duplicates"] == 2
    assert sorted(out["title"]) == ["Zegarek Casio A", "Zegarek Timex"]


def test_dedupe_groups_size_variants_and_drops_gender_mismatch():
    df = _products(
        [
            ("buty_meskie", "Trampki Pepe Jeans 41", "0f0f0f0f0f0f0f0f"),
            (
                "buty_meskie",
                "Trampki Pepe Jeans 43",
                "f0f0f0f0f0f0f0f0",
            ),  # inne zdjecie, ten sam produkt
            ("buty_meskie", "Nike Wmns Air Force 1 r37.5", "ff00ff00ff00ff00"),
            ("bluzy_damskie", "Bluza damska szara", "00ff00ff00ff00ff"),
        ]
    )
    out, stats = dedupe_and_group(df, DEDUP_PARAMS)
    assert stats["dropped_gender_mismatch"] == 1
    assert "Nike Wmns Air Force 1 r37.5" not in out["title"].tolist()
    pepe = out[out["title"].str.startswith("Trampki")]
    assert pepe["group_id"].nunique() == 1
    assert out["group_id"].nunique() == 2


def test_split_is_group_disjoint_and_covers_all_classes():
    rng = np.random.default_rng(0)
    labels = np.repeat(["a", "b", "c"], 140)
    df = pd.DataFrame({"label": labels, "group_id": rng.integers(0, 150, len(labels))})
    out = split_dataset(df, {"test_size": 0.15, "seed": 42})
    groups_per_split = out.groupby("group_id")["split"].nunique()
    assert (groups_per_split == 1).all()
    assert set(out["split"]) == {"train", "val", "test"}
    for split in ("val", "test"):
        assert set(out.loc[out["split"] == split, "label"]) == {"a", "b", "c"}
        assert 0.08 < (out["split"] == split).mean() < 0.22


def test_validate_images_rejects_blank_and_tiny(tmp_path):
    rng = np.random.default_rng(0)
    paths = {
        "ok": Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)),
        "blank": Image.new("RGB", (200, 200), "white"),
        "tiny": Image.fromarray(rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)),
    }
    for name, img in paths.items():
        img.save(tmp_path / f"{name}.webp")
    (tmp_path / "broken.webp").write_text("<svg/>")
    df = pd.DataFrame(
        {"image_path": [str(tmp_path / f"{n}.webp") for n in [*paths, "broken"]]}
    )

    out = validate_images(df, {"min_side_px": 100, "min_gray_std": 5.0})
    assert out["image_path"].tolist() == [str(tmp_path / "ok.webp")]
    assert len(out["phash"].iloc[0]) == 16
