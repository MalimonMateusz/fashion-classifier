"""Czyszczenie zescrapowanych danych: walidacja zdjec, deduplikacja, grupowanie i podzial.

Glowne zagrozenia, ktore adresujemy:
- ten sam produkt w wielu rozmiarach = osobne oferty z niemal identycznym zdjeciem
  (wyciek miedzy train a test, zawyzone metryki),
- to samo zdjecie w dwoch klasach (np. "TANIE PANTOFLE DOMOWE" w butach M i K),
- oczywisty szum w etykietach (tytul mowi "damska", a produkt jest w klasie meskiej).
"""

from __future__ import annotations

import logging
import re
import unicodedata

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

log = logging.getLogger(__name__)

SIZE_TOKEN_RE = re.compile(
    r"\b(?:roz(?:miar)?|r)\b\.?|\b\d+(?:[.,/-]\d+)*\s*(?:cm|mm|g|gr|ml)?\b|"
    r"\b(?:x{0,3}s|m|x{0,4}l|\d?xl)\b",
)
NON_WORD_RE = re.compile(r"[^a-z ]+")
GENDER_WORDS = {
    "meskie": re.compile(r"\b(?:damsk\w*|kobiec\w*|wmns|women\w*)\b"),
    "damskie": re.compile(r"\b(?:mesk\w*|mezczyzn\w*|men)\b"),
}


def _strip_accents(text: str) -> str:
    text = text.lower().replace("ł", "l")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_title(title: str) -> str:
    """Tytul bez rozmiarow, liczb i interpunkcji - klucz do laczenia wariantow rozmiarowych."""
    text = SIZE_TOKEN_RE.sub(" ", _strip_accents(title))
    return " ".join(NON_WORD_RE.sub(" ", text).split())


def validate_images(products: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Otwiera kazde zdjecie, liczy phash i odrzuca uszkodzone, za male lub jednolite."""
    records = []
    for path in products["image_path"]:
        try:
            with Image.open(path) as raw:
                img = raw.convert("RGB")
                gray = np.asarray(img.convert("L"), dtype=np.float32)
                records.append(
                    {
                        "width": img.width,
                        "height": img.height,
                        "gray_std": float(gray.std()),
                        "phash": str(imagehash.phash(img)),
                    }
                )
        except OSError as exc:
            log.warning("Nie da sie otworzyc %s: %s", path, exc)
            records.append({"width": 0, "height": 0, "gray_std": 0.0, "phash": None})

    df = pd.concat([products.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    valid = (
        df["phash"].notna()
        & (df[["width", "height"]].min(axis=1) >= params["min_side_px"])
        & (df["gray_std"] >= params["min_gray_std"])
    )
    log.info("Walidacja zdjec: odrzucono %d z %d", (~valid).sum(), len(df))
    return df[valid].reset_index(drop=True)


def _hamming_pairs(
    hashes: np.ndarray, max_distance: int, chunk: int = 1024
) -> np.ndarray:
    """Wszystkie pary (i, j), i < j, o odleglosci Hamminga phash <= max_distance."""
    pairs = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk, None] ^ hashes[None, :]
        dist = np.bitwise_count(block)
        i, j = np.nonzero(dist <= max_distance)
        i += start
        keep = i < j
        pairs.append(np.stack([i[keep], j[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def dedupe_and_group(products: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    """Usuwa duplikaty i nadaje `group_id` (grupa = ten sam produkt / wariant).

    1. Pary zdjec o phash-distance <= `duplicate_max_distance`:
       - w roznych klasach -> etykieta niejednoznaczna, wyrzucamy obie strony,
       - w tej samej klasie -> zostawiamy jedno zdjecie.
    2. Grupy (union-find) laczace zdjecia podobne (<= `group_max_distance`)
       albo o tym samym znormalizowanym tytule, w obrebie klasy.
    3. Opcjonalnie odrzucamy oferty, ktorych tytul przeczy plci klasy.
    """
    df = products.reset_index(drop=True).copy()
    stats: dict[str, int] = {"input": len(df)}
    hashes = np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)
    labels = df["label"].to_numpy()

    dup_pairs = _hamming_pairs(hashes, params["duplicate_max_distance"])
    cross = labels[dup_pairs[:, 0]] != labels[dup_pairs[:, 1]]
    ambiguous = np.unique(dup_pairs[cross].ravel())
    # w obrebie klasy duplikaty tworza komponenty (A~B~C) - z kazdej zostaje jedno zdjecie
    uf = _UnionFind(len(df))
    for a, b in dup_pairs[~cross]:
        uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(len(df))])
    redundant = np.flatnonzero(roots != np.arange(len(df)))
    drop = np.union1d(ambiguous, redundant)
    stats["dropped_cross_class_duplicates"] = len(ambiguous)
    stats["dropped_same_class_duplicates"] = len(np.setdiff1d(redundant, ambiguous))

    df["norm_title"] = df["title"].map(normalize_title)
    if params["drop_gender_mismatch"]:
        mismatch = np.zeros(len(df), dtype=bool)
        for suffix, pattern in GENDER_WORDS.items():
            in_class = df["label"].str.endswith(suffix)
            mismatch |= in_class & df["norm_title"].str.contains(pattern)
        drop = np.union1d(drop, np.flatnonzero(mismatch))
        stats["dropped_gender_mismatch"] = int(mismatch.sum())

    keep_mask = np.ones(len(df), dtype=bool)
    keep_mask[drop] = False
    df = df[keep_mask].reset_index(drop=True)

    uf = _UnionFind(len(df))
    hashes = np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)
    labels = df["label"].to_numpy()
    for a, b in _hamming_pairs(hashes, params["group_max_distance"]):
        if labels[a] == labels[b]:
            uf.union(a, b)
    for _, idx in df.groupby(["label", "norm_title"]).indices.items():
        for other in idx[1:]:
            uf.union(idx[0], other)
    roots = np.array([uf.find(i) for i in range(len(df))])
    df["group_id"] = pd.factorize(roots)[0]

    stats["output"] = len(df)
    stats["groups"] = int(df["group_id"].nunique())
    log.info("Deduplikacja: %s", stats)
    return df, stats


def split_dataset(products: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Podzial train/val/test stratyfikowany po klasie, rozlaczny po `group_id`."""
    n_splits = round(1 / params["test_size"])
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=params["seed"]
    )
    folds = np.empty(len(products), dtype=int)
    for fold, (_, idx) in enumerate(
        sgkf.split(products, products["label"], groups=products["group_id"])
    ):
        folds[idx] = fold
    df = products.copy()
    df["split"] = np.select([folds == 0, folds == 1], ["test", "val"], default="train")
    log.info("Podzial:\n%s", pd.crosstab(df["label"], df["split"]).to_string())
    return df


def plot_class_distribution(dataset: pd.DataFrame) -> plt.Figure:
    counts = pd.crosstab(dataset["label"], dataset["split"])[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.barh(stacked=True, ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xlabel("liczba zdjec")
    ax.set_ylabel("")
    ax.set_title("Rozklad klas po czyszczeniu")
    fig.tight_layout()
    return fig


def plot_samples(dataset: pd.DataFrame, params: dict) -> plt.Figure:
    labels = sorted(dataset["label"].unique())
    n = params["samples_per_class"]
    fig, axes = plt.subplots(len(labels), n, figsize=(n * 1.6, len(labels) * 1.8))
    for row, label in zip(axes, labels, strict=True):
        sample = dataset[dataset["label"] == label].sample(
            n, random_state=params["seed"]
        )
        for ax, path in zip(row, sample["image_path"], strict=True):
            ax.imshow(Image.open(path).convert("RGB"))
            ax.axis("off")
        row[0].set_title(label, loc="left", fontsize=9)
    fig.tight_layout()
    return fig
