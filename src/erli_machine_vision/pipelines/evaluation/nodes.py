"""Ewaluacja i porownanie modeli.

Model "najlepszy" wybieramy po val macro-F1 (nie po tescie - inaczej test
przestaje byc niezalezna ocena). Wyniki testowe raportujemy dla wszystkich.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from erli_machine_vision.pipelines.training.nodes import split_xy

log = logging.getLogger(__name__)


def evaluate_model(model, features: dict, dataset: pd.DataFrame, model_name: str):
    metrics: dict = {"model": model_name}
    predictions = []
    for split in ("val", "test"):
        X, y = split_xy(features, dataset, split)
        proba = model.predict_proba(X)
        pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
        metrics[f"{split}_accuracy"] = accuracy_score(y, pred)
        metrics[f"{split}_macro_f1"] = f1_score(y, pred, average="macro")
        rows = dataset[dataset["split"] == split]
        predictions.append(
            pd.DataFrame(
                {
                    "product_id": rows["product_id"].to_numpy(),
                    "split": split,
                    "label": y,
                    "pred": pred,
                    "confidence": proba.max(axis=1),
                    "model": model_name,
                }
            )
        )
    log.info(
        "%-26s val F1=%.3f  test acc=%.3f F1=%.3f",
        model_name,
        metrics["val_macro_f1"],
        metrics["test_accuracy"],
        metrics["test_macro_f1"],
    )
    return metrics, pd.concat(predictions, ignore_index=True)


def compare_models(*metrics: dict) -> tuple[pd.DataFrame, plt.Figure, dict]:
    table = (
        pd.DataFrame(metrics)
        .sort_values("val_macro_f1", ascending=False)
        .reset_index(drop=True)
        .round(4)
    )
    best = table.iloc[0].to_dict()
    log.info("Porownanie modeli:\n%s", table.to_string())

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(table) + 1.5))
    order = table.iloc[::-1]
    y = np.arange(len(order))
    ax.barh(
        y - 0.2,
        order["val_macro_f1"],
        height=0.4,
        label="val macro-F1",
        color="#DD8452",
    )
    ax.barh(
        y + 0.2,
        order["test_macro_f1"],
        height=0.4,
        label="test macro-F1",
        color="#4C72B0",
    )
    for yi, v in zip(y, order["test_macro_f1"], strict=True):
        ax.text(v + 0.005, yi + 0.2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, order["model"])
    ax.set_xlim(0, 1.05)
    ax.set_title("Porownanie modeli (sortowane po val macro-F1)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return table, fig, best


def report_best_model(
    best: dict, dataset: pd.DataFrame, params: dict, *all_predictions: pd.DataFrame
) -> tuple[plt.Figure, plt.Figure, dict]:
    preds = next(p for p in all_predictions if p["model"].iloc[0] == best["model"])
    test = preds[preds["split"] == "test"]
    labels = sorted(test["label"].unique())

    cm_fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        test["label"],
        test["pred"],
        labels=labels,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(
        f"Macierz pomylek (test, normalizowana po wierszach) - {best['model']}"
    )
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    cm_fig.tight_layout()

    # najbardziej "pewne siebie" bledy - czesto to szum w etykietach, nie blad modelu
    errors = test[test["label"] != test["pred"]].sort_values(
        "confidence", ascending=False
    )
    errors = errors.head(params["n_error_examples"]).merge(
        dataset[["product_id", "image_path", "title"]], on="product_id"
    )
    cols = 6
    rows = max(1, int(np.ceil(len(errors) / cols)))
    err_fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    for ax, row in zip(
        np.ravel(axes), [*errors.itertuples(), *[None] * (rows * cols)], strict=False
    ):
        ax.axis("off")
        if row is None:
            continue
        ax.imshow(Image.open(row.image_path).convert("RGB"))
        ax.set_title(
            f"jest: {row.label}\nmodel: {row.pred} ({row.confidence:.2f})\n{row.title[:32]}",
            fontsize=7,
        )
    err_fig.suptitle(f"Najpewniejsze bledy na tescie - {best['model']}")
    err_fig.tight_layout()

    report = classification_report(
        test["label"], test["pred"], output_dict=True, zero_division=0
    )
    return cm_fig, err_fig, report
