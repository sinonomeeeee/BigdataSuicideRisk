from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def probability_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ids,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=ids,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def confusion_matrix_frame(matrix: list[list[int]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
