#!/usr/bin/env python3
"""End-to-end local evidence extraction training/evaluation pipeline.

This script is designed around the files supplied with the current evidence
pipeline:

  * train_none.xlsx
  * train_none.xlsx
  * hard_negatives_modified.csv

Candidate generation and training-pair preparation are performed automatically.
Prebuilt candidates/training-pairs can still be supplied optionally.

It performs the whole model/evaluation loop locally:

  1. Load candidate-level training pairs.
  2. Apply human-reviewed hard-negative decisions:
       true_negative    -> label 0
       annotation_error -> label 1
       ambiguous        -> remove from modelling/evaluation
  3. Split by row_id into train/validation/test (no post leakage).
  4. Optionally duplicate confirmed hard negatives in TRAIN only.
  5. Fine-tune a Hugging Face sequence classifier.
  6. Save candidate-level validation/test probabilities.
  7. Tune aggregation settings on validation row-level F1.
  8. Aggregate test candidates into evidence phrases.
  9. Report both candidate-level and row-level metrics.
 10. Save the model, splits, predictions, metrics, and config locally.

The default model/text format matches the existing project:

    Candidate: <candidate>\nPost: <post>

Example
-------
python run_evidence_pipeline_local_v3.py \
  --posts train_none.xlsx \
  --reviewed-negatives hard_negatives_modified.csv \
  --output-dir evidence_local_run \
  --device auto

For an NVIDIA GPU, install a CUDA-enabled PyTorch build first. Typical Python
packages are:

  pip install "transformers>=4.46,<5" "datasets>=3.1,<4" \
      "accelerate>=1.1,<2" sentencepiece scikit-learn pandas openpyxl torch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    training_pairs: str | None = None
    posts: str = "train_none.xlsx"
    candidates: str | None = None
    reviewed_negatives: str = "hard_negatives_modified.csv"
    expected_candidate_count: int = 79478
    output_dir: str = "evidence_local_run"
    model_name: str = "microsoft/deberta-v3-base"
    row_id_col: str = None
    post_col: str = None
    candidate_col: str = None
    label_col: str = None
    gold_col: str = None
    risk_col: str | None = None
    max_length: int = 0
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    epochs: float = 0.0
    train_batch_size: int = 0
    eval_batch_size: int = 0
    gradient_accumulation_steps: int = 0
    warmup_ratio: float = 0.0
    early_stopping_patience: int = 0
    test_size: float = 0.0
    validation_size_from_remaining: float = 0.0
    seed: int = 0
    fp16: bool = False
    bf16: bool = False
    device: str = None
    hard_negative_extra_copies: int = 0
    use_reviewed_annotation_errors_as_positive: bool = False
    drop_ambiguous: bool = False
    tune_aggregation: bool = False
    threshold_grid: list[float] = 0.0
    overlap_grid: list[float] = 0.0
    max_phrases_grid: list[int] = 0
    default_threshold: float = 0.0
    default_overlap_threshold: float = 0.0
    default_max_phrases: int = 0
    min_match_word_ratio: float = 0.0
    max_match_word_ratio: float = 0.0
    force: bool = False
    no_resume: bool = False


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate the evidence extraction model locally."
    )
    parser.add_argument(
        "--training-pairs", default=None,
        help="Optional prebuilt training-pairs CSV. If omitted, rebuild from --posts + --candidates."
    )
    parser.add_argument("--posts", default="train_none.xlsx")
    parser.add_argument(
        "--candidates",
        default=None,
        help=(
            "Optional prebuilt candidate CSV. If omitted, candidates are "
            "generated directly from --posts and saved under output-dir/prepared/."
        ),
    )
    parser.add_argument("--reviewed-negatives", default="hard_negatives_modified.csv")
    parser.add_argument(
        "--expected-candidate-count", type=int, default=79478,
        help="Sanity-check generated E1 candidate rows. Set 0 to disable. Default: 79478.",
    )
    parser.add_argument("--output-dir", default="evidence_local_run")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore reusable outputs and rebuild/retrain every stage.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="If training is incomplete, start training from scratch instead of the latest checkpoint.",
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument(
        "--leaderboard",
        default="leaderboard.xlsx",
        help="Leaderboard posts file for final inference. Default: leaderboard.xlsx",
    )
    parser.add_argument(
        "--risk-file",
        default="SatAndSeriousTokyoScript.csv",
        help="CSV/XLSX containing row_id and suicide-risk prediction for leaderboard rows.",
    )
    parser.add_argument(
        "--skip-leaderboard",
        action="store_true",
        help="Skip final leaderboard inference.",
    )

    parser.add_argument("--row-id-col", default="row_id")
    parser.add_argument("--post-col", default="post_y")
    parser.add_argument("--candidate-col", default="candidate")
    parser.add_argument("--label-col", default="is_match")
    parser.add_argument("--gold-col", default="gold_phrases")
    parser.add_argument(
        "--risk-col",
        default=None,
        help=(
            "Risk column used for the Indicator -> none override. If omitted, "
            "common names such as 'suicide risk' are detected automatically."
        ),
    )

    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)

    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--validation-size-from-remaining",
        type=float,
        default=0.125,
        help="0.125 after a 20%% test split gives approximately 70/10/20.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--hard-negative-extra-copies",
        type=int,
        default=1,
        help=(
            "Extra TRAIN-only copies of each human-confirmed true_negative pair. "
            "Default 1 means a reviewed hard negative appears twice in train. "
            "Use 0 to disable oversampling."
        ),
    )
    parser.add_argument(
        "--do-not-promote-annotation-errors",
        action="store_true",
        help="Keep annotation_error pairs at their original label instead of forcing label 1.",
    )
    parser.add_argument(
        "--keep-ambiguous",
        action="store_true",
        help="Keep ambiguous reviewed pairs instead of removing them.",
    )

    parser.add_argument(
        "--no-tune-aggregation",
        action="store_true",
        help="Skip validation grid search and use the default aggregation settings.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=parse_float_list,
        default=parse_float_list("0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90"),
    )
    parser.add_argument(
        "--overlap-grid",
        type=parse_float_list,
        default=parse_float_list("0.0"),
    )
    parser.add_argument(
        "--max-phrases-grid",
        type=parse_int_list,
        default=parse_int_list("0,1,2,3,4"),
        help="0 means unlimited.",
    )
    parser.add_argument("--default-threshold", type=float, default=0.70)
    parser.add_argument("--default-overlap-threshold", type=float, default=0.0)
    parser.add_argument("--default-max-phrases", type=int, default=0)
    parser.add_argument("--min-match-word-ratio", type=float, default=1 / 3)
    parser.add_argument("--max-match-word-ratio", type=float, default=3.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: object) -> str:
    return clean_text(value).casefold()


def get_tokens(value: object) -> list[str]:
    return re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", normalize(value), flags=re.UNICODE)


def word_count(value: object) -> int:
    return len(get_tokens(value))


def split_phrases(value: object) -> list[str]:
    text = clean_text(value)
    if not text or text.casefold() == "none":
        return []
    return [
        clean_text(part)
        for part in text.split(";")
        if clean_text(part) and clean_text(part).casefold() != "none"
    ]


def unique_phrases(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for phrase in split_phrases(value):
            key = normalize(phrase)
            if key and key not in seen:
                seen.add(key)
                output.append(phrase)
    return output


def normalize_candidate_key(value: object) -> str:
    return normalize(value)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (set, tuple)):
            return list(value)
        return value

    if isinstance(data, dict):
        serializable = {key: convert(value) for key, value in data.items()}
    else:
        serializable = data

    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2, default=convert)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def detect_risk_column(df: pd.DataFrame, explicit: str | None) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Risk column not found: {explicit}")
        return explicit

    exact = [
        "suicide risk",
        "suicide_risk",
        "risk_level",
        "risk",
        "suicide risk prediction",
        "predicted suicide risk",
    ]
    for column in exact:
        if column in df.columns:
            return column

    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(column).casefold()): column
        for column in df.columns
    }
    for key in ["suiciderisk", "risklevel", "risk", "suicideriskprediction"]:
        if key in normalized:
            return normalized[key]
    return None


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def load_any_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp932")
    if suffix == ".tsv":
        try:
            return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep="\t", encoding="cp932")
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_join_row_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text



# ---------------------------------------------------------------------------
# Human review integration
# ---------------------------------------------------------------------------


def apply_reviewed_labels(
    base_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
    row_id_col: str,
    candidate_col: str,
    label_col: str,
    promote_annotation_errors: bool,
    drop_ambiguous: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_columns(base_df, [row_id_col, candidate_col, label_col], "training pairs")
    ensure_columns(reviewed_df, [row_id_col, candidate_col, "review_label"], "reviewed negatives")

    base = base_df.copy()
    reviewed = reviewed_df.copy()

    base["_candidate_key"] = base[candidate_col].map(normalize_candidate_key)
    reviewed["_candidate_key"] = reviewed[candidate_col].map(normalize_candidate_key)
    reviewed["review_label"] = reviewed["review_label"].map(normalize)

    reviewed = reviewed[
        reviewed[row_id_col].notna() & reviewed["_candidate_key"].ne("")
    ].copy()

    conflict_counts = (
        reviewed.groupby([row_id_col, "_candidate_key"])["review_label"].nunique()
    )
    conflicts = conflict_counts[conflict_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Reviewed file has {len(conflicts)} candidate pairs with conflicting review labels."
        )

    reviewed_unique = reviewed.drop_duplicates(
        subset=[row_id_col, "_candidate_key"], keep="first"
    )[[row_id_col, "_candidate_key", "review_label"]]

    base = base.merge(
        reviewed_unique,
        on=[row_id_col, "_candidate_key"],
        how="left",
        validate="many_to_one",
    )

    original_rows = len(base)
    matched_review_rows = int(base["review_label"].notna().sum())
    matched_review_pairs = int(
        base.loc[base["review_label"].notna(), [row_id_col, "_candidate_key"]]
        .drop_duplicates()
        .shape[0]
    )

    base["human_reviewed"] = base["review_label"].notna()
    base["confirmed_hard_negative"] = base["review_label"].eq("true_negative")

    true_negative_mask = base["review_label"].eq("true_negative")
    base.loc[true_negative_mask, label_col] = 0

    annotation_mask = base["review_label"].eq("annotation_error")
    if promote_annotation_errors:
        base.loc[annotation_mask, label_col] = 1

    ambiguous_mask = base["review_label"].eq("ambiguous")
    ambiguous_rows_removed = 0
    ambiguous_pairs_removed = 0
    if drop_ambiguous:
        ambiguous_rows_removed = int(ambiguous_mask.sum())
        ambiguous_pairs_removed = int(
            base.loc[ambiguous_mask, [row_id_col, "_candidate_key"]]
            .drop_duplicates()
            .shape[0]
        )
        base = base.loc[~ambiguous_mask].copy()

    base[label_col] = pd.to_numeric(base[label_col], errors="coerce")
    base = base[base[label_col].isin([0, 1])].copy()
    base[label_col] = base[label_col].astype(int)

    stats = {
        "base_rows_before_review": original_rows,
        "base_rows_after_review": len(base),
        "reviewed_input_rows": len(reviewed_df),
        "reviewed_unique_pairs": len(reviewed_unique),
        "matched_review_rows": matched_review_rows,
        "matched_review_pairs": matched_review_pairs,
        "review_label_counts_input_rows": reviewed_df["review_label"].value_counts(dropna=False).to_dict(),
        "review_label_counts_unique_pairs": reviewed_unique["review_label"].value_counts(dropna=False).to_dict(),
        "annotation_error_rows_promoted": int(annotation_mask.sum()) if promote_annotation_errors else 0,
        "true_negative_rows_confirmed": int(true_negative_mask.sum()),
        "ambiguous_rows_removed": ambiguous_rows_removed,
        "ambiguous_pairs_removed": ambiguous_pairs_removed,
        "label_counts_after_review": base[label_col].value_counts().sort_index().to_dict(),
    }

    return base, stats


# ---------------------------------------------------------------------------
# Cleaning and grouped split
# ---------------------------------------------------------------------------


def prepare_dataframe(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ensure_columns(
        df,
        [cfg.row_id_col, cfg.post_col, cfg.candidate_col, cfg.label_col, cfg.gold_col],
        "review-adjusted training data",
    )

    output = df.copy()
    output[cfg.post_col] = output[cfg.post_col].map(clean_text)
    output[cfg.candidate_col] = output[cfg.candidate_col].map(clean_text)
    output[cfg.gold_col] = output[cfg.gold_col].fillna("")

    output = output[
        output[cfg.row_id_col].notna()
        & output[cfg.post_col].ne("")
        & output[cfg.candidate_col].ne("")
        & output[cfg.candidate_col].str.casefold().ne("none")
    ].copy()

    # One model example per unique row/candidate/label. The phrase generator can
    # create the same surface candidate through multiple anchor/window paths.
    output["_candidate_key"] = output[cfg.candidate_col].map(normalize_candidate_key)
    sort_columns = [cfg.row_id_col, "_candidate_key", cfg.label_col]
    output = output.drop_duplicates(subset=sort_columns, keep="first").reset_index(drop=True)

    if output.empty:
        raise ValueError("No usable rows remain after cleaning.")
    if output[cfg.label_col].nunique() < 2:
        raise ValueError("Training data must contain both labels 0 and 1.")

    output["model_text"] = (
        "Candidate: " + output[cfg.candidate_col] + "\nPost: " + output[cfg.post_col]
    )
    return output


def grouped_split(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    train_val_idx, test_idx = next(
        first.split(df, y=df[cfg.label_col], groups=df[cfg.row_id_col])
    )
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    second = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.validation_size_from_remaining,
        random_state=cfg.seed,
    )
    train_idx, val_idx = next(
        second.split(
            train_val,
            y=train_val[cfg.label_col],
            groups=train_val[cfg.row_id_col],
        )
    )
    train = train_val.iloc[train_idx].reset_index(drop=True)
    validation = train_val.iloc[val_idx].reset_index(drop=True)

    split_sets = {
        "train": set(train[cfg.row_id_col].astype(str)),
        "validation": set(validation[cfg.row_id_col].astype(str)),
        "test": set(test[cfg.row_id_col].astype(str)),
    }
    if split_sets["train"] & split_sets["validation"]:
        raise AssertionError("row_id leakage between train and validation")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("row_id leakage between train and test")
    if split_sets["validation"] & split_sets["test"]:
        raise AssertionError("row_id leakage between validation and test")

    return train, validation, test


def oversample_hard_negatives(
    train_df: pd.DataFrame,
    extra_copies: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if extra_copies <= 0 or "confirmed_hard_negative" not in train_df.columns:
        return train_df.reset_index(drop=True), {
            "hard_negative_unique_train_rows": 0,
            "hard_negative_extra_rows_added": 0,
        }

    hard = train_df[train_df["confirmed_hard_negative"]].copy()
    if hard.empty:
        return train_df.reset_index(drop=True), {
            "hard_negative_unique_train_rows": 0,
            "hard_negative_extra_rows_added": 0,
        }

    additions = pd.concat([hard.copy() for _ in range(extra_copies)], ignore_index=True)
    output = pd.concat([train_df, additions], ignore_index=True)
    output = output.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return output, {
        "hard_negative_unique_train_rows": len(hard),
        "hard_negative_extra_rows_added": len(additions),
    }


# ---------------------------------------------------------------------------
# Hugging Face training and prediction
# ---------------------------------------------------------------------------


def to_hf_dataset(
    df: pd.DataFrame,
    cfg: Config,
    tokenizer: Any,
) -> Dataset:
    """Plain PyTorch dataset; avoids Hugging Face datasets/dill fingerprinting."""

    class EncodedTextDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int]) -> None:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=cfg.max_length,
                padding=False,
            )
            self.labels = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {
                key: values[index]
                for key, values in self.encodings.items()
            }
            item["labels"] = self.labels[index]
            return item

    texts = df["model_text"].astype(str).tolist()
    labels = df[cfg.label_col].astype(int).tolist()
    return EncodedTextDataset(texts, labels)


def candidate_metrics_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return candidate_metrics_from_arrays(labels, predictions)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)



def validation_gold_phrase_coverage(
    validation_df: pd.DataFrame,
    positive_predictions: np.ndarray,
    cfg: Config,
) -> float:
    """
    Fraction of unique validation gold phrases covered by at least one
    candidate predicted as evidence.

    Coverage is computed by row_id. A gold phrase is covered when any predicted
    positive candidate for that same row matches it using the pipeline's
    existing evidence_match rule and configured word-ratio bounds.

    This metric is intentionally candidate-model-only: it does not apply the
    later risk override, overlap suppression, max-phrases restriction, or
    adjacent-span merging.
    """
    if len(validation_df) != len(positive_predictions):
        raise ValueError(
            "Validation prediction count does not match validation dataframe: "
            f"{len(positive_predictions):,} vs {len(validation_df):,}."
        )

    working = validation_df.copy().reset_index(drop=True)
    working["_predicted_positive_for_coverage"] = (
        np.asarray(positive_predictions).astype(int)
    )

    total_gold = 0
    covered_gold = 0

    for _, group in working.groupby(
        cfg.row_id_col,
        sort=False,
        dropna=False,
    ):
        golds = unique_phrases(group[cfg.gold_col])
        if not golds:
            continue

        predicted_candidates = [
            clean_text(candidate)
            for candidate, is_positive in zip(
                group[cfg.candidate_col],
                group["_predicted_positive_for_coverage"],
            )
            if int(is_positive) == 1
            and clean_text(candidate)
            and clean_text(candidate).casefold() != "none"
        ]

        total_gold += len(golds)

        for gold in golds:
            if any(
                evidence_match(
                    predicted=candidate,
                    gold=gold,
                    min_word_ratio=cfg.min_match_word_ratio,
                    max_word_ratio=cfg.max_match_word_ratio,
                )
                for candidate in predicted_candidates
            ):
                covered_gold += 1

    if total_gold == 0:
        return 0.0

    return covered_gold / total_gold



def train_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    cfg: Config,
    model_dir: Path,
    checkpoint_dir: Path,
    resume_checkpoint: Path | None = None,
) -> tuple[Trainer, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label={0: "not_evidence", 1: "evidence"},
        label2id={"not_evidence": 0, "evidence": 1},
    )

    train_dataset = to_hf_dataset(train_df, cfg, tokenizer)
    validation_dataset = to_hf_dataset(validation_df, cfg, tokenizer)

    def compute_validation_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(
                labels, predictions, zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0
            ),
            "f1": f1_score(
                labels, predictions, zero_division=0
            ),
            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

        metrics["gold_phrase_coverage"] = validation_gold_phrase_coverage(
            validation_df=validation_df,
            positive_predictions=predictions,
            cfg=cfg,
        )
        return metrics

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = resolve_device(cfg.device)
    use_fp16 = bool(cfg.fp16 and device.type == "cuda")
    use_bf16 = bool(cfg.bf16 and device.type == "cuda")

    if cfg.fp16 and device.type != "cuda":
        print("Warning: --fp16 ignored because CUDA is not active.")
    if cfg.bf16 and device.type != "cuda":
        print("Warning: --bf16 ignored because CUDA is not active.")
    if use_fp16 and use_bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="gold_phrase_coverage",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_validation_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience
            )
        ],
    )

    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint.resolve()}")
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return trainer, tokenizer


def predict_dataframe(
    trainer: Trainer,
    tokenizer: Any,
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = to_hf_dataset(df, cfg, tokenizer)

    # The Trainer's compute_metrics callback is validation-specific because it
    # includes gold_phrase_coverage tied to validation_df. Disable it temporarily
    # for generic prediction (test/other splits), then restore it afterward.
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    try:
        output = trainer.predict(dataset)
    finally:
        trainer.compute_metrics = original_compute_metrics

    logits = output.predictions
    labels = output.label_ids
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=-1)

    result = df.copy().reset_index(drop=True)
    result["true_label"] = labels
    result["predicted_label_0_5"] = predictions
    result["prob_not_evidence"] = probabilities[:, 0]
    result["prob_evidence"] = probabilities[:, 1]
    result["candidate_correct_0_5"] = result["true_label"].eq(
        result["predicted_label_0_5"]
    )

    metrics = candidate_metrics_from_arrays(labels, predictions)
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()
    metrics["classification_report"] = classification_report(
        labels,
        predictions,
        target_names=["not_evidence", "evidence"],
        output_dict=True,
        zero_division=0,
    )
    return result, metrics


# ---------------------------------------------------------------------------
# Aggregation / row-level evaluation
# ---------------------------------------------------------------------------


def token_overlap_ratio(first: object, second: object) -> float:
    first_tokens = set(get_tokens(first))
    second_tokens = set(get_tokens(second))
    if not first_tokens or not second_tokens:
        return 0.0
    return len(first_tokens & second_tokens) / min(len(first_tokens), len(second_tokens))


def phrases_overlap(first: str, second: str, overlap_threshold: float) -> bool:
    if normalize(first) == normalize(second):
        return True
    return token_overlap_ratio(first, second) > overlap_threshold


def keep_highest_probability_per_overlap_group(
    candidates: list[tuple[str, float]],
    overlap_threshold: float,
) -> list[tuple[str, float]]:
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    count = len(candidates)
    adjacency: list[set[int]] = [set() for _ in range(count)]

    for i in range(count):
        for j in range(i + 1, count):
            if phrases_overlap(candidates[i][0], candidates[j][0], overlap_threshold):
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: set[int] = set()
    selected: list[tuple[str, float]] = []
    for start in range(count):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        selected.append(candidates[min(component)])

    return sorted(selected, key=lambda item: item[1], reverse=True)


def select_phrases_from_group(
    group: pd.DataFrame,
    candidate_col: str,
    threshold: float,
    overlap_threshold: float,
    max_phrases: int,
) -> tuple[list[str], list[float]]:
    working = group[[candidate_col, "prob_evidence"]].copy()
    working[candidate_col] = working[candidate_col].map(clean_text)
    working = working[
        working[candidate_col].ne("")
        & working[candidate_col].str.casefold().ne("none")
    ].copy()
    if working.empty:
        return [], []

    working["word_count"] = working[candidate_col].map(word_count)
    working["_normalized"] = working[candidate_col].map(normalize)
    working = (
        working.sort_values(
            ["prob_evidence", "word_count"],
            ascending=[False, False],
            kind="stable",
        )
        .drop_duplicates("_normalized", keep="first")
    )

    eligible = working[working["prob_evidence"] >= threshold].copy()
    candidates = [
        (clean_text(row[candidate_col]), float(row["prob_evidence"]))
        for _, row in eligible.iterrows()
    ]
    selected = keep_highest_probability_per_overlap_group(
        candidates, overlap_threshold=overlap_threshold
    )
    if max_phrases > 0:
        selected = selected[:max_phrases]
    return [item[0] for item in selected], [item[1] for item in selected]


def evidence_match(
    predicted: str,
    gold: str,
    min_word_ratio: float,
    max_word_ratio: float,
) -> bool:
    p = normalize(predicted)
    g = normalize(gold)
    if not p or not g:
        return False

    p_words = word_count(p)
    g_words = word_count(g)
    if p_words == 0 or g_words == 0:
        return False

    ratio = p_words / g_words
    if ratio < min_word_ratio or ratio > max_word_ratio:
        return False

    return f" {p} " in f" {g} " or f" {g} " in f" {p} "


def maximum_matches(
    predictions: list[str],
    golds: list[str],
    min_word_ratio: float,
    max_word_ratio: float,
) -> int:
    if not predictions or not golds:
        return 0

    adjacency: list[list[int]] = []
    for predicted in predictions:
        adjacency.append(
            [
                index
                for index, gold in enumerate(golds)
                if evidence_match(
                    predicted,
                    gold,
                    min_word_ratio=min_word_ratio,
                    max_word_ratio=max_word_ratio,
                )
            ]
        )

    gold_to_prediction = [-1] * len(golds)

    def assign(prediction_index: int, visited_gold: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous = gold_to_prediction[gold_index]
            if previous == -1 or assign(previous, visited_gold):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    matches = 0
    for prediction_index in range(len(predictions)):
        if assign(prediction_index, set()):
            matches += 1
    return matches


def row_metrics(
    predictions: list[str],
    golds: list[str],
    min_word_ratio: float,
    max_word_ratio: float,
) -> tuple[float, float, float, int]:
    tp = maximum_matches(
        predictions,
        golds,
        min_word_ratio=min_word_ratio,
        max_word_ratio=max_word_ratio,
    )

    if not predictions and not golds:
        return 1.0, 1.0, 1.0, 0

    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(golds) if golds else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1, tp



def merge_adjacent_candidates_in_post(
    post: str,
    phrases: list[str],
    probabilities: list[float],
) -> tuple[list[str], list[float]]:
    """
    Merge selected candidate phrases that are adjacent in the original post.

    Two spans are considered adjacent when, in post order, there is no word
    character between the end of the first span and the start of the second.
    Whitespace and punctuation-only separators are therefore mergeable.

    The merged phrase is copied from the original post so punctuation between
    the two candidates is preserved. Its probability is the maximum probability
    of the merged candidates.
    """
    if len(phrases) < 2 or not clean_text(post):
        return phrases, probabilities

    post_text = str(post)
    post_fold = post_text.casefold()

    located: list[tuple[int, int, str, float, int]] = []
    occurrence_cursor: dict[str, int] = {}

    for original_index, phrase in enumerate(phrases):
        phrase_text = clean_text(phrase)
        probability = (
            float(probabilities[original_index])
            if original_index < len(probabilities)
            else 0.0
        )

        if not phrase_text:
            continue

        needle = phrase_text.casefold()
        cursor = occurrence_cursor.get(needle, 0)
        start = post_fold.find(needle, cursor)

        # If the phrase only occurs before the cursor, retain the first match.
        if start < 0:
            start = post_fold.find(needle)

        if start < 0:
            # Keep candidates that cannot be mapped back to the post, but do not
            # attempt to merge them.
            located.append(
                (10**12 + original_index, 10**12 + original_index,
                 phrase_text, probability, original_index)
            )
            continue

        end = start + len(needle)
        occurrence_cursor[needle] = end
        located.append((start, end, phrase_text, probability, original_index))

    positioned = [item for item in located if item[0] < 10**12]
    unpositioned = [item for item in located if item[0] >= 10**12]

    if len(positioned) < 2:
        return phrases, probabilities

    positioned.sort(key=lambda item: (item[0], item[1]))

    merged: list[tuple[int, str, float]] = []
    current_start, current_end, current_phrase, current_prob, _ = positioned[0]

    for start, end, phrase, probability, _ in positioned[1:]:
        # Overlapping candidates should already have been handled by overlap
        # suppression. Only merge non-overlapping spans here.
        if start >= current_end:
            between = post_text[current_end:start]
            adjacent = re.search(r"\w", between, flags=re.UNICODE) is None
        else:
            adjacent = False

        if adjacent:
            current_end = end
            current_phrase = clean_text(post_text[current_start:current_end])
            current_prob = max(current_prob, probability)
        else:
            merged.append((current_start, current_phrase, current_prob))
            current_start = start
            current_end = end
            current_phrase = phrase
            current_prob = probability

    merged.append((current_start, current_phrase, current_prob))

    # Preserve any candidate we could not locate in the post.
    for start, _, phrase, probability, _ in unpositioned:
        merged.append((start, phrase, probability))

    merged.sort(key=lambda item: item[0])

    return (
        [phrase for _, phrase, _ in merged],
        [probability for _, _, probability in merged],
    )



def aggregate_split(
    prediction_df: pd.DataFrame,
    cfg: Config,
    threshold: float,
    overlap_threshold: float,
    max_phrases: int,
    risk_col: str | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []

    for row_id, group in prediction_df.groupby(cfg.row_id_col, sort=False, dropna=False):
        phrases, probabilities = select_phrases_from_group(
            group,
            candidate_col=cfg.candidate_col,
            threshold=threshold,
            overlap_threshold=overlap_threshold,
            max_phrases=max_phrases,
        )

        golds = unique_phrases(group[cfg.gold_col])
        risk_value = ""
        if risk_col and risk_col in group.columns:
            nonempty = [clean_text(v) for v in group[risk_col] if clean_text(v)]
            risk_value = nonempty[0] if nonempty else ""

        indicator_override = risk_value.casefold() == "indicator"
        if indicator_override:
            phrases = []
            probabilities = []

        post_values = [clean_text(v) for v in group[cfg.post_col] if clean_text(v)]
        post = post_values[0] if post_values else ""

        # Final aggregation rule: when two surviving candidates are adjacent
        # in the source post, merge them before row-level evaluation/output.
        if not indicator_override:
            phrases, probabilities = merge_adjacent_candidates_in_post(
                post=post,
                phrases=phrases,
                probabilities=probabilities,
            )

        precision, recall, f1, tp = row_metrics(
            phrases,
            golds,
            min_word_ratio=cfg.min_match_word_ratio,
            max_word_ratio=cfg.max_match_word_ratio,
        )

        rows.append(
            {
                cfg.row_id_col: row_id,
                "predicted": "; ".join(phrases) if phrases else "none",
                "gold_evidences": "; ".join(golds) if golds else "none",
                "prob_evidence": "; ".join(f"{p:.6f}" for p in probabilities),
                "suicide_risk": risk_value,
                "indicator_override": indicator_override,
                "row_precision": precision,
                "row_recall": recall,
                "row_f1": f1,
                "true_positive_phrases": tp,
                "predicted_phrase_count": len(phrases),
                "gold_phrase_count": len(golds),
                "post": post,
            }
        )

    output = pd.DataFrame(rows)
    total_predictions = int(output["predicted_phrase_count"].sum())
    total_gold = int(output["gold_phrase_count"].sum())
    total_tp = int(output["true_positive_phrases"].sum())
    micro_precision = total_tp / total_predictions if total_predictions else 0.0
    micro_recall = total_tp / total_gold if total_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    metrics = {
        "macro_row_precision": float(output["row_precision"].mean()),
        "macro_row_recall": float(output["row_recall"].mean()),
        "macro_row_f1": float(output["row_f1"].mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "rows": int(len(output)),
        "total_predictions": total_predictions,
        "total_gold": total_gold,
        "total_true_positives": total_tp,
    }
    return output, metrics


def tune_aggregation(
    validation_predictions: pd.DataFrame,
    cfg: Config,
    risk_col: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Tune aggregation settings on validation row-level F1 before final adjacent-span merging.

    Optimized relative to the previous brute-force implementation:
    - group rows once instead of rebuilding groupby for all 180 combinations;
    - normalize/dedupe/sort candidate rows once per row_id;
    - precompute pairwise overlap ratios once per row_id;
    - print progress for every combination so the process never looks hung.\n    Adjacent-span merging is deliberately NOT applied during tuning; it is\n    applied only after aggregation settings are selected.\n    """
    print(
        f"\nAggregation grid search: "
        f"{len(cfg.threshold_grid) * len(cfg.overlap_grid) * len(cfg.max_phrases_grid)} combinations"
    )

    # Precompute everything that does not depend on the grid setting.
    prepared_groups: list[dict[str, Any]] = []

    for row_id, group in validation_predictions.groupby(
        cfg.row_id_col,
        sort=False,
        dropna=False,
    ):
        working = group[[cfg.candidate_col, "prob_evidence"]].copy()
        working[cfg.candidate_col] = working[cfg.candidate_col].map(clean_text)
        working = working[
            working[cfg.candidate_col].ne("")
            & working[cfg.candidate_col].str.casefold().ne("none")
        ].copy()

        if not working.empty:
            working["word_count"] = working[cfg.candidate_col].map(word_count)
            working["_normalized"] = working[cfg.candidate_col].map(normalize)
            working = (
                working.sort_values(
                    ["prob_evidence", "word_count"],
                    ascending=[False, False],
                    kind="stable",
                )
                .drop_duplicates("_normalized", keep="first")
                .reset_index(drop=True)
            )

        phrases_all = (
            working[cfg.candidate_col].tolist()
            if not working.empty
            else []
        )
        probs_all = (
            working["prob_evidence"].astype(float).tolist()
            if not working.empty
            else []
        )

        n = len(phrases_all)
        overlap_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            overlap_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                ratio = token_overlap_ratio(phrases_all[i], phrases_all[j])
                overlap_matrix[i][j] = ratio
                overlap_matrix[j][i] = ratio

        golds = unique_phrases(group[cfg.gold_col])

        risk_value = ""
        if risk_col and risk_col in group.columns:
            nonempty = [
                clean_text(v)
                for v in group[risk_col]
                if clean_text(v)
            ]
            risk_value = nonempty[0] if nonempty else ""

        indicator_override = risk_value.casefold() == "indicator"

        post_values = [
            clean_text(v)
            for v in group[cfg.post_col]
            if clean_text(v)
        ]
        post = post_values[0] if post_values else ""

        prepared_groups.append(
            {
                "row_id": row_id,
                "phrases": phrases_all,
                "probs": probs_all,
                "overlap": overlap_matrix,
                "golds": golds,
                "indicator_override": indicator_override,
                "post": post,
            }
        )

    def select_cached(
        item: dict[str, Any],
        threshold: float,
        overlap_threshold: float,
        max_phrases: int,
    ) -> tuple[list[str], list[float]]:
        phrases_all = item["phrases"]
        probs_all = item["probs"]
        overlap_matrix = item["overlap"]

        eligible = [
            i
            for i, prob in enumerate(probs_all)
            if prob >= threshold
        ]

        if not eligible:
            return [], []

        # Connected components over eligible candidates, using precomputed
        # overlap ratios. Candidates are already sorted by probability desc.
        eligible_set = set(eligible)
        visited: set[int] = set()
        selected_indices: list[int] = []

        for start_idx in eligible:
            if start_idx in visited:
                continue

            stack = [start_idx]
            component: list[int] = []

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)

                for other in eligible:
                    if other in visited or other == current:
                        continue
                    if overlap_matrix[current][other] > overlap_threshold:
                        stack.append(other)

            # Lowest index = highest probability because working was sorted desc.
            selected_indices.append(min(component))

        selected_indices.sort(key=lambda i: probs_all[i], reverse=True)

        if max_phrases > 0:
            selected_indices = selected_indices[:max_phrases]

        return (
            [phrases_all[i] for i in selected_indices],
            [probs_all[i] for i in selected_indices],
        )

    rows: list[dict[str, Any]] = []
    total_combinations = (
        len(cfg.threshold_grid)
        * 1
        * len(cfg.max_phrases_grid)
    )
    combo_index = 0

    for threshold in cfg.threshold_grid:
        for overlap_threshold in (0.0,):
            for max_phrases in cfg.max_phrases_grid:
                combo_index += 1
                print(
                    f"  [{combo_index:3d}/{total_combinations}] "
                    f"threshold={threshold:.2f}, "
                    f"overlap={overlap_threshold:.2f}, "
                    f"max_phrases={max_phrases or 'all'}",
                    flush=True,
                )

                metric_rows: list[tuple[float, float, float, int, int, int]] = []

                for item in prepared_groups:
                    if item["indicator_override"]:
                        phrases = []
                        probabilities = []
                    else:
                        phrases, probabilities = select_cached(
                            item,
                            threshold=threshold,
                            overlap_threshold=overlap_threshold,
                            max_phrases=max_phrases,
                        )

                    precision, recall, f1, tp = row_metrics(
                        phrases,
                        item["golds"],
                        min_word_ratio=cfg.min_match_word_ratio,
                        max_word_ratio=cfg.max_match_word_ratio,
                    )

                    metric_rows.append(
                        (
                            precision,
                            recall,
                            f1,
                            tp,
                            len(phrases),
                            len(item["golds"]),
                        )
                    )

                macro_precision = float(
                    np.mean([x[0] for x in metric_rows])
                )
                macro_recall = float(
                    np.mean([x[1] for x in metric_rows])
                )
                macro_f1 = float(
                    np.mean([x[2] for x in metric_rows])
                )

                total_tp = sum(x[3] for x in metric_rows)
                total_predictions = sum(x[4] for x in metric_rows)
                total_gold = sum(x[5] for x in metric_rows)

                micro_precision = (
                    total_tp / total_predictions
                    if total_predictions
                    else 0.0
                )
                micro_recall = (
                    total_tp / total_gold
                    if total_gold
                    else 0.0
                )
                micro_f1 = (
                    2 * micro_precision * micro_recall
                    / (micro_precision + micro_recall)
                    if micro_precision + micro_recall
                    else 0.0
                )

                rows.append(
                    {
                        "threshold": float(threshold),
                        "overlap_threshold": 0.0,
                        "max_phrases": int(max_phrases),
                        "macro_row_precision": macro_precision,
                        "macro_row_recall": macro_recall,
                        "macro_row_f1": macro_f1,
                        "micro_precision": micro_precision,
                        "micro_recall": micro_recall,
                        "micro_f1": micro_f1,
                    }
                )

    tuning = pd.DataFrame(rows)

    # Primary objective: macro row F1.
    # Tie breakers: macro recall, then macro precision.
    tuning = tuning.sort_values(
        ["macro_row_f1", "macro_row_recall", "macro_row_precision"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    best = tuning.iloc[0].to_dict()
    print(
        "\nBest aggregation setting: "
        f"threshold={best['threshold']:.2f}, "
        f"overlap={best['overlap_threshold']:.2f}, "
        f"max_phrases={int(best['max_phrases']) or 'all'}, "
        f"macro_row_f1={best['macro_row_f1']:.6f}"
    )

    return best, tuning


ANCHOR_WORDS = ['accident', 'achieve', 'achieved', 'alcohol', 'alive', 'alone', 'ambulance', 'anti', 'antidepressants', 'antis', 'anymore', 'attempt', 'attempted', 'attempting', 'attempts', 'benzos', 'blade', 'blades', 'bleed', 'bleeding', 'blew', 'blood', 'blow', 'blown', 'bought', 'breathe', 'breathing', 'bridge', 'bridges', 'building', 'bullet', 'bullets', 'burn', 'burned', 'burning', 'burns', 'buy', 'buying', 'car', 'cars', 'chest', 'choke', 'choked', 'choking', 'commit', 'committing', 'consider', 'considering', 'crash', 'crashes', 'crashing', 'crisis', 'cut', 'cuts', 'cutting', 'date', 'day', 'days', 'dead', 'death', 'deaths', 'did', 'die', 'died', 'dies', 'do', 'does', 'doing', 'done', 'drank', 'drink', 'drinking', 'drinks', 'drown', 'drowned', 'drowning', 'drowns', 'drug', 'drugs', 'drunk', 'dying', 'emptiness', 'empty', 'end', 'ended', 'ending', 'ends', 'enough', 'escape', 'escaping', 'exist', 'existed', 'existing', 'exists', 'face', 'fall', 'falling', 'falls', 'farewell', 'fell', 'final', 'finale', 'finally', 'free', 'freed', 'freedom', 'funeral', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'go', 'goes', 'going', 'gone', 'goodbye', 'got', 'grave', 'gun', 'guns', 'gunshot', 'hang', 'hanged', 'hanging', 'harm', 'harmed', 'harming', 'head', 'heart', 'help', 'helped', 'helping', 'helpless', 'helps', 'hope', 'hopeless', 'hopelessness', 'hopes', 'hoping', 'hospital', 'hospitals', 'hour', 'hours', 'hung', 'hurt', 'hurting', 'hurts', 'isolated', 'jump', 'jumped', 'jumping', 'kill', 'killed', 'killing', 'kills', 'kms', 'knife', 'knives', 'koolaid', 'last', 'lastly', 'leave', 'leaves', 'leaving', 'left', 'lethal', 'letter', 'letters', 'life', 'live', 'lives', 'living', 'loneliness', 'lonely', 'me', 'message', 'messages', 'method', 'methods', 'minute', 'minutes', 'month', 'months', 'myself', 'neck', 'night', 'note', 'notes', 'numb', 'numbness', 'nytols', 'od', 'oding', 'off', 'offed', 'order', 'ordered', 'ordering', 'over', 'overdose', 'overdosed', 'overdosing', 'overpass', 'pain', 'painful', 'painless', 'paracetamol', 'pill', 'pills', 'plan', 'planning', 'plans', 'poison', 'poisoned', 'poisoning', 'potassium', 'put', 'putting', 'razor', 'razors', 'ready', 'rope', 'second', 'seconds', 'self', 'selfharm', 'shoot', 'shooting', 'shoots', 'shot', 'slash', 'sleep', 'sleeping', 'slit', 'slitting', 'sodium', 'stab', 'stabbing', 'stabs', 'stay', 'stayed', 'staying', 'stays', 'stop', 'stopped', 'stopping', 'stops', 'suffocate', 'suffocated', 'suffocating', 'suicidal', 'suicidality', 'suicide', 'suicides', 'survival', 'survive', 'surviving', 'take', 'taken', 'takes', 'taking', 'throat', 'time', 'times', 'today', 'tomorrow', 'tonight', 'took', 'tracks', 'traffic', 'train', 'trapped', 'tried', 'tries', 'truck', 'try', 'trying', 'use', 'used', 'uses', 'using', 'valium', 'wake', 'waking', 'way', 'ways', 'week', 'weeks', 'window', 'wine', 'work', 'worked', 'working', 'works', 'worth', 'worthless', 'worthy', 'wrist', 'xanax', 'year', 'years', 'zoloft']

CANDIDATE_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*"
)


def _candidate_tokens(text: str) -> list[str]:
    return CANDIDATE_TOKEN_PATTERN.findall(text)


def generate_phrase_candidates(cfg: Config, output_path: Path) -> pd.DataFrame:
    """Generate E1-style phrase candidates directly from the raw posts file."""
    posts_path = Path(cfg.posts)
    if not posts_path.exists():
        raise FileNotFoundError(f"Posts file not found: {posts_path}")

    if posts_path.suffix.lower() in {".xlsx", ".xls"}:
        posts = pd.read_excel(posts_path)
    elif posts_path.suffix.lower() == ".tsv":
        posts = pd.read_csv(posts_path, sep="\t", encoding="utf-8-sig")
    else:
        posts = pd.read_csv(posts_path, encoding="utf-8-sig")

    if cfg.row_id_col not in posts.columns:
        posts[cfg.row_id_col] = posts.index

    text_col = cfg.post_col
    if text_col not in posts.columns:
        text_col = next((c for c in ("post", "post_y") if c in posts.columns), None)
    if text_col is None:
        raise KeyError(f"Could not find post/post_y in {posts_path}.")

    keyword_set = {w.lower().replace("’", "'") for w in ANCHOR_WORDS}
    windows = [
        ("word", 0, 0),
        ("one_left", 1, 0),
        ("one_right", 0, 1),
        ("one_both", 1, 1),
        ("two_left", 2, 0),
        ("two_right", 0, 2),
        ("two_both", 2, 2),
    ]

    rows: list[dict[str, Any]] = []
    for source_row, row in posts.iterrows():
        raw = row[text_col]
        text = "" if pd.isna(raw) else str(raw)
        if not text.strip():
            continue

        row_id = row[cfg.row_id_col]
        global_token_offset = 0
        generated_for_source = False
        for sentence_index, sentence in enumerate(text.split(".")):
            tokens = _candidate_tokens(sentence)
            if not tokens:
                continue
            normalized = [t.lower().replace("’", "'") for t in tokens]

            for idx, norm_token in enumerate(normalized):
                if norm_token not in keyword_set:
                    continue
                for window_type, left, right in windows:
                    start = idx - left
                    end = idx + right + 1
                    if start < 0 or end > len(tokens):
                        continue
                    candidate = " ".join(tokens[start:end])
                    item = {
                        "source_row": source_row,
                        "sentence_index": sentence_index,
                        "matched_word": tokens[idx],
                        "matched_word_normalized": norm_token,
                        "token_index": global_token_offset + idx,
                        "sentence_token_index": idx,
                        "window_type": window_type,
                        "left_words": left,
                        "right_words": right,
                        cfg.row_id_col: row_id,
                        cfg.candidate_col: candidate,
                    }
                    for id_col in ("anon_user_id", "user_id", "post_id"):
                        if id_col in posts.columns:
                            item[id_col] = row[id_col]
                    item[text_col] = text
                    rows.append(item)
                    generated_for_source = True
            global_token_offset += len(tokens)

        # Match the original E1 artifact: posts with no generated anchor-window
        # candidate receive one explicit "none" row.
        if not generated_for_source:
            item = {
                "source_row": source_row,
                "sentence_index": pd.NA,
                "matched_word": pd.NA,
                "matched_word_normalized": pd.NA,
                "token_index": pd.NA,
                "sentence_token_index": pd.NA,
                "window_type": "none",
                "left_words": pd.NA,
                "right_words": pd.NA,
                cfg.row_id_col: row_id,
                cfg.candidate_col: "none",
            }
            for id_col in ("anon_user_id", "user_id", "post_id"):
                if id_col in posts.columns:
                    item[id_col] = row[id_col]
            item[text_col] = text
            rows.append(item)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Candidate generation produced zero candidates.")

    result = result.drop_duplicates(
        subset=["source_row", "matched_word_normalized", "window_type", cfg.candidate_col]
    ).reset_index(drop=True)

    if cfg.expected_candidate_count > 0 and len(result) != cfg.expected_candidate_count:
        raise ValueError(
            f"Candidate generation produced {len(result):,} rows; "
            f"expected {cfg.expected_candidate_count:,}. "
            "This usually means the raw posts file differs from the dataset used "
            "to create the reference E1 candidate artifact."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Generated candidates:   {len(result):,}")
    print(f"Candidate row_ids:      {result[cfg.row_id_col].nunique():,}")
    print(f"Saved candidates:       {output_path.resolve()}")
    return result



def generate_candidates_for_inference(
    input_path: str | Path,
    cfg: Config,
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate the same E1-style candidates for an arbitrary inference file.

    Unlike training candidate generation, no fixed expected row-count check is
    applied because leaderboard.xlsx has a different number of posts.
    """
    posts = load_any_table(input_path)

    if cfg.row_id_col not in posts.columns:
        posts[cfg.row_id_col] = posts.index

    text_col = cfg.post_col
    if text_col not in posts.columns:
        text_col = next((c for c in ("post", "post_y") if c in posts.columns), None)
    if text_col is None:
        raise KeyError(
            f"Could not find post/post_y in inference file {input_path}."
        )

    keyword_set = {w.lower().replace("’", "'") for w in ANCHOR_WORDS}
    windows = [
        ("word", 0, 0),
        ("one_left", 1, 0),
        ("one_right", 0, 1),
        ("one_both", 1, 1),
        ("two_left", 2, 0),
        ("two_right", 0, 2),
        ("two_both", 2, 2),
    ]

    rows: list[dict[str, Any]] = []

    for source_row, row in posts.iterrows():
        raw = row[text_col]
        post_text = "" if pd.isna(raw) else str(raw)
        row_id = row[cfg.row_id_col]
        generated_for_source = False
        global_token_offset = 0

        if post_text.strip():
            for sentence_index, sentence in enumerate(post_text.split(".")):
                tokens = _candidate_tokens(sentence)
                if not tokens:
                    continue

                normalized = [
                    token.lower().replace("’", "'")
                    for token in tokens
                ]

                for token_index, normalized_token in enumerate(normalized):
                    if normalized_token not in keyword_set:
                        continue

                    for window_type, left, right in windows:
                        start = token_index - left
                        end = token_index + right + 1

                        if start < 0 or end > len(tokens):
                            continue

                        candidate = " ".join(tokens[start:end])
                        item: dict[str, Any] = {
                            "source_row": source_row,
                            "sentence_index": sentence_index,
                            "matched_word": tokens[token_index],
                            "matched_word_normalized": normalized_token,
                            "token_index": global_token_offset + token_index,
                            "sentence_token_index": token_index,
                            "window_type": window_type,
                            "left_words": left,
                            "right_words": right,
                            cfg.row_id_col: row_id,
                            cfg.candidate_col: candidate,
                            cfg.post_col: post_text,
                        }

                        for id_col in ("anon_user_id", "user_id", "post_id"):
                            if id_col in posts.columns:
                                item[id_col] = row[id_col]

                        rows.append(item)
                        generated_for_source = True

                global_token_offset += len(tokens)

        if not generated_for_source:
            item = {
                "source_row": source_row,
                "sentence_index": pd.NA,
                "matched_word": pd.NA,
                "matched_word_normalized": pd.NA,
                "token_index": pd.NA,
                "sentence_token_index": pd.NA,
                "window_type": "none",
                "left_words": pd.NA,
                "right_words": pd.NA,
                cfg.row_id_col: row_id,
                cfg.candidate_col: "none",
                cfg.post_col: post_text,
            }
            for id_col in ("anon_user_id", "user_id", "post_id"):
                if id_col in posts.columns:
                    item[id_col] = row[id_col]
            rows.append(item)

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Leaderboard candidate generation produced zero rows.")

    result = result.drop_duplicates(
        subset=[
            "source_row",
            "matched_word_normalized",
            "window_type",
            cfg.candidate_col,
        ]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Leaderboard candidates: {len(result):,}")
    print(f"Leaderboard row_ids:    {result[cfg.row_id_col].nunique():,}")
    print(f"Saved candidates:       {output_path.resolve()}")

    return result


def prepare_inference_candidates(
    candidates: pd.DataFrame,
    risk_file: str | Path,
    cfg: Config,
) -> tuple[pd.DataFrame, str]:
    """
    Join leaderboard candidates to suicide risk and build model_text.

    The risk file is joined on normalized row_id so values such as 10 and 10.0
    match reliably.
    """
    candidates = candidates.copy()
    risk_df = load_any_table(risk_file).copy()

    if cfg.row_id_col not in candidates.columns:
        raise KeyError(f"Candidate data missing {cfg.row_id_col!r}.")
    if cfg.row_id_col not in risk_df.columns:
        raise KeyError(f"Risk file missing {cfg.row_id_col!r}.")

    risk_col = detect_risk_column(risk_df, None)
    if risk_col is None:
        raise KeyError(
            "Could not detect suicide-risk column in "
            f"{risk_file}. Expected e.g. risk_level or suicide risk."
        )

    candidates["_join_row_id"] = candidates[cfg.row_id_col].map(
        normalize_join_row_id
    )
    risk_df["_join_row_id"] = risk_df[cfg.row_id_col].map(
        normalize_join_row_id
    )

    nonempty_risks = risk_df[risk_df["_join_row_id"].ne("")].copy()

    duplicate_mask = nonempty_risks["_join_row_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = (
            nonempty_risks.loc[duplicate_mask, "_join_row_id"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Risk file contains duplicate row_id values. "
            f"Examples: {duplicate_values}"
        )

    merged = candidates.merge(
        nonempty_risks[["_join_row_id", risk_col]],
        on="_join_row_id",
        how="left",
        validate="many_to_one",
    )

    missing = merged[risk_col].isna()
    if missing.any():
        missing_ids = (
            merged.loc[missing, cfg.row_id_col]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{int(missing.sum()):,} candidate rows have no suicide-risk value "
            f"in {risk_file}. Example row_ids: {missing_ids}"
        )

    merged[risk_col] = merged[risk_col].map(clean_text)
    merged[cfg.candidate_col] = merged[cfg.candidate_col].map(clean_text)
    merged[cfg.post_col] = merged[cfg.post_col].map(clean_text)

    # Explicit "none" rows are retained for final row coverage but are not
    # scored by the model.
    merged["model_text"] = (
        "Candidate: "
        + merged[cfg.candidate_col]
        + "\nPost: "
        + merged[cfg.post_col]
    )

    return merged, risk_col


def predict_unlabeled_candidates(
    trainer: Trainer,
    tokenizer: Any,
    df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Score inference candidates without requiring gold labels."""
    working = df.copy().reset_index(drop=True)

    score_mask = (
        working[cfg.candidate_col].map(clean_text).ne("")
        & working[cfg.candidate_col].map(clean_text).str.casefold().ne("none")
    )

    working["prob_not_evidence"] = 1.0
    working["prob_evidence"] = 0.0
    working["predicted_label_0_5"] = 0

    if not score_mask.any():
        return working

    score_df = working.loc[score_mask].copy()

    class InferenceDataset(Dataset):
        def __init__(self, texts: list[str]) -> None:
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=cfg.max_length,
                padding=False,
            )

        def __len__(self) -> int:
            return len(self.encodings["input_ids"])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                key: values[index]
                for key, values in self.encodings.items()
            }

    dataset = InferenceDataset(
        score_df["model_text"].astype(str).tolist()
    )
    original_compute_metrics = trainer.compute_metrics
    trainer.compute_metrics = None
    try:
        prediction_output = trainer.predict(dataset)
    finally:
        trainer.compute_metrics = original_compute_metrics

    logits = prediction_output.predictions
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=-1)

    working.loc[score_mask, "prob_not_evidence"] = probabilities[:, 0]
    working.loc[score_mask, "prob_evidence"] = probabilities[:, 1]
    working.loc[score_mask, "predicted_label_0_5"] = predictions

    working["predicted_label_0_5"] = (
        working["predicted_label_0_5"].astype(int)
    )
    return working


def aggregate_inference_predictions(
    prediction_df: pd.DataFrame,
    cfg: Config,
    threshold: float,
    overlap_threshold: float,
    max_phrases: int,
    risk_col: str,
) -> pd.DataFrame:
    """
    Aggregate leaderboard evidence using the selected settings.

    Order is:
      score -> threshold/overlap/max-phrases aggregation
      -> Indicator override
      -> adjacent-span merge
      -> output
    """
    rows: list[dict[str, Any]] = []

    for row_id, group in prediction_df.groupby(
        cfg.row_id_col,
        sort=False,
        dropna=False,
    ):
        phrases, probabilities = select_phrases_from_group(
            group,
            candidate_col=cfg.candidate_col,
            threshold=threshold,
            overlap_threshold=overlap_threshold,
            max_phrases=max_phrases,
        )

        risk_values = [
            clean_text(value)
            for value in group[risk_col]
            if clean_text(value)
        ]
        risk_value = risk_values[0] if risk_values else ""

        indicator_override = risk_value.casefold() == "indicator"
        if indicator_override:
            phrases = []
            probabilities = []

        post_values = [
            clean_text(value)
            for value in group[cfg.post_col]
            if clean_text(value)
        ]
        post = post_values[0] if post_values else ""

        # IMPORTANT: adjacency is handled only after ordinary aggregation.
        if not indicator_override:
            phrases, probabilities = merge_adjacent_candidates_in_post(
                post=post,
                phrases=phrases,
                probabilities=probabilities,
            )

        rows.append(
            {
                cfg.row_id_col: row_id,
                "predicted": "; ".join(phrases) if phrases else "none",
                "prob_evidence": "; ".join(
                    f"{probability:.6f}"
                    for probability in probabilities
                ),
                "suicide_risk": risk_value,
                "indicator_override": indicator_override,
                "predicted_phrase_count": len(phrases),
                "post": post,
            }
        )

    return pd.DataFrame(rows)


def run_leaderboard_inference(
    cfg: Config,
    leaderboard_path: str | Path,
    risk_file: str | Path,
    model_dir: Path,
    metric_dir: Path,
    output_dir: Path,
) -> None:
    """Generate, score, and aggregate evidence for leaderboard.xlsx."""
    leaderboard_path = Path(leaderboard_path)
    risk_file = Path(risk_file)

    if not leaderboard_path.exists():
        raise FileNotFoundError(
            f"Leaderboard file not found: {leaderboard_path}"
        )
    if not risk_file.exists():
        raise FileNotFoundError(
            f"Suicide-risk file not found: {risk_file}"
        )
    if not saved_model_complete(model_dir):
        raise FileNotFoundError(
            f"Completed model not found at {model_dir}."
        )

    settings_path = metric_dir / "best_aggregation_settings.json"
    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)

        threshold = float(settings["threshold"])
        # Leaderboard aggregation intentionally uses zero overlap threshold.
        # Any pair sharing at least one token is treated as overlapping.
        overlap_threshold = 0.0
        max_phrases = int(settings["max_phrases"])
        settings_source = (
            str(settings_path.resolve())
            + " (overlap fixed globally at 0.0)"
        )
    else:
        threshold = cfg.default_threshold
        overlap_threshold = 0.0
        max_phrases = cfg.default_max_phrases
        settings_source = "Config defaults (overlap fixed globally at 0.0)"

    leaderboard_dir = output_dir / "leaderboard_inference"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = leaderboard_dir / "LB_E1_phrase_candidates.csv"
    scored_path = leaderboard_dir / "LB_candidate_scores.csv"
    aggregated_path = leaderboard_dir / "LB_aggregated_evidence.csv"

    print("\n=== Leaderboard inference ===")
    print(f"Leaderboard:           {leaderboard_path.resolve()}")
    print(f"Suicide risk:          {risk_file.resolve()}")
    print(f"Aggregation settings:  {settings_source}")
    print(f"  threshold:           {threshold:.3f}")
    print(f"  overlap threshold:   {overlap_threshold:.3f}")
    print(f"  max phrases:         {max_phrases or 'unlimited'}")

    if candidates_path.exists() and not cfg.force:
        print(f"Reusing leaderboard candidates: {candidates_path.resolve()}")
        candidates = pd.read_csv(candidates_path, encoding="utf-8-sig")
    else:
        print("Generating leaderboard candidates...")
        candidates = generate_candidates_for_inference(
            leaderboard_path,
            cfg,
            candidates_path,
        )

    prepared, risk_col = prepare_inference_candidates(
        candidates,
        risk_file,
        cfg,
    )

    if scored_path.exists() and not cfg.force:
        print(f"Reusing leaderboard candidate scores: {scored_path.resolve()}")
        scored = pd.read_csv(scored_path, encoding="utf-8-sig")
    else:
        print("Scoring leaderboard candidates...")
        trainer, tokenizer = load_saved_model_for_prediction(
            model_dir,
            cfg,
            leaderboard_dir / "_prediction_trainer",
        )
        scored = predict_unlabeled_candidates(
            trainer,
            tokenizer,
            prepared,
            cfg,
        )
        scored.to_csv(
            scored_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved candidate scores: {scored_path.resolve()}")

    print("Aggregating leaderboard evidence...")
    aggregated = aggregate_inference_predictions(
        scored,
        cfg=cfg,
        threshold=threshold,
        overlap_threshold=overlap_threshold,
        max_phrases=max_phrases,
        risk_col=risk_col,
    )
    aggregated.to_csv(
        aggregated_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "leaderboard_file": str(leaderboard_path.resolve()),
        "risk_file": str(risk_file.resolve()),
        "model_dir": str(model_dir.resolve()),
        "candidate_rows": int(len(scored)),
        "row_ids": int(aggregated[cfg.row_id_col].nunique()),
        "rows_with_evidence": int(aggregated["predicted"].ne("none").sum()),
        "indicator_overrides": int(aggregated["indicator_override"].sum()),
        "aggregation_settings": {
            "threshold": threshold,
            "overlap_threshold": overlap_threshold,
            "leaderboard_overlap_forced": True,
            "max_phrases": max_phrases,
            "source": settings_source,
        },
        "candidate_output": str(scored_path.resolve()),
        "aggregated_output": str(aggregated_path.resolve()),
    }
    save_json(summary, leaderboard_dir / "LB_inference_summary.json")

    print(f"Aggregated leaderboard: {aggregated_path.resolve()}")
    print(
        f"Rows with evidence:     "
        f"{summary['rows_with_evidence']:,}/{summary['row_ids']:,}"
    )
    print(f"Indicator overrides:    {summary['indicator_overrides']:,}")



# ---------------------------------------------------------------------------
# Rebuild training pairs locally
# ---------------------------------------------------------------------------

def rebuild_training_pairs(cfg: Config, output_path: Path, candidates_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reproduce prepare_evidence_training_data locally from posts + candidates."""
    posts_path = Path(cfg.posts)
    if posts_path.suffix.lower() in {".xlsx", ".xls"}:
        posts = pd.read_excel(posts_path)
    else:
        posts = pd.read_csv(posts_path, encoding="utf-8-sig")

    if candidates_df is not None:
        candidates = candidates_df.copy()
    elif cfg.candidates:
        candidates_path = Path(cfg.candidates)
        if not candidates_path.exists():
            raise FileNotFoundError(f"Candidate file not found: {candidates_path}")
        if candidates_path.suffix.lower() == ".tsv":
            candidates = pd.read_csv(candidates_path, sep="\t", encoding="utf-8-sig")
        else:
            candidates = pd.read_csv(candidates_path, encoding="utf-8-sig")
    else:
        raise ValueError("No candidates were supplied or generated.")

    if cfg.row_id_col not in posts.columns:
        posts[cfg.row_id_col] = posts.index

    post_col = cfg.post_col
    if post_col not in posts.columns:
        post_col = next((c for c in ("post", "post_y") if c in posts.columns), None)
    if post_col is None:
        raise KeyError("Could not find post/post_y in the posts file.")

    gold_col = cfg.gold_col
    if gold_col not in posts.columns:
        gold_col = next(
            (c for c in (
                "evidence for suicide risk level",
                "evidence for suicide level",
                "gold_phrases",
                "gold_evidences",
            ) if c in posts.columns),
            None,
        )
    if gold_col is None:
        raise KeyError("Could not find the gold-evidence column in the posts file.")

    required = {cfg.row_id_col, cfg.candidate_col}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"Candidate file missing columns: {sorted(missing)}")

    def norm(v: object) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return re.sub(r"\s+", " ", str(v).casefold()).strip()

    def split_gold(v: object) -> list[str]:
        t = norm(v)
        if not t or t == "none":
            return []
        return [x.strip() for x in t.split(";") if x.strip() and x.strip() != "none"]

    def wc(v: object) -> int:
        return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", norm(v), flags=re.UNICODE))

    def matches(candidate: str, gold: str) -> bool:
        c, g = norm(candidate), norm(gold)
        if not c or not g:
            return False
        cw, gw = wc(c), wc(g)
        if not cw or not gw:
            return False
        ratio = cw / gw
        if ratio < cfg.min_match_word_ratio or ratio > cfg.max_match_word_ratio:
            return False
        return c in g or g in c

    posts = posts.copy()
    posts["_gold_list"] = posts[gold_col].map(split_gold)

    keep = [cfg.row_id_col, post_col, gold_col, "_gold_list"]
    for col in ("anon_user_id", "user_id", "post_id", "suicide risk", "suicide_risk"):
        if col in posts.columns and col not in keep:
            keep.append(col)

    merged = candidates.merge(
        posts[keep],
        on=cfg.row_id_col,
        how="left",
        validate="many_to_one",
        suffixes=("", "_post"),
    )

    output_rows = []
    for _, row in merged.iterrows():
        candidate_raw = row[cfg.candidate_col]
        candidate = norm(candidate_raw)
        if not candidate or candidate == "none":
            continue
        golds = row["_gold_list"] if isinstance(row["_gold_list"], list) else []
        matched = [g for g in golds if matches(candidate, g)]

        out = row.drop(labels=["_gold_list"]).to_dict()
        out[cfg.candidate_col] = str(candidate_raw).strip()
        out[cfg.post_col] = row[post_col]
        out[cfg.gold_col] = "; ".join(golds)
        out["matched_gold"] = "; ".join(matched)
        out[cfg.label_col] = int(bool(matched))
        out["candidate_word_count"] = wc(candidate)
        out["matched_gold_count"] = len(matched)
        output_rows.append(out)

    result = pd.DataFrame(output_rows)
    if result.empty:
        raise ValueError("Rebuilding training pairs produced zero rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Rebuilt training pairs: {len(result):,}")
    print(f"Positive pairs:         {int(result[cfg.label_col].sum()):,}")
    print(f"Negative pairs:         {int((result[cfg.label_col] == 0).sum()):,}")
    print(f"Saved prepared pairs:   {output_path.resolve()}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def latest_trainer_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the highest numbered checkpoint-* directory."""
    if not checkpoint_dir.exists():
        return None

    found: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            found.append((int(match.group(1)), path))

    if not found:
        return None

    found.sort(key=lambda item: item[0])
    return found[-1][1]


def saved_model_complete(model_dir: Path) -> bool:
    """Conservative check that a Hugging Face model + tokenizer were saved."""
    if not model_dir.is_dir():
        return False

    has_config = (model_dir / "config.json").exists()
    has_weights = (
        (model_dir / "model.safetensors").exists()
        or (model_dir / "pytorch_model.bin").exists()
    )
    has_tokenizer = any(
        (model_dir / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "spm.model",
            "sentencepiece.bpe.model",
        )
    )
    return has_config and has_weights and has_tokenizer


def load_saved_model_for_prediction(
    model_dir: Path,
    cfg: Config,
    prediction_output_dir: Path,
) -> tuple[Trainer, Any]:
    """Load a completed model without retraining it."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = resolve_device(cfg.device)
    use_fp16 = bool(cfg.fp16 and device.type == "cuda")
    use_bf16 = bool(cfg.bf16 and device.type == "cuda")

    args = TrainingArguments(
        output_dir=str(prediction_output_dir),
        per_device_eval_batch_size=cfg.eval_batch_size,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return trainer, tokenizer



def main() -> None:
    args = parse_args()
    cfg = Config(
        training_pairs=args.training_pairs,
        posts=args.posts,
        candidates=args.candidates,
        reviewed_negatives=args.reviewed_negatives,
        expected_candidate_count=args.expected_candidate_count,
        output_dir=args.output_dir,
        model_name=args.model_name,
        row_id_col=args.row_id_col,
        post_col=args.post_col,
        candidate_col=args.candidate_col,
        label_col=args.label_col,
        gold_col=args.gold_col,
        risk_col=args.risk_col,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        test_size=args.test_size,
        validation_size_from_remaining=args.validation_size_from_remaining,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        device=args.device,
        hard_negative_extra_copies=args.hard_negative_extra_copies,
        use_reviewed_annotation_errors_as_positive=not args.do_not_promote_annotation_errors,
        drop_ambiguous=not args.keep_ambiguous,
        tune_aggregation=not args.no_tune_aggregation,
        threshold_grid=args.threshold_grid,
        overlap_grid=args.overlap_grid,
        max_phrases_grid=args.max_phrases_grid,
        default_threshold=args.default_threshold,
        default_overlap_threshold=args.default_overlap_threshold,
        default_max_phrases=args.default_max_phrases,
        min_match_word_ratio=args.min_match_word_ratio,
        max_match_word_ratio=args.max_match_word_ratio,
        force=args.force,
        no_resume=args.no_resume,
    )

    output_dir = Path(cfg.output_dir)
    prepared_dir = output_dir / "prepared"
    split_dir = output_dir / "splits"
    prediction_dir = output_dir / "predictions"
    metric_dir = output_dir / "metrics"
    model_dir = output_dir / "best_model"
    checkpoint_dir = output_dir / "checkpoints"

    for directory in [
        output_dir,
        prepared_dir,
        split_dir,
        prediction_dir,
        metric_dir,
        model_dir,
        checkpoint_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    final_metrics_path = metric_dir / "final_metrics.json"
    final_test_rows_path = prediction_dir / "test_aggregated_predictions.csv"

    print("=== Local Evidence Pipeline ===")
    print(f"Output directory:    {output_dir.resolve()}")
    print(f"Model:               {cfg.model_name}")
    print(f"Resolved device:     {resolve_device(cfg.device)}")
    print("Overlap threshold:   0.0 (fixed globally)")
    print(f"Resume mode:         {'OFF (--force)' if cfg.force else 'ON'}")

    # If the entire run already completed, leave it untouched.
    if final_metrics_path.exists() and final_test_rows_path.exists() and not cfg.force:
        print("\nPipeline already complete; reusing final outputs.")
        print(f"Final metrics:        {final_metrics_path.resolve()}")
        print(f"Test row predictions: {final_test_rows_path.resolve()}")
        print("Use --force to rebuild/retrain everything.")

        if not args.skip_leaderboard:
            run_leaderboard_inference(
                cfg=cfg,
                leaderboard_path=args.leaderboard,
                risk_file=args.risk_file,
                model_dir=model_dir,
                metric_dir=metric_dir,
                output_dir=output_dir,
            )
        return

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    set_seed(cfg.seed)

    # --------------------------------------------------------------
    # Stage 1: candidate generation
    # --------------------------------------------------------------
    generated_candidates = None
    generated_candidate_path = prepared_dir / "E1_phrase_candidates.csv"

    if cfg.training_pairs:
        print(f"\nUsing prebuilt training pairs: {Path(cfg.training_pairs).resolve()}")
    elif cfg.candidates:
        print(f"\nUsing prebuilt candidates: {Path(cfg.candidates).resolve()}")
    elif generated_candidate_path.exists() and not cfg.force:
        print(f"\nReusing generated candidates: {generated_candidate_path.resolve()}")
        generated_candidates = pd.read_csv(
            generated_candidate_path,
            encoding="utf-8-sig",
        )
        if cfg.expected_candidate_count > 0 and len(generated_candidates) != cfg.expected_candidate_count:
            raise ValueError(
                "Existing candidate file failed sanity check: "
                f"{len(generated_candidates):,} rows, expected "
                f"{cfg.expected_candidate_count:,}. Use --force to regenerate it."
            )
    else:
        print("\nGenerating phrase candidates...")
        generated_candidates = generate_phrase_candidates(
            cfg,
            generated_candidate_path,
        )

    # --------------------------------------------------------------
    # Stage 2: candidate/gold training pairs
    # --------------------------------------------------------------
    prepared_pairs_path = prepared_dir / "evidence_training_pairs.csv"

    if cfg.training_pairs:
        base_df = pd.read_csv(cfg.training_pairs, encoding="utf-8-sig")
    elif prepared_pairs_path.exists() and not cfg.force:
        print(f"Reusing training pairs: {prepared_pairs_path.resolve()}")
        base_df = pd.read_csv(prepared_pairs_path, encoding="utf-8-sig")
    else:
        print("Building training pairs...")
        base_df = rebuild_training_pairs(
            cfg,
            prepared_pairs_path,
            candidates_df=generated_candidates,
        )

    # --------------------------------------------------------------
    # Stage 3: reviewed labels
    # --------------------------------------------------------------
    reviewed_df = pd.read_csv(cfg.reviewed_negatives, encoding="utf-8-sig")
    reviewed_adjusted, review_stats = apply_reviewed_labels(
        base_df,
        reviewed_df,
        row_id_col=cfg.row_id_col,
        candidate_col=cfg.candidate_col,
        label_col=cfg.label_col,
        promote_annotation_errors=cfg.use_reviewed_annotation_errors_as_positive,
        drop_ambiguous=cfg.drop_ambiguous,
    )
    data = prepare_dataframe(reviewed_adjusted, cfg)
    risk_col = detect_risk_column(data, cfg.risk_col)

    print("\nHuman review integration:")
    print(f"  Unique reviewed pairs:  {review_stats['reviewed_unique_pairs']:,}")
    print(
        "  Annotation errors -> 1: "
        f"{review_stats['annotation_error_rows_promoted']:,} source rows"
    )
    print(
        "  Confirmed TN rows:      "
        f"{review_stats['true_negative_rows_confirmed']:,} source rows"
    )
    print(
        "  Ambiguous removed:      "
        f"{review_stats['ambiguous_rows_removed']:,} source rows"
    )
    print(f"  Risk column:            {risk_col or '(none detected)'}")

    # --------------------------------------------------------------
    # Stage 4: grouped splits
    # --------------------------------------------------------------
    raw_train_path = split_dir / "train_split_before_hard_negative_oversampling.csv"
    train_path = split_dir / "train_split.csv"
    validation_path = split_dir / "validation_split.csv"
    test_path = split_dir / "test_split.csv"

    reusable_splits = (
        raw_train_path.exists()
        and train_path.exists()
        and validation_path.exists()
        and test_path.exists()
        and not cfg.force
    )

    if reusable_splits:
        print(f"\nReusing grouped splits: {split_dir.resolve()}")
        raw_train_df = pd.read_csv(raw_train_path, encoding="utf-8-sig")
        train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        validation_df = pd.read_csv(validation_path, encoding="utf-8-sig")
        test_df = pd.read_csv(test_path, encoding="utf-8-sig")
        oversampling_stats = {
            "hard_negative_extra_copies": cfg.hard_negative_extra_copies,
            "hard_negative_rows_before_oversampling": int(
                raw_train_df.get("_review_decision", pd.Series(dtype=str))
                .eq("true_negative")
                .sum()
            ),
            "hard_negative_extra_rows_added": max(0, len(train_df) - len(raw_train_df)),
        }
    else:
        print("\nCreating grouped splits...")
        train_df, validation_df, test_df = grouped_split(data, cfg)
        raw_train_df = train_df.copy()
        train_df, oversampling_stats = oversample_hard_negatives(
            train_df,
            extra_copies=cfg.hard_negative_extra_copies,
        )

        raw_train_df.to_csv(raw_train_path, index=False, encoding="utf-8-sig")
        train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
        validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
        test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    split_stats = {
        "train_rows_before_hard_negative_oversampling": len(raw_train_df),
        "train_rows_after_hard_negative_oversampling": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "train_row_ids": raw_train_df[cfg.row_id_col].nunique(),
        "validation_row_ids": validation_df[cfg.row_id_col].nunique(),
        "test_row_ids": test_df[cfg.row_id_col].nunique(),
        "train_label_counts_before_oversampling": (
            raw_train_df[cfg.label_col].value_counts().sort_index().to_dict()
        ),
        "train_label_counts_after_oversampling": (
            train_df[cfg.label_col].value_counts().sort_index().to_dict()
        ),
        "validation_label_counts": (
            validation_df[cfg.label_col].value_counts().sort_index().to_dict()
        ),
        "test_label_counts": (
            test_df[cfg.label_col].value_counts().sort_index().to_dict()
        ),
        **oversampling_stats,
    }

    save_json(asdict(cfg), output_dir / "config.json")
    save_json(review_stats, metric_dir / "review_integration.json")
    save_json(split_stats, metric_dir / "split_stats.json")

    # --------------------------------------------------------------
    # Stage 5: model training / checkpoint resume
    # --------------------------------------------------------------
    best_checkpoint_value = None
    best_metric_value = None

    if saved_model_complete(model_dir) and not cfg.force:
        print(f"\nReusing completed model: {model_dir.resolve()}")
        trainer, tokenizer = load_saved_model_for_prediction(
            model_dir,
            cfg,
            output_dir / "_prediction_trainer",
        )
    else:
        resume_checkpoint = None
        if not cfg.force and not cfg.no_resume:
            resume_checkpoint = latest_trainer_checkpoint(checkpoint_dir)

        print("\nTraining model...")
        if resume_checkpoint is not None:
            print(f"Latest checkpoint found: {resume_checkpoint.resolve()}")

        trainer, tokenizer = train_model(
            train_df=train_df,
            validation_df=validation_df,
            cfg=cfg,
            model_dir=model_dir,
            checkpoint_dir=checkpoint_dir,
            resume_checkpoint=resume_checkpoint,
        )
        best_checkpoint_value = trainer.state.best_model_checkpoint
        best_metric_value = trainer.state.best_metric

    # --------------------------------------------------------------
    # Stage 6: validation candidate predictions
    # --------------------------------------------------------------
    validation_predictions_path = prediction_dir / "validation_candidate_predictions.csv"
    validation_candidate_metrics_path = metric_dir / "validation_candidate_metrics.json"

    if (
        validation_predictions_path.exists()
        and validation_candidate_metrics_path.exists()
        and not cfg.force
    ):
        print(f"\nReusing validation predictions: {validation_predictions_path.resolve()}")
        validation_predictions = pd.read_csv(
            validation_predictions_path,
            encoding="utf-8-sig",
        )
        with validation_candidate_metrics_path.open("r", encoding="utf-8") as handle:
            validation_candidate_metrics = json.load(handle)
    else:
        print("\nPredicting validation candidates...")
        validation_predictions, validation_candidate_metrics = predict_dataframe(
            trainer,
            tokenizer,
            validation_df,
            cfg,
        )
        validation_predictions.to_csv(
            validation_predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        save_json(
            validation_candidate_metrics,
            validation_candidate_metrics_path,
        )

    # --------------------------------------------------------------
    # Stage 7: aggregation tuning
    # --------------------------------------------------------------
    best_settings_path = metric_dir / "best_aggregation_settings.json"
    tuning_table_path = metric_dir / "aggregation_grid_search.csv"

    if cfg.tune_aggregation:
        if best_settings_path.exists() and not cfg.force:
            print(f"Reusing aggregation settings: {best_settings_path.resolve()}")
            with best_settings_path.open("r", encoding="utf-8") as handle:
                best_settings = json.load(handle)
        else:
            best_settings, tuning_table = tune_aggregation(
                validation_predictions,
                cfg=cfg,
                risk_col=risk_col,
            )
            tuning_table.to_csv(
                tuning_table_path,
                index=False,
                encoding="utf-8-sig",
            )
            save_json(best_settings, best_settings_path)

        threshold = float(best_settings["threshold"])
        overlap_threshold = 0.0
        max_phrases = int(best_settings["max_phrases"])
    else:
        threshold = cfg.default_threshold
        overlap_threshold = 0.0
        max_phrases = cfg.default_max_phrases
        best_settings = {
            "threshold": threshold,
            "overlap_threshold": overlap_threshold,
            "max_phrases": max_phrases,
            "source": "defaults (--no-tune-aggregation)",
        }

    # Always recompute aggregated validation rows because aggregation code may
    # have changed even when candidate probabilities are reused.
    validation_rows, validation_row_metrics = aggregate_split(
        validation_predictions,
        cfg=cfg,
        threshold=threshold,
        overlap_threshold=overlap_threshold,
        max_phrases=max_phrases,
        risk_col=risk_col,
    )
    validation_rows.to_csv(
        prediction_dir / "validation_aggregated_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSelected aggregation settings:")
    print(f"  probability threshold: {threshold:.3f}")
    print(f"  overlap threshold:     {overlap_threshold:.3f}")
    print(f"  max phrases:           {max_phrases or 'unlimited'}")
    print(f"  validation row F1:     {validation_row_metrics['macro_row_f1']:.6f}")

    # --------------------------------------------------------------
    # Stage 8: untouched test candidate predictions
    # --------------------------------------------------------------
    test_predictions_path = prediction_dir / "test_candidate_predictions.csv"
    test_candidate_metrics_path = metric_dir / "test_candidate_metrics.json"

    if (
        test_predictions_path.exists()
        and test_candidate_metrics_path.exists()
        and not cfg.force
    ):
        print(f"\nReusing test predictions: {test_predictions_path.resolve()}")
        test_predictions = pd.read_csv(
            test_predictions_path,
            encoding="utf-8-sig",
        )
        with test_candidate_metrics_path.open("r", encoding="utf-8") as handle:
            test_candidate_metrics = json.load(handle)
    else:
        print("\nPredicting untouched grouped test split...")
        test_predictions, test_candidate_metrics = predict_dataframe(
            trainer,
            tokenizer,
            test_df,
            cfg,
        )
        test_predictions.to_csv(
            test_predictions_path,
            index=False,
            encoding="utf-8-sig",
        )
        save_json(test_candidate_metrics, test_candidate_metrics_path)

    # Final aggregation is deliberately recomputed on every incomplete run so
    # changes such as adjacent-span merging are reflected immediately.
    test_rows, test_row_metrics = aggregate_split(
        test_predictions,
        cfg=cfg,
        threshold=threshold,
        overlap_threshold=overlap_threshold,
        max_phrases=max_phrases,
        risk_col=risk_col,
    )
    test_rows.to_csv(
        final_test_rows_path,
        index=False,
        encoding="utf-8-sig",
    )

    final_metrics = {
        "review_integration": review_stats,
        "split_stats": split_stats,
        "aggregation_settings": {
            "threshold": threshold,
            "overlap_threshold": overlap_threshold,
            "max_phrases": max_phrases,
            "validation_best_grid_row": best_settings,
        },
        "validation_candidate_metrics_at_0_5": validation_candidate_metrics,
        "validation_row_metrics": validation_row_metrics,
        "test_candidate_metrics_at_0_5": test_candidate_metrics,
        "test_row_metrics": test_row_metrics,
        "best_checkpoint": best_checkpoint_value,
        "best_validation_gold_phrase_coverage": best_metric_value,
        "model_selection": {
            "metric": "gold_phrase_coverage",
            "positive_decision_threshold": 0.5,
            "learning_rate": cfg.learning_rate,
        },
        "resume": {
            "force": cfg.force,
            "no_resume": cfg.no_resume,
            "latest_checkpoint_seen": (
                str(latest_trainer_checkpoint(checkpoint_dir))
                if latest_trainer_checkpoint(checkpoint_dir)
                else None
            ),
        },
    }
    save_json(final_metrics, final_metrics_path)

    print("\n=== Final test results ===")
    print(f"Candidate precision @0.5: {test_candidate_metrics['precision']:.6f}")
    print(f"Candidate recall @0.5:    {test_candidate_metrics['recall']:.6f}")
    print(f"Candidate F1 @0.5:        {test_candidate_metrics['f1']:.6f}")
    print()
    print(f"Macro row precision:      {test_row_metrics['macro_row_precision']:.6f}")
    print(f"Macro row recall:         {test_row_metrics['macro_row_recall']:.6f}")
    print(f"Macro row F1:             {test_row_metrics['macro_row_f1']:.6f}")
    print(f"Micro row F1:             {test_row_metrics['micro_f1']:.6f}")
    print()
    print(f"Best model:               {model_dir.resolve()}")
    print(f"Final metrics:            {final_metrics_path.resolve()}")
    print(f"Test row predictions:     {final_test_rows_path.resolve()}")

    if not args.skip_leaderboard:
        run_leaderboard_inference(
            cfg=cfg,
            leaderboard_path=args.leaderboard,
            risk_file=args.risk_file,
            model_dir=model_dir,
            metric_dir=metric_dir,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
