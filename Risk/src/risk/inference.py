from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from risk_deberta_large.metrics import probability_softmax


class TokenizedTextDataset(Dataset):
    """
    Same tokenization contract as the original training implementation.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return {
            key: value[index]
            for key, value in self.encodings.items()
        }


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(requested)


def checkpoint_path(
    model_root: Path,
    branch: str,
    seed: int,
    fold: int,
) -> Path:
    path = (
        model_root
        / branch
        / f"seed{seed}"
        / f"fold{fold}"
    )

    required = [
        path / "config.json",
        path / "tokenizer_config.json",
        path / "tokenizer.json",
    ]

    for item in required:
        if not item.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint file: {item}"
            )

    has_weights = (
        (path / "model.safetensors").is_file()
        or (path / "pytorch_model.bin").is_file()
    )

    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found under {path}"
        )

    return path


def load_tokenizer(
    checkpoint: Path,
    use_fast: bool = False,
):
    return AutoTokenizer.from_pretrained(
        str(checkpoint),
        use_fast=use_fast,
    )


def predict_checkpoint(
    checkpoint: Path,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_bf16_on_cuda: bool,
    tokenizer_use_fast: bool = False,
) -> np.ndarray:
    if not texts:
        raise ValueError(
            "predict_checkpoint received an empty text list"
        )

    tokenizer = load_tokenizer(
        checkpoint,
        use_fast=tokenizer_use_fast,
    )

    dataset = TokenizedTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=(
            8 if device.type == "cuda" else None
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(str(checkpoint))
    )

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = int(
            tokenizer.pad_token_id
        )

    model.config.problem_type = (
        "single_label_classification"
    )

    model = model.to(device)
    model.eval()

    use_bf16 = (
        device.type == "cuda"
        and use_bf16_on_cuda
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    logits_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(
                    device,
                    non_blocking=(device.type == "cuda"),
                )
                for key, value in batch.items()
            }

            if use_bf16:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    output = model(**batch)
            else:
                output = model(**batch)

            logits_parts.append(
                output.logits
                .float()
                .cpu()
                .numpy()
            )

    logits = np.concatenate(
        logits_parts,
        axis=0,
    )

    probabilities = probability_softmax(
        logits
    ).astype(np.float32)

    del loader
    del dataset
    del collator
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probabilities


def predict_branch(
    model_root: Path,
    branch: str,
    seeds: list[int],
    folds: list[int],
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_bf16_on_cuda: bool,
    tokenizer_use_fast: bool,
) -> np.ndarray:
    seed_probabilities: list[np.ndarray] = []

    for seed in seeds:
        fold_probabilities: list[np.ndarray] = []

        for fold in folds:
            checkpoint = checkpoint_path(
                model_root=model_root,
                branch=branch,
                seed=seed,
                fold=fold,
            )

            print(
                f"[risk] branch={branch} "
                f"seed={seed} fold={fold}"
            )

            probs = predict_checkpoint(
                checkpoint=checkpoint,
                texts=texts,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
                use_bf16_on_cuda=use_bf16_on_cuda,
                tokenizer_use_fast=tokenizer_use_fast,
            )

            fold_probabilities.append(probs)

        seed_mean = np.mean(
            np.stack(
                fold_probabilities,
                axis=0,
            ),
            axis=0,
        )

        seed_probabilities.append(seed_mean)

    branch_mean = np.mean(
        np.stack(
            seed_probabilities,
            axis=0,
        ),
        axis=0,
    )

    return branch_mean
