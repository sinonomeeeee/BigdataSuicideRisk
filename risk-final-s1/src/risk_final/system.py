from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


PROBABILITY_DTYPE = np.float32


def probability_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.

    Fold-level probabilities are converted to float32 to reproduce
    the probability precision used by the original training pipeline.
    """
    values = np.asarray(logits, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(
            f"logits must be a 2D array; received shape={values.shape}"
        )

    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denominator = np.sum(exp_values, axis=1, keepdims=True)

    return exp_values / denominator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


@dataclass(frozen=True)
class WeightEntry:
    branch: str
    seed: int
    fold: int
    filename: str
    bytes: int
    sha256: str


class _TextDataset(torch.utils.data.Dataset):
    """
    Tokenization semantics matching the original TokenizedTextDataset:

        tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
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

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {
            key: value[index]
            for key, value in self.encodings.items()
        }


class FinalS1RiskClassifier:
    """
    Frozen public implementation of the Final-S1 Risk classifier.

    Final system
    ------------
    Short branch:
        ML384
        - seed 514, five-fold equal-probability mean
        - seed 9,   five-fold equal-probability mean
        - seed 42,  five-fold equal-probability mean
        - equal-probability AVG3 across the three seeds

    Long branch:
        ML640
        - seed 514
        - five-fold equal-probability mean

    Routing:
        token_count_with_specials > 384 -> ML640
        otherwise                       -> ML384 AVG3

    Calibration:
        scalar temperature scaling
    """

    def __init__(
        self,
        repository_root: str | Path,
        weights_dir: str | Path,
        device: str = "auto",
        verify_weights: bool = False,
    ) -> None:
        self.root = Path(repository_root).resolve()
        self.weights_dir = Path(weights_dir).resolve()

        self.config_path = self.root / "configs" / "final_risk.yaml"
        self.manifest_path = (
            self.root / "metadata" / "weights_manifest.tsv"
        )

        self.common_config_dir = (
            self.root / "model_assets" / "common"
        )
        self.short_tokenizer_dir = (
            self.root
            / "model_assets"
            / "tokenizer_short"
        )
        self.long_tokenizer_dir = (
            self.root
            / "model_assets"
            / "tokenizer_long"
        )

        self.config = self._load_config()
        self.labels = list(self.config["task"]["labels"])

        if self.labels != [
            "indicator",
            "ideation",
            "behavior",
            "attempt",
        ]:
            raise RuntimeError(
                "Unexpected label order in final configuration"
            )

        self.route_threshold = int(
            self.config["routing"]["threshold"]
        )
        self.temperature = float(
            self.config["calibration"]["temperature"]
        )

        if self.route_threshold != 384:
            raise RuntimeError(
                "Frozen Final-S1 routing threshold must be 384"
            )

        if not np.isclose(
            self.temperature,
            1.8528777092173991,
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                "Frozen Final-S1 deployment temperature mismatch"
            )

        self.device = self._resolve_device(device)

        self.entries = self._load_manifest()

        self._validate_manifest_structure()

        if verify_weights:
            self.verify_all_weights()

        # Training used use_fast=False for model input construction.
        self.short_tokenizer = AutoTokenizer.from_pretrained(
            self.short_tokenizer_dir,
            local_files_only=True,
            use_fast=False,
        )

        self.long_tokenizer = AutoTokenizer.from_pretrained(
            self.long_tokenizer_dir,
            local_files_only=True,
            use_fast=False,
        )

        # Phase6A routing explicitly used the fast tokenizer.
        self.route_tokenizer = AutoTokenizer.from_pretrained(
            self.short_tokenizer_dir,
            local_files_only=True,
            use_fast=True,
        )

    def _load_config(self) -> dict:
        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return yaml.safe_load(handle)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        value = str(device).strip().lower()

        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"

        resolved = torch.device(value)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable"
            )

        return resolved

    def _load_manifest(self) -> list[WeightEntry]:
        entries: list[WeightEntry] = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle,
                delimiter="\t",
            )

            for row in reader:
                entries.append(
                    WeightEntry(
                        branch=row["branch"],
                        seed=int(row["seed"]),
                        fold=int(row["fold"]),
                        filename=row["filename"],
                        bytes=int(row["bytes"]),
                        sha256=row["sha256"],
                    )
                )

        return entries

    def _validate_manifest_structure(self) -> None:
        expected = {
            ("ML384", 514, fold)
            for fold in range(5)
        }
        expected |= {
            ("ML384", 9, fold)
            for fold in range(5)
        }
        expected |= {
            ("ML384", 42, fold)
            for fold in range(5)
        }
        expected |= {
            ("ML640", 514, fold)
            for fold in range(5)
        }

        actual = {
            (entry.branch, entry.seed, entry.fold)
            for entry in self.entries
        }

        if actual != expected:
            raise RuntimeError(
                "Weight manifest does not match the frozen "
                "20-checkpoint Final-S1 structure"
            )

        if len(self.entries) != 20:
            raise RuntimeError(
                f"Expected 20 weights; found {len(self.entries)}"
            )

    def _entries(
        self,
        branch: str,
        seed: int,
    ) -> list[WeightEntry]:
        selected = [
            entry
            for entry in self.entries
            if entry.branch == branch
            and entry.seed == seed
        ]

        selected.sort(key=lambda entry: entry.fold)

        if [entry.fold for entry in selected] != list(range(5)):
            raise RuntimeError(
                f"Incomplete fold set for {branch} seed {seed}"
            )

        return selected

    def _weight_path(
        self,
        entry: WeightEntry,
    ) -> Path:
        return self.weights_dir / entry.filename

    def verify_all_weights(self) -> None:
        for entry in self.entries:
            path = self._weight_path(entry)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing weight: {path}"
                )

            actual_size = path.stat().st_size

            if actual_size != entry.bytes:
                raise RuntimeError(
                    f"Weight size mismatch for {entry.filename}: "
                    f"expected={entry.bytes}, actual={actual_size}"
                )

            actual_hash = sha256_file(path)

            if actual_hash != entry.sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {entry.filename}"
                )

    def _check_weight_identity(
        self,
        entry: WeightEntry,
    ) -> Path:
        path = self._weight_path(entry)

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing model weight: {path}"
            )

        if path.stat().st_size != entry.bytes:
            raise RuntimeError(
                f"Unexpected file size for {path.name}"
            )

        return path

    def token_count_with_specials(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        values = []

        for text in texts:
            ids = self.route_tokenizer(
                str(text),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]

            values.append(len(ids))

        return np.asarray(
            values,
            dtype=np.int64,
        )

    def _new_model(self) -> torch.nn.Module:
        config = AutoConfig.from_pretrained(
            self.common_config_dir,
            local_files_only=True,
        )

        model = AutoModelForSequenceClassification.from_config(
            config
        )

        return model

    def _predict_one_checkpoint(
        self,
        texts: list[str],
        tokenizer,
        max_length: int,
        batch_size: int,
        entry: WeightEntry,
    ) -> np.ndarray:
        weight_path = self._check_weight_identity(entry)

        model = self._new_model()

        state_dict = load_file(
            str(weight_path),
            device="cpu",
        )

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if missing or unexpected:
            raise RuntimeError(
                "State-dict mismatch for "
                f"{entry.filename}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        del state_dict

        model.eval()
        model.to(self.device)

        dataset = _TextDataset(
            texts,
            tokenizer,
            max_length,
        )

        collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=(
                8 if self.device.type == "cuda" else None
            ),
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        logits_parts: list[np.ndarray] = []

        # Original official runs used BF16 mixed precision on CUDA.
        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )

        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }

                with autocast_context:
                    output = model(**inputs)

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
        ).astype(
            PROBABILITY_DTYPE
        )

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"Probability normalization failed for {entry.filename}"
            )

        model.to("cpu")
        del model
        del dataset
        del loader
        del logits
        del logits_parts

        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return probabilities

    def _fold_mean(
        self,
        texts: list[str],
        branch: str,
        seed: int,
    ) -> np.ndarray:
        if branch == "ML384":
            tokenizer = self.short_tokenizer
            max_length = 384
            batch_size = 16
        elif branch == "ML640":
            tokenizer = self.long_tokenizer
            max_length = 640
            batch_size = 4
        else:
            raise ValueError(
                f"Unknown branch: {branch}"
            )

        accumulated: np.ndarray | None = None

        for entry in self._entries(branch, seed):
            probabilities = self._predict_one_checkpoint(
                texts=texts,
                tokenizer=tokenizer,
                max_length=max_length,
                batch_size=batch_size,
                entry=entry,
            )

            if accumulated is None:
                accumulated = np.zeros_like(
                    probabilities,
                    dtype=np.float32,
                )

            accumulated += probabilities

        if accumulated is None:
            raise RuntimeError(
                f"No models found for {branch} seed {seed}"
            )

        # Equal-probability mean over five folds.
        return (
            accumulated / np.float32(5.0)
        ).astype(np.float32)

    def _ml384_avg3(
        self,
        texts: list[str],
    ) -> np.ndarray:
        seed_outputs = [
            self._fold_mean(
                texts,
                branch="ML384",
                seed=seed,
            )
            for seed in (514, 9, 42)
        ]

        # Phase6A loaded run-level probability CSVs into float64
        # before computing the seed-level AVG3. Cast to float64 here
        # to preserve the same final composition semantics.
        stacked = np.stack(
            [
                output.astype(np.float64)
                for output in seed_outputs
            ],
            axis=0,
        )

        return np.mean(
            stacked,
            axis=0,
        )

    def _ml640_seed514(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return self._fold_mean(
            texts,
            branch="ML640",
            seed=514,
        ).astype(np.float64)

    def _apply_temperature(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        safe = np.clip(
            probabilities.astype(np.float64),
            1e-300,
            1.0,
        )

        powered = np.power(
            safe,
            1.0 / self.temperature,
        )

        return (
            powered
            / powered.sum(
                axis=1,
                keepdims=True,
            )
        )

    def predict(
        self,
        texts: Iterable[str],
    ) -> list[dict]:
        text_list = [
            "" if text is None else str(text)
            for text in texts
        ]

        if not text_list:
            return []

        lengths = self.token_count_with_specials(
            text_list
        )

        route_long = lengths > self.route_threshold

        # Exact Final-S1 composition:
        # the original Phase6A source runs contained predictions for
        # the complete input population before deterministic routing.
        ml384 = self._ml384_avg3(text_list)
        ml640 = self._ml640_seed514(text_list)

        final_raw = ml384.copy()
        final_raw[route_long] = ml640[route_long]

        final_calibrated = self._apply_temperature(
            final_raw
        )

        raw_pred = final_raw.argmax(axis=1)
        calibrated_pred = final_calibrated.argmax(axis=1)

        # Positive scalar temperature scaling must preserve argmax.
        if not np.array_equal(
            raw_pred,
            calibrated_pred,
        ):
            raise RuntimeError(
                "Temperature scaling unexpectedly changed argmax"
            )

        outputs = []

        for index, text in enumerate(text_list):
            probabilities = {
                label: float(
                    final_calibrated[index, label_id]
                )
                for label_id, label in enumerate(self.labels)
            }

            raw_probabilities = {
                label: float(
                    final_raw[index, label_id]
                )
                for label_id, label in enumerate(self.labels)
            }

            outputs.append(
                {
                    "text": text,
                    "token_count_with_specials": int(
                        lengths[index]
                    ),
                    "route": (
                        "ML640"
                        if bool(route_long[index])
                        else "ML384"
                    ),
                    "prediction_id": int(
                        calibrated_pred[index]
                    ),
                    "prediction": self.labels[
                        int(calibrated_pred[index])
                    ],
                    "probabilities": probabilities,
                    "raw_probabilities": raw_probabilities,
                    "temperature": self.temperature,
                }
            )

        return outputs
