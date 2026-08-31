from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers.trainer_utils import get_last_checkpoint

from .io_utils import sha256_file, utc_now, write_json
from .metrics import classification_metrics, probability_softmax

LOGGER = logging.getLogger(__name__)


class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int,
        labels: list[int] | None = None,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[index])
        return item


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits.float(),
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def smoothed_class_weights(
    labels: np.ndarray,
    num_classes: int,
    power: float,
) -> torch.Tensor:
    classes = np.arange(num_classes)
    present = set(np.unique(labels).tolist())
    missing = sorted(set(classes.tolist()) - present)
    if missing:
        raise ValueError(f"Training fold is missing class IDs: {missing}")
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    smoothed = np.power(balanced, power)
    smoothed = smoothed / smoothed.mean()
    return torch.tensor(smoothed, dtype=torch.float32)


def _set_seed(seed: int, full_determinism: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if full_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def _training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    normalized = dict(kwargs)
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        normalized["evaluation_strategy"] = normalized.pop("eval_strategy")
    if "use_cpu" not in parameters and "no_cuda" in parameters:
        normalized["no_cuda"] = normalized.pop("use_cpu")
    filtered = {key: value for key, value in normalized.items() if key in parameters}
    return TrainingArguments(**filtered)


def _processing_kwargs(tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def _precision_flags(precision: str) -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    choice = precision.lower()
    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    if choice == "auto":
        return (False, True) if bf16_supported else (True, False)
    if choice == "bf16":
        if not bf16_supported:
            raise RuntimeError("BF16 requested but the visible GPU/PyTorch build does not support it")
        return False, True
    if choice == "fp16":
        return True, False
    if choice in {"fp32", "none"}:
        return False, False
    raise ValueError(f"Unknown precision setting: {precision}")


def _metric_callback(labels: list[str]):
    def compute(eval_prediction: Any) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        y_pred = np.asarray(logits).argmax(axis=1)
        y_true = np.asarray(eval_prediction.label_ids)
        metrics = classification_metrics(y_true, y_pred, labels)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "micro_f1": metrics["micro_f1"],
        }

    return compute


def _enable_gradient_checkpointing(model: Any, enabled: bool) -> tuple[bool, Any]:
    if not enabled:
        return False, None
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "gradient_checkpointing_kwargs" in parameters:
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return True, {"use_reentrant": False}
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return False, None
    except TypeError:
        LOGGER.warning(
            "This Transformers version cannot safely set non-reentrant gradient "
            "checkpointing; disabling gradient checkpointing."
        )
        return False, None


def _is_qlora_4bit(model_cfg: dict[str, Any]) -> bool:
    return str(model_cfg.get("quantization", "none")) == "qlora_4bit"


def _build_qlora_quantization_config(
    model_cfg: dict[str, Any],
) -> BitsAndBytesConfig:
    qlora_cfg = model_cfg["qlora"]
    compute_dtype = str(qlora_cfg["compute_dtype"]).lower()
    if compute_dtype != "bf16":
        raise ValueError(
            "Only bf16 QLoRA compute dtype is supported"
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(qlora_cfg["quant_type"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bool(
            qlora_cfg["double_quant"]
        ),
    )


def _build_qlora_lora_config(
    model_cfg: dict[str, Any],
) -> LoraConfig:
    qlora_cfg = model_cfg["qlora"]

    return LoraConfig(
        r=int(qlora_cfg["lora_r"]),
        lora_alpha=int(qlora_cfg["lora_alpha"]),
        lora_dropout=float(qlora_cfg["lora_dropout"]),
        bias=str(qlora_cfg["lora_bias"]),
        task_type=TaskType.SEQ_CLS,
        target_modules=str(qlora_cfg["target_modules"]),
    )


def _prepare_qlora_model(
    model: Any,
    model_cfg: dict[str, Any],
    gradient_checkpointing: bool,
) -> Any:
    prepare_kwargs: dict[str, Any] = {}

    parameters = inspect.signature(
        prepare_model_for_kbit_training
    ).parameters

    if "use_gradient_checkpointing" in parameters:
        prepare_kwargs["use_gradient_checkpointing"] = bool(
            gradient_checkpointing
        )

    if (
        gradient_checkpointing
        and "gradient_checkpointing_kwargs" in parameters
    ):
        prepare_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False
        }

    model = prepare_model_for_kbit_training(
        model,
        **prepare_kwargs,
    )

    if gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model = get_peft_model(
        model,
        _build_qlora_lora_config(model_cfg),
    )

    return model


def train_fold(
    config: dict[str, Any],
    train: pd.DataFrame,
    leaderboard: pd.DataFrame,
    run_dir: Path,
    fold: int,
    smoke_max_steps: int | None = None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    split_cfg = config["splits"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    labels = list(data_cfg["labels"])
    row_col = str(data_cfg["row_id_column"])
    group_col = str(data_cfg["group_column"])
    text_col = str(data_cfg["text_column"])
    label_col = str(data_cfg["label_column"])
    n_splits = int(split_cfg["n_splits"])
    if fold not in range(n_splits):
        raise ValueError(f"Fold must be between 0 and {n_splits - 1}")

    fold_dir = run_dir / "folds" / f"fold_{fold}"
    trainer_dir = fold_dir / "trainer"
    best_model_dir = fold_dir / "best_model"
    final_artifacts = [
        fold_dir / "metrics.json",
        fold_dir / "valid_predictions.csv",
        fold_dir / "leaderboard_predictions.csv",
    ]
    if all(path.exists() for path in final_artifacts):
        raise FileExistsError(
            f"Fold {fold} is already completed in {fold_dir}; completed artifacts are immutable."
        )
    fold_dir.mkdir(parents=True, exist_ok=True)
    for path in final_artifacts:
        if path.exists():
            incomplete_dir = fold_dir / "incomplete_artifacts"
            incomplete_dir.mkdir(parents=True, exist_ok=True)
            target = incomplete_dir / path.name
            if target.exists():
                target = incomplete_dir / f"{path.stem}_{utc_now().replace(':', '')}{path.suffix}"
            path.replace(target)

    train_fold_frame = train[train["fold"] != fold].reset_index(drop=True)
    valid_fold_frame = train[train["fold"] == fold].reset_index(drop=True)
    if set(train_fold_frame[group_col]) & set(valid_fold_frame[group_col]):
        raise RuntimeError("Group leakage detected between train and validation")

    base_seed = int(training_cfg["seed"])
    seed = base_seed + fold if bool(training_cfg.get("seed_per_fold", False)) else base_seed
    _set_seed(seed, bool(training_cfg.get("full_determinism", False)))

    model_name = str(model_cfg["name"])
    revision = str(model_cfg.get("revision", "main"))
    tokenizer_revision = str(
        model_cfg.get("tokenizer_revision", revision)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=tokenizer_revision,
        use_fast=bool(model_cfg.get("tokenizer_use_fast", False)),
    )
    tokenizer_class = tokenizer.__class__.__name__
    max_length = int(model_cfg["max_length"])
    train_dataset = TokenizedTextDataset(
        train_fold_frame[text_col].astype(str).tolist(),
        tokenizer,
        max_length,
        train_fold_frame["label_id"].astype(int).tolist(),
    )
    valid_dataset = TokenizedTextDataset(
        valid_fold_frame[text_col].astype(str).tolist(),
        tokenizer,
        max_length,
        valid_fold_frame["label_id"].astype(int).tolist(),
    )
    leaderboard_dataset = TokenizedTextDataset(
        leaderboard[text_col].astype(str).tolist(),
        tokenizer,
        max_length,
        None,
    )

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}

    qlora_enabled = _is_qlora_4bit(model_cfg)
    quantization_method = str(
        model_cfg.get("quantization", "none")
    )

    model_load_kwargs: dict[str, Any] = {
        "revision": revision,
        "num_labels": len(labels),
        "id2label": id2label,
        "label2id": label2id,
    }

    use_safetensors = model_cfg.get("use_safetensors", False)
    if use_safetensors is not None:
        model_load_kwargs["use_safetensors"] = bool(use_safetensors)

    if qlora_enabled:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA 4-bit training requires CUDA"
            )

        model_load_kwargs["quantization_config"] = (
            _build_qlora_quantization_config(model_cfg)
        )
        model_load_kwargs["dtype"] = torch.bfloat16
        model_load_kwargs["device_map"] = {
            "": torch.cuda.current_device()
        }
    else:
        model_load_kwargs["dtype"] = torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_load_kwargs,
    )

    base_model_architecture = model.__class__.__name__

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = int(tokenizer.pad_token_id)

    gradient_checkpointing_requested = bool(
        training_cfg.get("gradient_checkpointing", False)
    )

    if qlora_enabled:
        model = _prepare_qlora_model(
            model,
            model_cfg,
            gradient_checkpointing_requested,
        )
    else:
        model = model.float()

    model.config.problem_type = "single_label_classification"
    model_architecture = model.__class__.__name__

    resolved_model_commit = getattr(model.config, "_commit_hash", None)
    if resolved_model_commit is not None:
        resolved_model_commit = str(resolved_model_commit)

    if (
        len(revision) == 40
        and all(char in "0123456789abcdef" for char in revision)
        and resolved_model_commit != revision
    ):
        raise RuntimeError(
            "Resolved model commit does not match requested exact revision: "
            f"requested={revision}, resolved={resolved_model_commit}"
        )

    resolved_tokenizer_commit = (
        tokenizer_revision
        if len(tokenizer_revision) == 40
        and all(
            char in "0123456789abcdef"
            for char in tokenizer_revision
        )
        else None
    )

    parameter_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
    }

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for _, parameter in trainable_parameters
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if qlora_enabled:
        if not trainable_parameters:
            raise RuntimeError(
                "QLoRA model has no trainable parameters"
            )

        trainable_names = [
            name
            for name, _ in trainable_parameters
        ]

        if not any(
            "lora_" in name
            for name in trainable_names
        ):
            raise RuntimeError(
                "QLoRA model has no trainable LoRA parameters"
            )

        if not any(
            "score" in name
            for name in trainable_names
        ):
            raise RuntimeError(
                "QLoRA sequence-classification head is not trainable"
            )

        trainable_dtypes = {
            parameter.dtype
            for _, parameter in trainable_parameters
        }

        allowed_trainable_dtypes = {
            torch.float32,
            torch.bfloat16,
        }

        if not trainable_dtypes.issubset(
            allowed_trainable_dtypes
        ):
            raise RuntimeError(
                "Unexpected QLoRA trainable parameter dtypes: "
                f"{sorted(map(str, trainable_dtypes))}"
            )

        LOGGER.info(
            "QLoRA trainable parameters: %d / %d (%.6f%%)",
            trainable_parameter_count,
            total_parameter_count,
            (
                100.0
                * trainable_parameter_count
                / max(total_parameter_count, 1)
            ),
        )
        LOGGER.info(
            "QLoRA trainable dtypes: %s",
            sorted(map(str, trainable_dtypes)),
        )
    else:
        if parameter_dtypes != {torch.float32}:
            raise RuntimeError(
                "Model parameters must be FP32 before Trainer starts; "
                f"found dtypes: {sorted(map(str, parameter_dtypes))}"
            )

        LOGGER.info(
            "Model parameter dtype before Trainer: %s",
            sorted(map(str, parameter_dtypes)),
        )

    if qlora_enabled:
        # prepare_model_for_kbit_training already handles model-level
        # gradient-checkpointing preparation when requested.
        trainer_gc = False
        gc_kwargs = None
    else:
        trainer_gc, gc_kwargs = _enable_gradient_checkpointing(
            model,
            gradient_checkpointing_requested,
        )
    class_weights = smoothed_class_weights(
        train_fold_frame["label_id"].to_numpy(),
        num_classes=len(labels),
        power=float(training_cfg["class_weight_power"]),
    )

    train_batch_size = int(training_cfg["train_batch_size"])
    gradient_accumulation = int(training_cfg["gradient_accumulation_steps"])
    epochs = float(training_cfg["epochs"])
    updates_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / (train_batch_size * gradient_accumulation)),
    )
    total_updates = (
        int(smoke_max_steps)
        if smoke_max_steps is not None
        else max(1, math.ceil(updates_per_epoch * epochs))
    )
    warmup_steps = int(round(total_updates * float(training_cfg["warmup_ratio"])))
    fp16, bf16 = _precision_flags(str(training_cfg.get("precision", "auto")))

    tf32_enabled = False
    if torch.cuda.is_available() and bool(training_cfg.get("tf32", True)):
        major, _ = torch.cuda.get_device_capability()
        tf32_enabled = major >= 8

    arguments_payload: dict[str, Any] = {
        "output_dir": str(trainer_dir),
        "learning_rate": float(training_cfg["learning_rate"]),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": int(training_cfg["eval_batch_size"]),
        "gradient_accumulation_steps": gradient_accumulation,
        "num_train_epochs": epochs,
        "weight_decay": float(training_cfg["weight_decay"]),
        "warmup_steps": warmup_steps,
        "max_grad_norm": float(training_cfg["max_grad_norm"]),
        "eval_strategy": "epoch" if smoke_max_steps is None else "steps",
        "eval_steps": int(smoke_max_steps) if smoke_max_steps is not None else None,
        "save_strategy": "epoch" if smoke_max_steps is None else "steps",
        "save_steps": int(smoke_max_steps) if smoke_max_steps is not None else None,
        "logging_strategy": "steps",
        "logging_steps": min(int(training_cfg["logging_steps"]), total_updates),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(training_cfg["primary_metric"]),
        "greater_is_better": True,
        "save_total_limit": int(training_cfg["save_total_limit"]),
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "fp16": fp16,
        "bf16": bf16,
        "tf32": tf32_enabled,
        "use_cpu": False,
        "dataloader_num_workers": int(training_cfg["dataloader_num_workers"]),
        "dataloader_pin_memory": True,
        "group_by_length": bool(training_cfg.get("group_by_length", True)),
        "gradient_checkpointing": trainer_gc,
        "gradient_checkpointing_kwargs": gc_kwargs,
        "save_safetensors": True,
        "optim": str(training_cfg["optimizer"]),
        "full_determinism": bool(training_cfg.get("full_determinism", False)),
        "max_steps": int(smoke_max_steps) if smoke_max_steps is not None else -1,
        "remove_unused_columns": True,
    }
    arguments_payload = {
        key: value for key, value in arguments_payload.items() if value is not None
    }
    training_args = _training_arguments(arguments_payload)
    write_json(
        fold_dir / "training_arguments.json",
        arguments_payload,
    )

    callbacks = []
    early_stopping_patience = int(
        training_cfg.get("early_stopping_patience", 0)
    )
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=float(
                    training_cfg.get("early_stopping_threshold", 0.0)
                ),
            )
        )
    trainer = WeightedClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        ),
        compute_metrics=_metric_callback(labels),
        callbacks=callbacks,
        class_weights=class_weights,
        **_processing_kwargs(tokenizer),
    )

    started_at = utc_now()
    resume_checkpoint = None
    if trainer_dir.exists():
        resume_checkpoint = get_last_checkpoint(str(trainer_dir))
        if resume_checkpoint:
            LOGGER.info("Resuming fold %d from checkpoint: %s", fold, resume_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    if not np.isfinite(float(train_result.metrics.get("train_loss", np.nan))):
        raise RuntimeError("Non-finite train loss detected")
    if not np.isfinite(float(eval_metrics.get("eval_loss", np.nan))):
        raise RuntimeError("Non-finite evaluation loss detected")

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    valid_output = trainer.predict(valid_dataset)
    leaderboard_output = trainer.predict(leaderboard_dataset)
    valid_logits = (
        valid_output.predictions[0]
        if isinstance(valid_output.predictions, tuple)
        else valid_output.predictions
    )
    leaderboard_logits = (
        leaderboard_output.predictions[0]
        if isinstance(leaderboard_output.predictions, tuple)
        else leaderboard_output.predictions
    )
    if not np.isfinite(valid_logits).all() or not np.isfinite(leaderboard_logits).all():
        raise RuntimeError("Non-finite prediction logits detected")

    valid_probabilities = probability_softmax(np.asarray(valid_logits)).astype(np.float32)
    leaderboard_probabilities = probability_softmax(np.asarray(leaderboard_logits)).astype(
        np.float32
    )
    if not np.allclose(valid_probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Validation probabilities do not sum to 1")
    if not np.allclose(leaderboard_probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Leaderboard probabilities do not sum to 1")

    valid_predictions = valid_probabilities.argmax(axis=1)
    unique_predictions = np.unique(valid_predictions)
    if (
        smoke_max_steps is None
        and bool(training_cfg.get("fail_on_single_class_prediction", True))
        and len(unique_predictions) == 1
    ):
        raise RuntimeError(
            f"Fold {fold} collapsed to a single predicted class: {labels[int(unique_predictions[0])]}"
        )

    metrics = classification_metrics(
        valid_fold_frame["label_id"].to_numpy(), valid_predictions, labels
    )
    valid_frame = valid_fold_frame[[row_col, group_col, label_col, "label_id"]].copy()
    valid_frame.insert(1, "fold", fold)
    leaderboard_frame = leaderboard[[row_col, group_col]].copy()
    for index, label in enumerate(labels):
        valid_frame[f"prob_{label}"] = valid_probabilities[:, index]
        leaderboard_frame[f"prob_{label}"] = leaderboard_probabilities[:, index]
    valid_frame["predicted_label_id"] = valid_predictions
    valid_frame["predicted_risk"] = [labels[int(index)] for index in valid_predictions]
    leaderboard_predictions = leaderboard_probabilities.argmax(axis=1)
    leaderboard_frame["predicted_label_id"] = leaderboard_predictions
    leaderboard_frame["predicted_risk"] = [
        labels[int(index)] for index in leaderboard_predictions
    ]

    valid_path = fold_dir / "valid_predictions.csv"
    leaderboard_path = fold_dir / "leaderboard_predictions.csv"
    valid_frame.to_csv(valid_path, index=False, encoding="utf-8", lineterminator="\n")
    leaderboard_frame.to_csv(
        leaderboard_path, index=False, encoding="utf-8", lineterminator="\n"
    )

    report = {
        "status": "completed",
        "kind": "smoke" if smoke_max_steps is not None else "official",
        "fold": fold,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "model_name": model_name,
        "model_revision": revision,
        "resolved_model_commit": resolved_model_commit,
        "architecture": model_architecture,
        "base_model_architecture": base_model_architecture,
        "quantization": quantization_method,
        "qlora": (
            dict(model_cfg["qlora"])
            if qlora_enabled
            else None
        ),
        "trainable_parameters": int(trainable_parameter_count),
        "total_parameters": int(total_parameter_count),
        "trainable_parameter_percentage": float(
            100.0
            * trainable_parameter_count
            / max(total_parameter_count, 1)
        ),
        "tokenizer_name": model_name,
        "tokenizer_revision": tokenizer_revision,
        "resolved_tokenizer_commit": resolved_tokenizer_commit,
        "tokenizer_class": tokenizer_class,
        "use_safetensors": use_safetensors,
        "seed": seed,
        "split_seed": int(split_cfg["seed"]),
        "train_rows": int(len(train_fold_frame)),
        "valid_rows": int(len(valid_fold_frame)),
        "train_groups": int(train_fold_frame[group_col].nunique()),
        "valid_groups": int(valid_fold_frame[group_col].nunique()),
        "group_overlap": 0,
        "class_weights": class_weights.tolist(),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "classification": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "precision": {"fp16": fp16, "bf16": bf16, "tf32": tf32_enabled},
        "smoke_max_steps": smoke_max_steps,
        "artifacts": {
            "valid_predictions": str(valid_path.resolve()),
            "leaderboard_predictions": str(leaderboard_path.resolve()),
            "best_model": str(best_model_dir.resolve()),
            "valid_predictions_sha256": sha256_file(valid_path),
            "leaderboard_predictions_sha256": sha256_file(leaderboard_path),
        },
    }
    write_json(fold_dir / "metrics.json", report)
    write_json(fold_dir / "training_arguments.json", arguments_payload)

    del trainer, model, tokenizer, train_dataset, valid_dataset, leaderboard_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
