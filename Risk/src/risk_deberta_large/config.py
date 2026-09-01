from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


_EXACT_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_exact_hf_commit(value: str) -> bool:
    return bool(_EXACT_HF_COMMIT_RE.fullmatch(value))


def load_config(path: Path, project_root: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    config = deepcopy(raw)
    config["_meta"] = {
        "source_path": str(path.resolve()),
        "project_root": str(project_root.resolve()),
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "experiment",
        "data",
        "splits",
        "model",
        "training",
        "runtime",
        "outputs",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {missing}")

    labels = config["data"].get("labels")
    expected = ["indicator", "ideation", "behavior", "attempt"]
    if labels != expected:
        raise ConfigError(f"data.labels must be exactly {expected}, got {labels}")

    model_cfg = config["model"]

    model_name = model_cfg.get("name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ConfigError("model.name must be a non-empty string")

    revision = model_cfg.get("revision", "main")
    if not isinstance(revision, str) or not revision.strip():
        raise ConfigError("model.revision must be a non-empty string")

    tokenizer_revision = model_cfg.get("tokenizer_revision", revision)
    if (
        not isinstance(tokenizer_revision, str)
        or not tokenizer_revision.strip()
    ):
        raise ConfigError(
            "model.tokenizer_revision must be a non-empty string"
        )

    historical_floating_revision_allowed = (
        model_name == "microsoft/deberta-v3-large"
    )
    if (
        not historical_floating_revision_allowed
        and not _is_exact_hf_commit(revision)
    ):
        raise ConfigError(
            "Non-DeBERTa model.revision must be an exact 40-character "
            "lowercase hexadecimal Hugging Face commit SHA"
        )
    if (
        not historical_floating_revision_allowed
        and not _is_exact_hf_commit(tokenizer_revision)
    ):
        raise ConfigError(
            "Non-DeBERTa model.tokenizer_revision must be an exact "
            "40-character lowercase hexadecimal Hugging Face commit SHA"
        )

    max_length = model_cfg.get("max_length")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length <= 0
    ):
        raise ConfigError("model.max_length must be a positive integer")

    num_labels = model_cfg.get("num_labels")
    if (
        isinstance(num_labels, bool)
        or not isinstance(num_labels, int)
        or num_labels != len(expected)
    ):
        raise ConfigError(
            f"model.num_labels must equal {len(expected)}, got {num_labels}"
        )

    use_safetensors = model_cfg.get("use_safetensors", False)
    if use_safetensors is not None and not isinstance(use_safetensors, bool):
        raise ConfigError("model.use_safetensors must be true, false, or null")

    quantization = model_cfg.get("quantization", "none")
    if quantization not in {"none", "qlora_4bit"}:
        raise ConfigError(
            "model.quantization must be either 'none' or 'qlora_4bit'"
        )

    qlora_cfg = model_cfg.get("qlora")
    if quantization == "qlora_4bit":
        if use_safetensors is not True:
            raise ConfigError(
                "QLoRA requires model.use_safetensors=true"
            )
        if not isinstance(qlora_cfg, dict):
            raise ConfigError(
                "QLoRA requires a model.qlora mapping"
            )

        required_qlora_fields = {
            "quant_type",
            "compute_dtype",
            "double_quant",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_bias",
            "target_modules",
        }
        missing_qlora_fields = sorted(
            required_qlora_fields - set(qlora_cfg)
        )
        if missing_qlora_fields:
            raise ConfigError(
                "Missing model.qlora fields: "
                f"{missing_qlora_fields}"
            )

        if qlora_cfg["quant_type"] != "nf4":
            raise ConfigError(
                "model.qlora.quant_type must be 'nf4'"
            )
        if qlora_cfg["compute_dtype"] != "bf16":
            raise ConfigError(
                "model.qlora.compute_dtype must be 'bf16'"
            )
        if not isinstance(qlora_cfg["double_quant"], bool):
            raise ConfigError(
                "model.qlora.double_quant must be boolean"
            )

        lora_r = qlora_cfg["lora_r"]
        if (
            isinstance(lora_r, bool)
            or not isinstance(lora_r, int)
            or lora_r <= 0
        ):
            raise ConfigError(
                "model.qlora.lora_r must be a positive integer"
            )

        lora_alpha = qlora_cfg["lora_alpha"]
        if (
            isinstance(lora_alpha, bool)
            or not isinstance(lora_alpha, int)
            or lora_alpha <= 0
        ):
            raise ConfigError(
                "model.qlora.lora_alpha must be a positive integer"
            )

        lora_dropout = qlora_cfg["lora_dropout"]
        if (
            isinstance(lora_dropout, bool)
            or not isinstance(lora_dropout, (int, float))
            or not 0.0 <= float(lora_dropout) < 1.0
        ):
            raise ConfigError(
                "model.qlora.lora_dropout must be in [0, 1)"
            )

        if qlora_cfg["lora_bias"] != "none":
            raise ConfigError(
                "model.qlora.lora_bias must be 'none'"
            )

        if qlora_cfg["target_modules"] != "all-linear":
            raise ConfigError(
                "model.qlora.target_modules must be 'all-linear'"
            )

        if config["training"].get("precision") != "bf16":
            raise ConfigError(
                "QLoRA v1 requires training.precision='bf16'"
            )
    elif qlora_cfg is not None:
        raise ConfigError(
            "model.qlora is only valid when "
            "model.quantization='qlora_4bit'"
        )

    training_seed = config.get("training", {}).get("seed")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ConfigError(
            "training.seed must be a non-negative integer"
        )
    if int(config["splits"].get("seed")) != 514:
        raise ConfigError("splits.seed must be 514")

    n_splits = int(config["splits"].get("n_splits", 0))
    if n_splits < 2:
        raise ConfigError("splits.n_splits must be >= 2")

    batch = int(config["training"].get("train_batch_size", 0))
    accumulation = int(config["training"].get("gradient_accumulation_steps", 0))
    effective = int(config["training"].get("effective_batch_size", 0))
    if batch <= 0 or accumulation <= 0:
        raise ConfigError("Batch size and gradient accumulation must be positive")
    if batch * accumulation != effective:
        raise ConfigError(
            "training.effective_batch_size must equal train_batch_size * "
            "gradient_accumulation_steps"
        )

    power = float(config["training"].get("class_weight_power", -1.0))
    if not 0.0 <= power <= 1.0:
        raise ConfigError("training.class_weight_power must be between 0 and 1")


def project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
