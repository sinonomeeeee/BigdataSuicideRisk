from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import project_path
from .io_utils import append_jsonl_once, utc_now


def _resolve_architecture(
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    architecture = metrics.get("architecture")
    if isinstance(architecture, str) and architecture.strip():
        return architecture.strip()

    configured = config.get("model", {}).get("architecture")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    # Backward-compatible fallback for historical DeBERTa runs whose
    # fold metrics predate dynamic architecture capture.
    if config.get("model", {}).get("name") == "microsoft/deberta-v3-large":
        return "DebertaV2ForSequenceClassification"

    return "unknown"


def register_completed_run(
    config: dict[str, Any],
    project_root: Path,
    run_id: str,
    run_dir: Path,
    metrics: dict[str, Any],
    artifacts: dict[str, str],
) -> dict[str, bool]:
    experiment_entry = {
        "run_id": run_id,
        "experiment_id": config["experiment"]["id"],
        "task": config["experiment"]["task"],
        "status": "completed",
        "registered_at_utc": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "model_name": config["model"]["name"],
        "model_revision": config["model"].get("revision", "main"),
        "resolved_model_commit": metrics.get(
            "resolved_model_commit"
        ),
        "tokenizer_revision": metrics.get(
            "tokenizer_revision",
            config["model"].get(
                "tokenizer_revision",
                config["model"].get("revision", "main"),
            ),
        ),
        "seed": int(config["training"]["seed"]),
        "split_seed": int(config["splits"]["seed"]),
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "artifacts": artifacts,
    }
    model_entry = {
        "run_id": run_id,
        "model_id": f"{config['experiment']['id']}::{run_id}",
        "registered_at_utc": utc_now(),
        "architecture": _resolve_architecture(config, metrics),
        "pretrained_model": config["model"]["name"],
        "model_revision": config["model"].get("revision", "main"),
        "resolved_model_commit": metrics.get(
            "resolved_model_commit"
        ),
        "tokenizer_name": metrics.get(
            "tokenizer_name",
            config["model"]["name"],
        ),
        "tokenizer_revision": metrics.get(
            "tokenizer_revision",
            config["model"].get(
                "tokenizer_revision",
                config["model"].get("revision", "main"),
            ),
        ),
        "resolved_tokenizer_commit": metrics.get(
            "resolved_tokenizer_commit"
        ),
        "tokenizer_class": metrics.get("tokenizer_class"),
        "task": config["experiment"]["task"],
        "labels": config["data"]["labels"],
        "seed": int(config["training"]["seed"]),
        "pooled_oof_metrics": {
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        },
        "run_dir": str(run_dir.resolve()),
    }

    experiment_registry = project_path(
        project_root, str(config["outputs"]["registry_jsonl"])
    )
    model_registry = project_path(
        project_root, str(config["outputs"]["model_registry_jsonl"])
    )
    return {
        "experiment_registry_appended": append_jsonl_once(
            experiment_registry, experiment_entry, "run_id"
        ),
        "model_registry_appended": append_jsonl_once(
            model_registry, model_entry, "run_id"
        ),
    }
