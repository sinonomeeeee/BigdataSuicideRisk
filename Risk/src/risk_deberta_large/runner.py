from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import load_config, project_path
from .data import DataBundle, load_and_validate_data
from .io_utils import (
    atomic_write_text,
    compact_utc_timestamp,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from .metrics import classification_metrics, confusion_matrix_frame
from .provenance import capture_provenance
from .registry import register_completed_run
from .splits import SplitBundle, attach_folds, create_or_load_splits

LOGGER = logging.getLogger(__name__)


def configure_logging(run_dir: Path | None = None, fold: int | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if run_dir is not None:
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_name = "run.log" if fold is None else f"fold_{fold}.log"
        file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _active_pointer(config: dict[str, Any], project_root: Path) -> Path:
    return project_path(project_root, str(config["runtime"]["active_run_pointer"]))


def resolve_run_id(
    config: dict[str, Any], project_root: Path, run_id: str | None
) -> str:
    if run_id:
        return run_id
    pointer = _active_pointer(config, project_root)
    if not pointer.exists():
        raise FileNotFoundError(
            f"No active run pointer exists at {pointer}. Create a new run first."
        )
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Active run pointer is empty: {pointer}")
    return value


def _run_dir(project_root: Path, run_id: str) -> Path:
    return project_root / "runs" / run_id


def _prepare_inputs(
    config: dict[str, Any], project_root: Path
) -> tuple[DataBundle, SplitBundle, pd.DataFrame]:
    data = load_and_validate_data(config, project_root)
    splits = create_or_load_splits(config, project_root, data.train, data.train_sha256)
    train = attach_folds(
        data.train,
        splits.assignments,
        str(config["data"]["row_id_column"]),
    )
    return data, splits, train


def create_run(
    config_path: Path,
    project_root: Path,
    kind: str,
    explicit_run_id: str | None = None,
) -> str:
    config = load_config(config_path, project_root)
    configure_logging()
    data, splits, train = _prepare_inputs(config, project_root)

    if kind not in {"official", "smoke"}:
        raise ValueError("Run kind must be official or smoke")
    experiment_id = str(config["experiment"]["id"])
    suffix = "smoke" if kind == "smoke" else "run"
    run_id = explicit_run_id or f"{experiment_id}__{suffix}__{compact_utc_timestamp()}"
    run_dir = _run_dir(project_root, run_id)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    for relative in [
        "config",
        "provenance",
        "logs",
        "folds",
        "predictions",
        "metrics",
    ]:
        (run_dir / relative).mkdir(parents=True, exist_ok=False)

    shutil.copy2(config_path, run_dir / "config" / "source.yaml")
    resolved_config = dict(config)
    resolved_config.pop("_meta", None)
    (run_dir / "config" / "resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(data.data_manifest_path, run_dir / "provenance" / "data_manifest_source.json")
    shutil.copy2(splits.csv_path, run_dir / "provenance" / "split_manifest.csv")
    shutil.copy2(splits.json_path, run_dir / "provenance" / "split_manifest_source.json")

    manifest = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "kind": kind,
        "status": "initialized",
        "created_at_utc": utc_now(),
        "project_root": str(project_root.resolve()),
        "model_name": config["model"]["name"],
        "model_revision": config["model"].get("revision", "main"),
        "tokenizer_name": config["model"]["name"],
        "tokenizer_revision": config["model"].get(
            "tokenizer_revision",
            config["model"].get("revision", "main"),
        ),
        "use_safetensors": config["model"].get(
            "use_safetensors", False
        ),
        "seed": int(config["training"]["seed"]),
        "split_seed": int(config["splits"]["seed"]),
        "train_sha256": data.train_sha256,
        "leaderboard_sha256": data.leaderboard_sha256,
        "split_sha256": splits.csv_sha256,
        "row_counts": {
            "train": int(len(train)),
            "leaderboard": int(len(data.leaderboard)),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "status": "initialized",
            "kind": kind,
            "updated_at_utc": utc_now(),
            "completed_folds": [],
            "failed_folds": [],
        },
    )
    capture_provenance(
        run_dir,
        project_root,
        sys.argv,
        data.data_manifest,
        splits.metadata,
    )
    pointer = _active_pointer(config, project_root)
    atomic_write_text(pointer, run_id + "\n")
    configure_logging(run_dir)
    LOGGER.info("Created %s run: %s", kind, run_id)
    LOGGER.info("Run directory: %s", run_dir)
    return run_id


def _load_run(
    config_path: Path,
    project_root: Path,
    run_id: str | None,
) -> tuple[dict[str, Any], str, Path, DataBundle, SplitBundle, pd.DataFrame]:
    config = load_config(config_path, project_root)
    resolved_run_id = resolve_run_id(config, project_root, run_id)
    run_dir = _run_dir(project_root, resolved_run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("experiment_id") != config["experiment"]["id"]:
        raise ValueError("Run experiment_id does not match the selected config")
    data, splits, train = _prepare_inputs(config, project_root)
    if manifest.get("train_sha256") != data.train_sha256:
        raise ValueError("Run train SHA256 no longer matches current processed data")
    if manifest.get("split_sha256") != splits.csv_sha256:
        raise ValueError("Run split SHA256 no longer matches canonical split manifest")
    return config, resolved_run_id, run_dir, data, splits, train


def _update_status(run_dir: Path, **updates: Any) -> None:
    status_path = run_dir / "status.json"
    status = read_json(status_path)
    status.update(updates)
    status["updated_at_utc"] = utc_now()
    write_json(status_path, status)


def execute_fold(
    config_path: Path,
    project_root: Path,
    fold: int,
    run_id: str | None = None,
    smoke_max_steps: int | None = None,
) -> dict[str, Any]:
    config, resolved_run_id, run_dir, data, _, train = _load_run(
        config_path, project_root, run_id
    )
    configure_logging(run_dir, fold)
    status = read_json(run_dir / "status.json")
    if status.get("status") == "completed":
        raise RuntimeError("Completed runs are immutable")
    if status.get("kind") == "official" and smoke_max_steps is not None:
        raise ValueError("Smoke training must use a smoke run, not an official run")
    if status.get("kind") == "smoke" and smoke_max_steps is None:
        raise ValueError("A smoke run must be trained with --smoke-max-steps")

    _update_status(run_dir, status="running", current_fold=fold)
    try:
        if bool(config.get("sentence_windows", {}).get("enabled", False)):
            from .sentence_training import train_fold
        else:
            from .training import train_fold

        report = train_fold(
            config,
            train,
            data.leaderboard,
            run_dir,
            fold,
            smoke_max_steps=smoke_max_steps,
        )
        status = read_json(run_dir / "status.json")
        completed = sorted(set(status.get("completed_folds", [])) | {fold})
        _update_status(
            run_dir,
            status="smoke_completed" if smoke_max_steps is not None else "partial",
            current_fold=None,
            completed_folds=completed,
        )
        LOGGER.info("Fold %d completed for run %s", fold, resolved_run_id)
        return report
    except Exception as exc:
        status = read_json(run_dir / "status.json")
        failed = sorted(set(status.get("failed_folds", [])) | {fold})
        _update_status(
            run_dir,
            status="failed",
            current_fold=None,
            failed_folds=failed,
            last_error=repr(exc),
        )
        LOGGER.exception("Fold %d failed", fold)
        raise


def aggregate_run(
    config_path: Path,
    project_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    config, resolved_run_id, run_dir, data, _, train = _load_run(
        config_path, project_root, run_id
    )
    configure_logging(run_dir)
    status = read_json(run_dir / "status.json")
    if status.get("kind") != "official":
        raise ValueError("Smoke runs cannot be aggregated as official models")
    if status.get("status") == "completed":
        return read_json(run_dir / "metrics" / "pooled_metrics.json")

    labels = list(config["data"]["labels"])
    row_col = str(config["data"]["row_id_column"])
    label_col = str(config["data"]["label_column"])
    group_col = str(config["data"]["group_column"])
    n_splits = int(config["splits"]["n_splits"])
    probability_columns = [f"prob_{label}" for label in labels]

    missing = []
    fold_metrics_rows = []
    validation_frames = []
    leaderboard_probabilities = []
    fold_architectures: set[str] = set()
    fold_resolved_model_commits: set[str] = set()
    fold_tokenizer_revisions: set[str] = set()
    fold_resolved_tokenizer_commits: set[str] = set()
    fold_tokenizer_classes: set[str] = set()
    for fold in range(n_splits):
        fold_dir = run_dir / "folds" / f"fold_{fold}"
        valid_path = fold_dir / "valid_predictions.csv"
        leaderboard_path = fold_dir / "leaderboard_predictions.csv"
        metrics_path = fold_dir / "metrics.json"
        if not all(path.exists() for path in (valid_path, leaderboard_path, metrics_path)):
            missing.append(fold)
            continue
        valid = pd.read_csv(valid_path, dtype={row_col: str, group_col: str})
        leaderboard_frame = pd.read_csv(
            leaderboard_path, dtype={row_col: str, group_col: str}
        )
        fold_report = read_json(metrics_path)
        architecture = str(fold_report.get("architecture", "")).strip()
        if architecture:
            fold_architectures.add(architecture)

        resolved_model_commit = str(
            fold_report.get("resolved_model_commit") or ""
        ).strip()
        if resolved_model_commit:
            fold_resolved_model_commits.add(resolved_model_commit)

        tokenizer_revision = str(
            fold_report.get("tokenizer_revision") or ""
        ).strip()
        if tokenizer_revision:
            fold_tokenizer_revisions.add(tokenizer_revision)

        resolved_tokenizer_commit = str(
            fold_report.get("resolved_tokenizer_commit") or ""
        ).strip()
        if resolved_tokenizer_commit:
            fold_resolved_tokenizer_commits.add(
                resolved_tokenizer_commit
            )

        tokenizer_class = str(
            fold_report.get("tokenizer_class") or ""
        ).strip()
        if tokenizer_class:
            fold_tokenizer_classes.add(tokenizer_class)

        validation_frames.append(valid)
        leaderboard_lookup = leaderboard_frame.set_index(row_col)
        ordered = leaderboard_lookup.loc[data.leaderboard[row_col], probability_columns]
        leaderboard_probabilities.append(ordered.to_numpy(dtype=np.float32))
        classification = fold_report["classification"]
        fold_metrics_rows.append(
            {
                "fold": fold,
                "accuracy": classification["accuracy"],
                "macro_f1": classification["macro_f1"],
                "weighted_f1": classification["weighted_f1"],
                "micro_f1": classification["micro_f1"],
                "best_metric": fold_report.get("best_metric"),
            }
        )
    if missing:
        raise RuntimeError(f"Cannot aggregate; missing completed folds: {missing}")
    if len(fold_architectures) > 1:
        raise RuntimeError(
            "Completed folds disagree on model architecture: "
            f"{sorted(fold_architectures)}"
        )
    pooled_architecture = (
        next(iter(fold_architectures)) if fold_architectures else None
    )

    metadata_sets = {
        "resolved model commit": fold_resolved_model_commits,
        "tokenizer revision": fold_tokenizer_revisions,
        "resolved tokenizer commit": fold_resolved_tokenizer_commits,
        "tokenizer class": fold_tokenizer_classes,
    }
    for field_name, values in metadata_sets.items():
        if len(values) > 1:
            raise RuntimeError(
                f"Completed folds disagree on {field_name}: "
                f"{sorted(values)}"
            )

    pooled_resolved_model_commit = (
        next(iter(fold_resolved_model_commits))
        if fold_resolved_model_commits
        else None
    )
    pooled_tokenizer_revision = (
        next(iter(fold_tokenizer_revisions))
        if fold_tokenizer_revisions
        else config["model"].get(
            "tokenizer_revision",
            config["model"].get("revision", "main"),
        )
    )
    pooled_resolved_tokenizer_commit = (
        next(iter(fold_resolved_tokenizer_commits))
        if fold_resolved_tokenizer_commits
        else None
    )
    pooled_tokenizer_class = (
        next(iter(fold_tokenizer_classes))
        if fold_tokenizer_classes
        else None
    )

    oof = pd.concat(validation_frames, ignore_index=True)
    if not oof[row_col].is_unique:
        duplicates = oof[oof[row_col].duplicated(keep=False)][row_col].tolist()[:10]
        raise RuntimeError(f"OOF contains duplicate row IDs: {duplicates}")
    if set(oof[row_col]) != set(train[row_col]):
        raise RuntimeError("OOF row IDs do not exactly cover the training data")
    oof = oof.set_index(row_col).loc[train[row_col]].reset_index()
    y_true = train["label_id"].to_numpy(dtype=int)
    y_pred = oof["predicted_label_id"].to_numpy(dtype=int)
    pooled = classification_metrics(y_true, y_pred, labels)

    oof["gold"] = train[label_col].to_numpy()
    oof["gold_label_id"] = y_true
    oof_path = run_dir / "predictions" / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8", lineterminator="\n")

    mean_leaderboard = np.mean(np.stack(leaderboard_probabilities, axis=0), axis=0)
    leaderboard_output = data.leaderboard[[row_col, group_col]].copy()
    for index, label in enumerate(labels):
        leaderboard_output[f"prob_{label}"] = mean_leaderboard[:, index]
    leaderboard_pred = mean_leaderboard.argmax(axis=1)
    leaderboard_output["predicted_label_id"] = leaderboard_pred
    leaderboard_output["predicted_risk"] = [labels[int(index)] for index in leaderboard_pred]
    leaderboard_path = run_dir / "predictions" / "leaderboard_probabilities.csv"
    leaderboard_output.to_csv(
        leaderboard_path, index=False, encoding="utf-8", lineterminator="\n"
    )

    fold_metrics = pd.DataFrame(fold_metrics_rows).sort_values("fold")
    fold_metrics_path = run_dir / "metrics" / "fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)
    confusion_path = run_dir / "metrics" / "confusion_matrix.csv"
    confusion_matrix_frame(pooled["confusion_matrix"], labels).to_csv(confusion_path)

    pooled_payload = {
        **pooled,
        "run_id": resolved_run_id,
        "experiment_id": config["experiment"]["id"],
        "model_name": config["model"]["name"],
        "architecture": pooled_architecture,
        "model_revision": config["model"].get("revision", "main"),
        "resolved_model_commit": pooled_resolved_model_commit,
        "tokenizer_name": config["model"]["name"],
        "tokenizer_revision": pooled_tokenizer_revision,
        "resolved_tokenizer_commit": (
            pooled_resolved_tokenizer_commit
        ),
        "tokenizer_class": pooled_tokenizer_class,
        "seed": int(config["training"]["seed"]),
        "split_seed": int(config["splits"]["seed"]),
        "completed_at_utc": utc_now(),
        "fold_mean": {
            metric: float(fold_metrics[metric].mean())
            for metric in ["accuracy", "macro_f1", "weighted_f1", "micro_f1"]
        },
        "fold_std": {
            metric: float(fold_metrics[metric].std(ddof=0))
            for metric in ["accuracy", "macro_f1", "weighted_f1", "micro_f1"]
        },
        "artifacts": {
            "oof_predictions": str(oof_path.resolve()),
            "oof_predictions_sha256": sha256_file(oof_path),
            "leaderboard_probabilities": str(leaderboard_path.resolve()),
            "leaderboard_probabilities_sha256": sha256_file(leaderboard_path),
            "fold_metrics": str(fold_metrics_path.resolve()),
            "confusion_matrix": str(confusion_path.resolve()),
        },
    }
    pooled_path = run_dir / "metrics" / "pooled_metrics.json"
    write_json(pooled_path, pooled_payload)

    model_card = f"""# {config['experiment']['id']}

- Run ID: `{resolved_run_id}`
- Model: `{config['model']['name']}`
- Task: four-class suicide-risk classification
- Labels: `{', '.join(labels)}`
- Seed: `{config['training']['seed']}`
- Split: five-fold StratifiedGroupKFold by `{group_col}`, seed `{config['splits']['seed']}`
- Max length: `{config['model']['max_length']}`
- Class-weight power: `{config['training']['class_weight_power']}`
- Pooled OOF weighted F1: `{pooled['weighted_f1']:.6f}`
- Pooled OOF macro F1: `{pooled['macro_f1']:.6f}`
- Pooled OOF accuracy: `{pooled['accuracy']:.6f}`

This model card is generated from the immutable run artifacts. See `manifest.json`,
`provenance/`, `folds/`, `predictions/`, and `metrics/` in the run directory.
"""
    atomic_write_text(run_dir / "model_card.md", model_card)

    registry_result = register_completed_run(
        config,
        project_root,
        resolved_run_id,
        run_dir,
        pooled_payload,
        pooled_payload["artifacts"],
    )
    _update_status(
        run_dir,
        status="completed",
        completed_folds=list(range(n_splits)),
        current_fold=None,
        pooled_metrics=str(pooled_path.resolve()),
        registry=registry_result,
    )
    manifest = read_json(run_dir / "manifest.json")
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "pooled_metrics": {
                "accuracy": pooled["accuracy"],
                "macro_f1": pooled["macro_f1"],
                "weighted_f1": pooled["weighted_f1"],
            },
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    LOGGER.info(
        "Aggregation completed: weighted_f1=%.6f macro_f1=%.6f",
        pooled["weighted_f1"],
        pooled["macro_f1"],
    )
    return pooled_payload


def run_status(
    config_path: Path, project_root: Path, run_id: str | None = None
) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    resolved_run_id = resolve_run_id(config, project_root, run_id)
    run_dir = _run_dir(project_root, resolved_run_id)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    status = read_json(run_dir / "status.json")
    status["run_dir"] = str(run_dir.resolve())
    status["fold_directories"] = sorted(
        path.name for path in (run_dir / "folds").glob("fold_*") if path.is_dir()
    )
    return status
