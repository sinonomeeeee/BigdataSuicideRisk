from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import project_path
from .io_utils import sha256_file


@dataclass(frozen=True)
class DataBundle:
    train: pd.DataFrame
    leaderboard: pd.DataFrame
    train_path: Path
    leaderboard_path: Path
    data_manifest_path: Path
    train_sha256: str
    leaderboard_sha256: str
    data_manifest: dict[str, Any]


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _manifest_output(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        output = manifest["outputs"][key]
    except KeyError as exc:
        raise ValueError(f"Data manifest missing outputs.{key}") from exc
    if not isinstance(output, dict):
        raise ValueError(f"Data manifest outputs.{key} must be a mapping")
    return output


def load_and_validate_data(config: dict[str, Any], project_root: Path) -> DataBundle:
    cfg = config["data"]
    train_path = project_path(project_root, str(cfg["train_path"]))
    leaderboard_path = project_path(project_root, str(cfg["leaderboard_path"]))
    manifest_path = project_path(project_root, str(cfg["data_manifest_path"]))

    for path in (train_path, leaderboard_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_sha = sha256_file(train_path)
    leaderboard_sha = sha256_file(leaderboard_path)

    expected_train_sha = str(_manifest_output(manifest, "train_csv").get("sha256", ""))
    expected_leaderboard_sha = str(
        _manifest_output(manifest, "leaderboard_csv").get("sha256", "")
    )
    if train_sha != expected_train_sha:
        raise ValueError(
            f"Training CSV SHA256 mismatch: actual={train_sha}, manifest={expected_train_sha}"
        )
    if leaderboard_sha != expected_leaderboard_sha:
        raise ValueError(
            "Leaderboard CSV SHA256 mismatch: "
            f"actual={leaderboard_sha}, manifest={expected_leaderboard_sha}"
        )

    train = _load_csv(train_path)
    leaderboard = _load_csv(leaderboard_path)

    row_col = str(cfg["row_id_column"])
    group_col = str(cfg["group_column"])
    text_col = str(cfg["text_column"])
    label_col = str(cfg["label_column"])
    labels = list(cfg["labels"])

    required_train = {row_col, group_col, text_col, label_col}
    required_leaderboard = {row_col, group_col, text_col}
    missing_train = required_train - set(train.columns)
    missing_leaderboard = required_leaderboard - set(leaderboard.columns)
    if missing_train:
        raise ValueError(f"Training CSV missing columns: {sorted(missing_train)}")
    if missing_leaderboard:
        raise ValueError(
            f"Leaderboard CSV missing columns: {sorted(missing_leaderboard)}"
        )

    if not train[row_col].is_unique:
        raise ValueError("Training row_id values are not unique")
    if not leaderboard[row_col].is_unique:
        raise ValueError("Leaderboard row_id values are not unique")
    if (train[group_col].str.strip() == "").any():
        raise ValueError("Training data contains empty group identifiers")
    if (train[text_col].str.strip() == "").any():
        raise ValueError("Training data contains empty post text")

    normalized = train[label_col].str.strip().str.lower()
    invalid = sorted(set(normalized.unique()) - set(labels))
    if invalid:
        raise ValueError(f"Unknown labels in training CSV: {invalid}")
    if not normalized.equals(train[label_col]):
        raise ValueError(
            "Training labels are not already normalized to lowercase; rerun the main "
            "data preparation step"
        )

    missing_labels = [label for label in labels if label not in set(normalized)]
    if missing_labels:
        raise ValueError(f"Training data is missing labels: {missing_labels}")

    label_to_id = {label: index for index, label in enumerate(labels)}
    train = train.copy()
    train["label_id"] = train[label_col].map(label_to_id).astype(int)

    expected_train_rows = int(_manifest_output(manifest, "train_csv").get("rows", -1))
    expected_leaderboard_rows = int(
        _manifest_output(manifest, "leaderboard_csv").get("rows", -1)
    )
    if len(train) != expected_train_rows:
        raise ValueError(
            f"Training row count mismatch: actual={len(train)}, manifest={expected_train_rows}"
        )
    if len(leaderboard) != expected_leaderboard_rows:
        raise ValueError(
            "Leaderboard row count mismatch: "
            f"actual={len(leaderboard)}, manifest={expected_leaderboard_rows}"
        )

    return DataBundle(
        train=train,
        leaderboard=leaderboard,
        train_path=train_path,
        leaderboard_path=leaderboard_path,
        data_manifest_path=manifest_path,
        train_sha256=train_sha,
        leaderboard_sha256=leaderboard_sha,
        data_manifest=manifest,
    )
