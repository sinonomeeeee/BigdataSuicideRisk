from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import project_path
from .io_utils import sha256_file, utc_now, write_json


@dataclass(frozen=True)
class SplitBundle:
    assignments: pd.DataFrame
    csv_path: Path
    json_path: Path
    csv_sha256: str
    metadata: dict[str, Any]


def _validate_assignments(
    assignments: pd.DataFrame,
    train: pd.DataFrame,
    row_col: str,
    group_col: str,
    n_splits: int,
) -> None:
    required = {row_col, group_col, "fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    if not assignments[row_col].is_unique:
        raise ValueError("Split manifest row_id values are not unique")
    if set(assignments[row_col]) != set(train[row_col]):
        missing_rows = sorted(set(train[row_col]) - set(assignments[row_col]))[:10]
        extra_rows = sorted(set(assignments[row_col]) - set(train[row_col]))[:10]
        raise ValueError(
            f"Split row coverage mismatch; missing={missing_rows}, extra={extra_rows}"
        )
    folds = assignments["fold"].astype(int)
    if set(folds.unique()) != set(range(n_splits)):
        raise ValueError(
            f"Split folds must be 0..{n_splits - 1}, got {sorted(folds.unique())}"
        )
    group_fold_count = assignments.groupby(group_col)["fold"].nunique()
    leaking = group_fold_count[group_fold_count != 1]
    if not leaking.empty:
        raise ValueError(f"Groups assigned to multiple folds: {leaking.index.tolist()[:10]}")

    expected_group = train.set_index(row_col)[group_col]
    actual_group = assignments.set_index(row_col)[group_col]
    aligned = actual_group.loc[expected_group.index]
    if not aligned.equals(expected_group):
        raise ValueError("Split manifest group values do not match training data")


def create_or_load_splits(
    config: dict[str, Any],
    project_root: Path,
    train: pd.DataFrame,
    train_sha256: str,
) -> SplitBundle:
    data_cfg = config["data"]
    split_cfg = config["splits"]
    row_col = str(data_cfg["row_id_column"])
    group_col = str(data_cfg["group_column"])
    label_col = str(data_cfg["label_column"])
    n_splits = int(split_cfg["n_splits"])
    seed = int(split_cfg["seed"])
    csv_path = project_path(project_root, str(split_cfg["manifest_csv"]))
    json_path = project_path(project_root, str(split_cfg["manifest_json"]))

    if csv_path.exists() != json_path.exists():
        raise ValueError(
            "Split manifest is incomplete: CSV and JSON must either both exist or both be absent"
        )

    if csv_path.exists():
        assignments = pd.read_csv(csv_path, dtype={row_col: str, group_col: str})
        assignments["fold"] = assignments["fold"].astype(int)
        metadata = __import__("json").loads(json_path.read_text(encoding="utf-8"))
        if metadata.get("train_sha256") != train_sha256:
            raise ValueError("Existing split manifest was generated from a different train.csv")
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError("Existing split manifest seed does not match config")
        if int(metadata.get("n_splits", -1)) != n_splits:
            raise ValueError("Existing split manifest n_splits does not match config")
        _validate_assignments(assignments, train, row_col, group_col, n_splits)
        return SplitBundle(
            assignments=assignments,
            csv_path=csv_path,
            json_path=json_path,
            csv_sha256=sha256_file(csv_path),
            metadata=metadata,
        )

    if not bool(split_cfg.get("create_if_missing", False)):
        raise FileNotFoundError(f"Split manifest does not exist: {csv_path}")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=bool(split_cfg.get("shuffle", True)),
        random_state=seed,
    )
    fold_values = pd.Series(-1, index=train.index, dtype=int)
    for fold, (_, valid_index) in enumerate(
        splitter.split(train, y=train[label_col], groups=train[group_col])
    ):
        fold_values.iloc[valid_index] = fold
    if (fold_values < 0).any():
        raise RuntimeError("Failed to assign every training row to a fold")

    assignments = train[[row_col, group_col, label_col]].copy()
    assignments["fold"] = fold_values.to_numpy()
    _validate_assignments(assignments, train, row_col, group_col, n_splits)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    csv_sha = sha256_file(csv_path)

    fold_summary = []
    for fold in range(n_splits):
        subset = assignments[assignments["fold"] == fold]
        fold_summary.append(
            {
                "fold": fold,
                "rows": int(len(subset)),
                "groups": int(subset[group_col].nunique()),
                "label_counts": {
                    str(key): int(value)
                    for key, value in subset[label_col].value_counts().sort_index().items()
                },
            }
        )

    metadata = {
        "created_at_utc": utc_now(),
        "method": "StratifiedGroupKFold",
        "seed": seed,
        "n_splits": n_splits,
        "shuffle": bool(split_cfg.get("shuffle", True)),
        "train_sha256": train_sha256,
        "split_csv": str(csv_path.resolve()),
        "split_csv_sha256": csv_sha,
        "row_count": int(len(assignments)),
        "group_count": int(assignments[group_col].nunique()),
        "group_leakage_count": 0,
        "folds": fold_summary,
    }
    write_json(json_path, metadata)
    return SplitBundle(
        assignments=assignments,
        csv_path=csv_path,
        json_path=json_path,
        csv_sha256=csv_sha,
        metadata=metadata,
    )


def attach_folds(
    train: pd.DataFrame,
    assignments: pd.DataFrame,
    row_col: str,
) -> pd.DataFrame:
    fold_lookup = assignments.set_index(row_col)["fold"]
    result = train.copy()
    result["fold"] = result[row_col].map(fold_lookup)
    if result["fold"].isna().any():
        raise ValueError("Some training rows are missing fold assignments")
    result["fold"] = result["fold"].astype(int)
    return result
