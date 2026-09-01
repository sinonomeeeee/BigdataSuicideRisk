#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risk_deberta_large.runner import (  # noqa: E402
    aggregate_run,
    create_run,
    execute_fold,
    run_status,
)


def default_config() -> Path:
    return (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "risk"
        / "risk_dv3large_cwp025_ml384_seed514_v1.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main-framework DeBERTa-v3-large four-class risk experiment"
    )
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create an immutable run directory")
    new_parser.add_argument("--kind", choices=["official", "smoke"], default="official")
    new_parser.add_argument("--run-id")

    train_parser = subparsers.add_parser("train", help="Train one fold")
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--run-id")
    train_parser.add_argument("--smoke-max-steps", type=int)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate five completed official folds"
    )
    aggregate_parser.add_argument("--run-id")

    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = args.config.resolve()
    project_root = args.project_root.resolve()
    if args.command == "new":
        result = {
            "run_id": create_run(config, project_root, args.kind, args.run_id)
        }
    elif args.command == "train":
        result = execute_fold(
            config,
            project_root,
            args.fold,
            run_id=args.run_id,
            smoke_max_steps=args.smoke_max_steps,
        )
    elif args.command == "aggregate":
        result = aggregate_run(config, project_root, run_id=args.run_id)
    elif args.command == "status":
        result = run_status(config, project_root, run_id=args.run_id)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
