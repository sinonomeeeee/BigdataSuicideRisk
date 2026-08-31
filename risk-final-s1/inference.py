#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk_final import FinalS1RiskClassifier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Final-S1 four-class "
            "suicide-risk research classifier."
        )
    )

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--text",
        help="One input post.",
    )

    source.add_argument(
        "--input-jsonl",
        type=Path,
        help=(
            "UTF-8 JSONL file containing one object per line "
            'with a "text" field.'
        ),
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the 20 .safetensors files "
            "listed in metadata/weights_manifest.tsv."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help='Device: "auto", "cuda", "cuda:0", or "cpu".',
    )

    parser.add_argument(
        "--verify-weights",
        action="store_true",
        help=(
            "Verify SHA256 for all 20 weights before inference. "
            "This reads approximately 35 GB of model files."
        ),
    )

    return parser


def read_texts(args: argparse.Namespace) -> list[str]:
    if args.text is not None:
        return [args.text]

    texts = []

    with args.input_jsonl.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            if "text" not in obj:
                raise ValueError(
                    f"Line {line_number} has no 'text' field"
                )

            texts.append(str(obj["text"]))

    return texts


def main() -> int:
    args = build_parser().parse_args()

    classifier = FinalS1RiskClassifier(
        repository_root=ROOT,
        weights_dir=args.weights_dir,
        device=args.device,
        verify_weights=args.verify_weights,
    )

    outputs = classifier.predict(
        read_texts(args)
    )

    for output in outputs:
        print(
            json.dumps(
                output,
                ensure_ascii=False,
                sort_keys=False,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
