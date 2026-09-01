from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from risk.calibration import apply_temperature
from risk.inference import (
    checkpoint_path,
    load_tokenizer,
    predict_branch,
    resolve_device,
)
from risk.routing import (
    long_context_mask,
    token_count_with_specials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final IEEE BigData 2026 "
            "Risk Assessment system."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV containing the post column.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output CSV. Input columns are preserved and "
            "the suicide-risk column is written."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/risk_final.yaml",
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "models",
    )

    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic CSV containing final raw "
            "and calibrated probabilities."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, e.g. cuda, cuda:0, cpu.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.config.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    input_cfg = config["input"]
    model_cfg = config["model"]
    route_cfg = config["routing"]
    calibration_cfg = config["calibration"]
    runtime_cfg = config["runtime"]

    labels = list(config["labels"])
    text_col = str(input_cfg["text_column"])
    risk_col = str(
        input_cfg["output_risk_column"]
    )

    frame = pd.read_csv(args.input)

    if text_col not in frame.columns:
        raise KeyError(
            f"Input CSV is missing required column: {text_col}"
        )

    texts = (
        frame[text_col]
        .astype(str)
        .tolist()
    )

    if not texts:
        raise RuntimeError(
            "Input CSV contains no rows"
        )

    requested_device = (
        args.device
        if args.device is not None
        else str(runtime_cfg["device"])
    )

    device = resolve_device(
        requested_device
    )

    use_fast = bool(
        model_cfg["tokenizer_use_fast"]
    )

    ml384_cfg = model_cfg["ml384"]
    ml640_cfg = model_cfg["ml640"]

    route_checkpoint = checkpoint_path(
        model_root=args.model_root,
        branch="ml384",
        seed=int(ml384_cfg["seeds"][0]),
        fold=int(ml384_cfg["folds"][0]),
    )

    route_tokenizer = load_tokenizer(
        route_checkpoint,
        use_fast=use_fast,
    )

    lengths = token_count_with_specials(
        texts=texts,
        tokenizer=route_tokenizer,
    )

    threshold = int(
        route_cfg["threshold"]
    )

    long_mask = long_context_mask(
        lengths,
        threshold=threshold,
    )

    short_mask = ~long_mask

    short_indices = np.flatnonzero(
        short_mask
    )

    long_indices = np.flatnonzero(
        long_mask
    )

    print(
        f"[risk] rows={len(frame)} "
        f"short={len(short_indices)} "
        f"long={len(long_indices)} "
        f"threshold={threshold}"
    )

    # Reproduce the original deployment procedure:
    # each branch first predicts the complete input set using all
    # corresponding folds/checkpoints. Routing is applied afterwards.
    ml384_probs = predict_branch(
        model_root=args.model_root,
        branch="ml384",
        seeds=[
            int(value)
            for value in ml384_cfg["seeds"]
        ],
        folds=[
            int(value)
            for value in ml384_cfg["folds"]
        ],
        texts=texts,
        max_length=int(
            ml384_cfg["max_length"]
        ),
        batch_size=int(
            ml384_cfg["batch_size"]
        ),
        device=device,
        use_bf16_on_cuda=bool(
            runtime_cfg["use_bf16_on_cuda"]
        ),
        tokenizer_use_fast=use_fast,
    )

    ml640_probs = predict_branch(
        model_root=args.model_root,
        branch="ml640",
        seeds=[
            int(value)
            for value in ml640_cfg["seeds"]
        ],
        folds=[
            int(value)
            for value in ml640_cfg["folds"]
        ],
        texts=texts,
        max_length=int(
            ml640_cfg["max_length"]
        ),
        batch_size=int(
            ml640_cfg["batch_size"]
        ),
        device=device,
        use_bf16_on_cuda=bool(
            runtime_cfg["use_bf16_on_cuda"]
        ),
        tokenizer_use_fast=use_fast,
    )

    final_raw = ml384_probs.copy()
    final_raw[long_indices] = ml640_probs[long_indices]

    if not np.allclose(
        final_raw.sum(axis=1),
        1.0,
        atol=2e-5,
    ):
        raise RuntimeError(
            "Final raw probabilities do not sum to one"
        )

    if bool(calibration_cfg["enabled"]):
        final_calibrated = apply_temperature(
            final_raw,
            float(
                calibration_cfg["temperature"]
            ),
        )
    else:
        final_calibrated = (
            final_raw.astype(float)
        )

    predicted_ids = final_calibrated.argmax(
        axis=1
    )

    predicted_labels = [
        labels[int(index)]
        for index in predicted_ids
    ]

    output = frame.copy()
    output[risk_col] = predicted_labels

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if args.probabilities_output is not None:
        row_col = str(
            input_cfg["row_id_column"]
        )
        group_col = str(
            input_cfg["group_column"]
        )

        diagnostic = pd.DataFrame(
            index=frame.index
        )

        if row_col in frame.columns:
            diagnostic[row_col] = frame[row_col]

        if group_col in frame.columns:
            diagnostic[group_col] = frame[group_col]

        diagnostic[
            "token_count_with_specials"
        ] = lengths

        diagnostic["risk_branch"] = np.where(
            long_mask,
            "ml640_seed514",
            "ml384_avg3",
        )

        for index, label in enumerate(labels):
            diagnostic[
                f"raw_prob_{label}"
            ] = final_raw[:, index]

            diagnostic[
                f"calibrated_prob_{label}"
            ] = final_calibrated[:, index]

        diagnostic[
            "predicted_risk"
        ] = predicted_labels

        args.probabilities_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        diagnostic.to_csv(
            args.probabilities_output,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

    print(
        f"[risk] wrote {args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
