from pathlib import Path
import argparse
import ast
import hashlib
import json

import numpy as np
import pandas as pd


LABELS = [
    "hopelessness",
    "low self-esteem",
    "emotion dysregulation",
    "poor social support",
    "coping strategy",
    "psychological capital",
    "suicide means (with access)",
    "interpersonal difficulty",
    "mental health issues",
    "prior self-harm or suicidal thought/attempt",
    "stressful life event",
    "dysfunctional family",
    "social support",
    "interpersonal violence",
    "physical health/characteristic",
    "sense of responsibility",
    "low socio-economic status",
    "traumatic experience",
    "meaning in life",
    "cognitive deficits",
    "substance use",
    "exposure to others' suicide",
    "poor school performance",
    "sexual orientation related issues",
]

SAFE_LABELS = {
    "meaning in life",
    "sense of responsibility",
    "exposure to others' suicide",
    "poor school performance",
    "sexual orientation related issues",
    "cognitive deficits",
    "substance use",
}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def mean_thresholds(directory, prefix):
    directory = Path(directory)
    per_fold = []

    for fold in range(5):
        p = directory / f"{prefix}_fold{fold}_summary.json"

        if not p.exists():
            raise FileNotFoundError(p)

        data = json.loads(p.read_text())

        if "thresholds" not in data:
            raise RuntimeError(f"No thresholds in {p}")

        per_fold.append(data["thresholds"])

    result = {}

    for label in LABELS:
        result[label] = float(
            np.mean([
                fold[label]
                for fold in per_fold
            ])
        )

    return result


def load_fold(path, fold):
    x = pd.read_csv(path)
    x["row_id"] = x["row_id"].astype(str)

    x = x[
        ["row_id", "factor", "prob_unit"]
    ].copy()

    return x.rename(
        columns={
            "prob_unit": f"prob_fold{fold}"
        }
    )


def rebuild_f12(
    historical_fold0,
    live_f12_leaderboard,
):
    frames = [
        load_fold(historical_fold0, 0)
    ]

    live_f12_leaderboard = Path(
        live_f12_leaderboard
    )

    for fold in range(1, 5):
        frames.append(
            load_fold(
                live_f12_leaderboard
                / f"F12_leaderboard_fold{fold}_post_probs_long.csv",
                fold,
            )
        )

    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["row_id", "factor"],
            how="inner",
            validate="one_to_one",
        )

    prob_cols = [
        f"prob_fold{fold}"
        for fold in range(5)
    ]

    merged["prob_f12"] = (
        merged[prob_cols]
        .mean(axis=1)
    )

    return merged[
        ["row_id", "factor", "prob_f12"]
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--historical-fold0",
        required=True,
    )

    parser.add_argument(
        "--f12-leaderboard-dir",
        required=True,
    )

    parser.add_argument(
        "--f12-summary-dir",
        required=True,
    )

    parser.add_argument(
        "--f13-probs",
        required=True,
    )

    parser.add_argument(
        "--f13-summary-dir",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--reference",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    f12 = rebuild_f12(
        args.historical_fold0,
        args.f12_leaderboard_dir,
    )

    f13 = pd.read_csv(
        args.f13_probs
    )

    f13["row_id"] = (
        f13["row_id"].astype(str)
    )

    f13 = f13.rename(
        columns={"prob": "prob_f13"}
    )

    merged = f12.merge(
        f13[
            [
                "row_id",
                "factor",
                "prob_f13",
            ]
        ],
        on=["row_id", "factor"],
        validate="one_to_one",
    )

    thr12 = mean_thresholds(
        args.f12_summary_dir,
        "F12_taxonomy_verifier",
    )

    thr13 = mean_thresholds(
        args.f13_summary_dir,
        "F13_evidence_taxonomy_verifier",
    )

    thresholds = {}

    for label in LABELS:
        threshold = (
            0.6 * thr12[label]
            + 0.4 * thr13[label]
        )

        if label in SAFE_LABELS:
            threshold = max(
                threshold,
                0.60,
            )

        threshold = max(
            threshold,
            0.50,
        )

        thresholds[label] = float(
            threshold
        )

    merged["blend"] = (
        0.6 * merged["prob_f12"]
        + 0.4 * merged["prob_f13"]
    )

    merged["threshold"] = (
        merged["factor"]
        .map(thresholds)
    )

    merged["positive"] = (
        merged["blend"]
        >= merged["threshold"]
    )

    rank = {
        label: index
        for index, label
        in enumerate(LABELS)
    }

    prediction_map = {}

    for row_id, group in merged.groupby(
        "row_id",
        sort=False,
    ):
        factors = (
            group.loc[
                group["positive"],
                "factor",
            ]
            .tolist()
        )

        factors = sorted(
            factors,
            key=lambda x: rank[x],
        )

        prediction_map[row_id] = str(
            factors
        )

    template = pd.read_csv(
        args.template
    )

    template["row_id"] = (
        template["row_id"].astype(str)
    )

    template["factors"] = (
        template["row_id"]
        .map(prediction_map)
        .fillna("[]")
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template.to_csv(
        output,
        index=False,
    )

    reference = pd.read_csv(
        args.reference
    )

    reference["row_id"] = (
        reference["row_id"].astype(str)
    )

    a = reference.sort_values(
        "row_id"
    ).reset_index(drop=True)

    b = template.sort_values(
        "row_id"
    ).reset_index(drop=True)

    assert a["row_id"].equals(
        b["row_id"]
    )

    exact = (
        a["factors"].astype(str)
        == b["factors"].astype(str)
    )

    set_equal = np.array([
        set(ast.literal_eval(x))
        == set(ast.literal_eval(y))
        for x, y in zip(
            a["factors"],
            b["factors"],
        )
    ])

    print("rows:", len(a))
    print(
        "exact factor strings:",
        int(exact.sum()),
        "/",
        len(a),
    )
    print(
        "factor-set equality:",
        int(set_equal.sum()),
        "/",
        len(a),
    )

    target_hash = sha256(
        args.reference
    )

    output_hash = sha256(
        output
    )

    print(
        "reference sha256:",
        target_hash,
    )

    print(
        "output sha256:",
        output_hash,
    )

    if target_hash != output_hash:
        raise SystemExit(
            "ERROR: reproduced file hash "
            "does not match reference"
        )

    print(
        "SUCCESS: LB6627 reproduced "
        "byte-for-byte."
    )


if __name__ == "__main__":
    main()
