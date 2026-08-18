from pathlib import Path
import argparse
import ast
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

CONSERVATIVE_LABELS = {
    "meaning in life",
    "sense of responsibility",
    "exposure to others' suicide",
    "poor school performance",
    "sexual orientation related issues",
    "cognitive deficits",
    "substance use",
}


def load_mean_thresholds(model_dir):
    model_dir = Path(model_dir)

    folds = []

    for fold in range(5):
        path = model_dir / f"fold{fold}_summary.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing summary: {path}"
            )

        data = json.loads(path.read_text())

        if "thresholds" not in data:
            raise ValueError(
                f"No thresholds in {path}"
            )

        folds.append(data["thresholds"])

    return {
        label: float(
            np.mean([
                fold[label]
                for fold in folds
            ])
        )
        for label in LABELS
    }


def load_probs(path, prob_name):
    df = pd.read_csv(path)

    required = {
        "row_id",
        "factor",
        "prob",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    df = df[
        ["row_id", "factor", "prob"]
    ].copy()

    df["row_id"] = df["row_id"].astype(str)

    return df.rename(
        columns={
            "prob": prob_name
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Combine verifier probabilities."
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--taxonomy-probs",
        required=True,
    )

    parser.add_argument(
        "--evidence-probs",
        required=True,
    )

    parser.add_argument(
        "--taxonomy-model-dir",
        required=True,
    )

    parser.add_argument(
        "--evidence-model-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    inp = pd.read_csv(args.input)

    if "row_id" not in inp.columns:
        raise ValueError(
            "Input CSV must contain row_id."
        )

    inp["row_id"] = inp["row_id"].astype(str)

    if inp["row_id"].duplicated().any():
        raise ValueError(
            "row_id values must be unique."
        )

    taxonomy = load_probs(
        args.taxonomy_probs,
        "taxonomy_prob",
    )

    evidence = load_probs(
        args.evidence_probs,
        "evidence_prob",
    )

    merged = taxonomy.merge(
        evidence,
        on=["row_id", "factor"],
        how="inner",
        validate="one_to_one",
    )

    expected = len(inp) * len(LABELS)

    if len(merged) != expected:
        raise RuntimeError(
            "Unexpected number of post-factor rows: "
            f"{len(merged)}; expected {expected}"
        )

    taxonomy_thresholds = load_mean_thresholds(
        args.taxonomy_model_dir
    )

    evidence_thresholds = load_mean_thresholds(
        args.evidence_model_dir
    )

    thresholds = {}

    for label in LABELS:
        threshold = (
            0.60 * taxonomy_thresholds[label]
            + 0.40 * evidence_thresholds[label]
        )

        if label in CONSERVATIVE_LABELS:
            threshold = max(
                threshold,
                0.60,
            )

        thresholds[label] = max(
            threshold,
            0.50,
        )

    merged["final_probability"] = (
        0.60 * merged["taxonomy_prob"]
        + 0.40 * merged["evidence_prob"]
    )

    merged["threshold"] = (
        merged["factor"]
        .map(thresholds)
    )

    merged["positive"] = (
        merged["final_probability"]
        >= merged["threshold"]
    )

    rank = {
        label: i
        for i, label
        in enumerate(LABELS)
    }

    predictions = {}

    for row_id, group in merged.groupby(
        "row_id",
        sort=False,
    ):
        factors = group.loc[
            group["positive"],
            "factor",
        ].tolist()

        factors.sort(
            key=lambda x: rank[x]
        )

        predictions[row_id] = str(factors)

    out = inp[["row_id"]].copy()

    out["factors"] = (
        out["row_id"]
        .map(predictions)
        .fillna("[]")
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    counts = (
        out["factors"]
        .map(ast.literal_eval)
        .map(len)
    )

    print("rows:", len(out))
    print(
        "average predicted factors:",
        float(counts.mean()),
    )
    print(
        "min predicted factors:",
        int(counts.min()),
    )
    print(
        "max predicted factors:",
        int(counts.max()),
    )
    print("wrote:", output_path)


if __name__ == "__main__":
    main()
