from pathlib import Path
import argparse
import subprocess
import sys
import tempfile

import pandas as pd


def run(cmd):
    print()
    print("+", " ".join(str(x) for x in cmd))

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict suicide-risk-related factors "
            "from input posts."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing row_id and post.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output prediction CSV.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
    )

    args = parser.parse_args()

    package_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        "row_id",
        "post",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["row_id"].duplicated().any():
        raise ValueError(
            "row_id values must be unique."
        )

    taxonomy_models = (
        package_dir
        / "models"
        / "taxonomy"
    )

    evidence_models = (
        package_dir
        / "models"
        / "evidence"
    )

    taxonomy_script = (
        package_dir
        / "src"
        / "taxonomy_verifier.py"
    )

    evidence_script = (
        package_dir
        / "src"
        / "evidence_verifier.py"
    )

    ensemble_script = (
        package_dir
        / "src"
        / "ensemble.py"
    )

    with tempfile.TemporaryDirectory(
        prefix="factor_ensemble_"
    ) as tmpdir:
        tmpdir = Path(tmpdir)

        taxonomy_probs = (
            tmpdir / "taxonomy_probs.csv"
        )

        evidence_probs = (
            tmpdir / "evidence_probs.csv"
        )

        run([
            sys.executable,
            taxonomy_script,
            "--input",
            input_path,
            "--model-dir",
            taxonomy_models,
            "--output",
            taxonomy_probs,
            "--batch-size",
            str(args.batch_size),
        ])

        run([
            sys.executable,
            evidence_script,
            "--input",
            input_path,
            "--model-dir",
            evidence_models,
            "--output",
            evidence_probs,
            "--batch-size",
            str(args.batch_size),
        ])

        run([
            sys.executable,
            ensemble_script,
            "--input",
            input_path,
            "--taxonomy-probs",
            taxonomy_probs,
            "--evidence-probs",
            evidence_probs,
            "--taxonomy-model-dir",
            taxonomy_models,
            "--evidence-model-dir",
            evidence_models,
            "--output",
            output_path,
        ])

    print()
    print("Prediction complete.")
    print("Output:", output_path)


if __name__ == "__main__":
    main()
