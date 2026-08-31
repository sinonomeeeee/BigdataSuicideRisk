# Risk Assessment Source Code

IEEE BigData 2026
Explainable Suicide Risk Detection Challenge

This directory contains the source code for the Risk Assessment component
only. Evidence Grounding and Factor Identification are maintained by the
corresponding team members and are not included here.

## 1. Final Risk System

The final Risk Assessment system predicts one of four classes:

- indicator
- ideation
- behavior
- attempt

It uses the following inference pipeline:

1. Compute the full, untruncated token count including special tokens.
2. For posts with token_count_with_specials <= 384:
   - run three ML384 models trained with seeds 514, 9, and 42;
   - each seed contains five cross-validation fold checkpoints;
   - average probabilities across the five folds;
   - then equally average the three seed-level probability vectors.
3. For posts with token_count_with_specials > 384:
   - use the five-fold seed514 ML640 model;
   - average the five fold probability vectors.
4. Apply scalar temperature scaling with:
   T = 1.8528777092173991.
5. Use argmax to obtain the final Risk label.

Both ML384 and ML640 use microsoft/deberta-v3-large.

## 2. Environment

The code was developed with Python 3.10.

A CUDA-capable GPU is recommended.

Install inference dependencies with:

    python -m pip install -r requirements.txt

Additional training dependencies can be installed with:

    python -m pip install -r requirements-training.txt

## 3. Model Weights

The model weights are not included directly in this source-code package
because the complete Risk ensemble contains 20 large checkpoints.

See:

    models/README.md

for the download location and required directory structure.

## 4. Input Format

The inference script expects a CSV file containing at least:

    post

The following columns are supported and preserved when present:

    row_id
    anon_user_id
    suicide risk
    evidence
    factors

The script preserves all input columns and writes or replaces only:

    suicide risk

## 5. Run Final Inference

From this Risk directory:

    python run_inference.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --model-root models

To additionally save diagnostic Risk probabilities:

    python run_inference.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv \
        --model-root models \
        --probabilities-output /path/to/risk_probabilities.csv

By default, CUDA is used when available. CPU inference is also supported.

A specific device can be selected with:

    --device cuda:0

or:

    --device cpu

## 6. Configuration

The final public inference configuration is:

    configs/risk_final.yaml

The frozen deployment temperature is also retained separately in:

    configs/deployment_temperature.json

## 7. Training Source

The original DeBERTa training framework used for the final Risk models is
included under:

    src/risk_deberta_large/

The original training entrypoint is:

    scripts/risk/run_deberta_classifier.py

The four final training configurations are under:

    configs/training/

They correspond to:

- ML384 seed514
- ML384 seed9
- ML384 seed42
- ML640 seed514

Competition training and test data are not included in this package.

## 8. Tokenization

Inference reproduces the final training preprocessing:

    tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False
    )

Dynamic batch padding is performed with DataCollatorWithPadding.

The routing length is computed separately from the full, untruncated
tokenization including special tokens.

## 9. Notes

The deterministic routing rule is not a learned router.

The final Risk prediction path does not use the auxiliary Phase 3 diagnostic
models or rejected specialist models.

Large model checkpoints, experiment logs, intermediate runs, cached files,
OOF analysis artifacts, and competition data are intentionally excluded from
this source-code package.
