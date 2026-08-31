# Final-S1 Reproducibility Guide

## Frozen System Identity

Final-S1 uses 20 fold-level DeBERTa-v3-large checkpoints.

## ML384

Training seeds:

- 514
- 9
- 42

Each training seed contains five fold models.

Within each seed:

    five fold probabilities
        ->
    equal-probability mean

Across seeds:

    seed514
    seed9
    seed42
        ->
    equal-probability AVG3

The short-context branch therefore contains 15 neural checkpoints.

## ML640

Training seed:

- 514

Aggregation:

    five fold probabilities
        ->
    equal-probability mean

The long-context branch therefore contains five neural checkpoints.

## Routing

The routing rule is:

    token_count_with_specials <= 384 -> ML384 AVG3
    token_count_with_specials > 384  -> ML640 seed514

Routing is deterministic and does not use a learned router.

## Calibration

Final raw probabilities are calibrated with scalar temperature
scaling.

    T = 1.8528777092173991

## Label Order

    0 = indicator
    1 = ideation
    2 = behavior
    3 = attempt

The label order must not be changed.

## Required Weight Layout

Place the 20 model files in one directory using the canonical names
defined in:

    metadata/weights_manifest.tsv

Example layout:

    weights/
        ml384_seed514_fold0.safetensors
        ml384_seed514_fold1.safetensors
        ml384_seed514_fold2.safetensors
        ml384_seed514_fold3.safetensors
        ml384_seed514_fold4.safetensors

        ml384_seed9_fold0.safetensors
        ...
        ml384_seed9_fold4.safetensors

        ml384_seed42_fold0.safetensors
        ...
        ml384_seed42_fold4.safetensors

        ml640_seed514_fold0.safetensors
        ...
        ml640_seed514_fold4.safetensors

## Verify Weight Identity

Before inference:

    python scripts/verify_weights.py \
      --weights-dir /path/to/weights

All 20 files should report PASS.

## Single-text Inference

    python inference.py \
      --text "Synthetic example text." \
      --weights-dir /path/to/weights \
      --device auto

## JSONL Inference

Example JSONL input:

    {"text": "Synthetic example A."}
    {"text": "Synthetic example B."}

Run:

    python inference.py \
      --input-jsonl examples/example.jsonl \
      --weights-dir /path/to/weights \
      --device auto

## Memory Behavior

The public implementation does not load all 20 DeBERTa models into
GPU memory simultaneously.

Each checkpoint is loaded, evaluated, accumulated into the relevant
ensemble probability vector, and released before the next checkpoint
is processed.

This reduces peak GPU memory requirements while preserving the frozen
ensemble calculation.

## Frozen Golden Reproduction Test

Before the first public release, the standalone implementation was
tested against the frozen internal Final-S1 deployment artifacts.

The test covered all 378 internal leaderboard rows.

The audit confirmed:

    token-length mismatch rows: 0
    routing mismatch rows: 0
    long-route rows: 14
    final prediction mismatch rows: 0
    temperature argmax changes: 0

All branch and final-system probability comparisons agreed within
floating-point tolerance.

The largest observed differences were on the order of 1e-8.

No competition data is included in this repository as part of that
verification.
