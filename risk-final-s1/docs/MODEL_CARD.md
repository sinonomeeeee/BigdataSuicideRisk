# Final-S1 Risk Assessment Model Card

## Overview

Final-S1 is a four-class text classification system developed for the
Risk Assessment component of the IEEE Big Data Cup 2026 Explainable
Suicide Risk Detection task.

The fixed model label order is:

    0 = Indicator
    1 = Ideation
    2 = Behavior
    3 = Attempt

The model uses Microsoft DeBERTa-v3-large as its backbone.

## Final Architecture

Final-S1 contains two context-length branches.

### Short-context branch: ML384

The short-context branch contains three independently trained model
sets with training seeds 514, 9, and 42.

Each seed contains five cross-validation fold models.

For each seed, the five fold-level probability vectors are averaged
with equal weight.

The three resulting seed-level probability vectors are then averaged
with equal weight.

The short branch therefore uses 15 fold-level neural checkpoints.

### Long-context branch: ML640

The long-context branch uses training seed 514 and five fold-level
models.

Their probability vectors are averaged with equal weight.

The long branch therefore uses five neural checkpoints.

### Routing

Routing is deterministic rather than learned.

The input is tokenized without truncation and with special tokens.
Let L(x) denote this token count.

    L(x) <= 384  -> use ML384 AVG3
    L(x) > 384   -> use ML640 seed514

### Calibration

After branch selection, scalar temperature scaling is applied to the
raw Final-S1 probability vector.

The frozen deployment temperature is:

    T = 1.8528777092173991

Temperature scaling does not change the argmax class.

## Model Input Processing

Model inference uses the frozen DeBERTa tokenizer.

For model input construction:

- the slow DebertaV2Tokenizer is used;
- truncation is enabled;
- ML384 uses max_length=384;
- ML640 uses max_length=640;
- dynamic batch padding is used;
- CUDA evaluation pads to a multiple of 8;
- CUDA inference uses BF16 mixed precision.

Routing token length is computed separately without truncation and
includes special tokens.

## Model Files

Full Final-S1 inference requires 20 distinct fold-level
model.safetensors files.

The canonical filenames, file sizes, and SHA256 values are recorded in:

    metadata/weights_manifest.tsv

Model weights are intentionally excluded from ordinary Git history.

## Intended Use

The model is provided for research, reproducibility, and benchmarking.

It is not a clinical diagnostic instrument.

It must not be used as the sole basis for clinical decisions,
emergency intervention, diagnosis, treatment, or individual-level
suicide-risk decisions.

## Limitations

The system was developed and evaluated within a specific research
competition setting.

Performance outside the original data distribution has not been
established by this release.

Model probabilities should not be interpreted as clinical
probabilities of suicide or self-harm.

Errors may be associated with ambiguous language, annotation
uncertainty, context truncation, domain shift, and model instability.

## Data

The competition dataset is not distributed in this repository.

No competition post text, training data, leaderboard data, or
submission file is included in the public repository.

Users seeking access to the competition dataset must obtain it through
the appropriate official source and comply with the applicable data
terms.

## Reproducibility

See:

    docs/REPRODUCIBILITY.md

for the exact inference structure and verification procedure.
