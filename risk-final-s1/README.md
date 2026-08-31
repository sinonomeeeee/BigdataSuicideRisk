# Final-S1 Suicide Risk Assessment

Research implementation of the Final-S1 Risk Assessment system
developed for the IEEE Big Data Cup 2026 Explainable Suicide Risk
Detection task.

## System

Final-S1 is a four-class classifier:

    0  Indicator
    1  Ideation
    2  Behavior
    3  Attempt

It uses a DeBERTa-v3-large backbone with two context-length branches.

The final inference structure is:

                             ML384 seed514
                              5-fold mean
                                   |
                             ML384 seed9
                              5-fold mean
                                   |
                             ML384 seed42
                              5-fold mean
                                   |
                             seed-level AVG3
                                   |
                    token_count_with_specials
                            threshold=384
                                   |
                     +-------------+-------------+
                     |                           |
                  <= 384                       > 384
                     |                           |
                 ML384 AVG3              ML640 seed514
                                            5-fold mean
                     |                           |
                     +-------------+-------------+
                                   |
                             Final-S1 raw
                                   |
                        temperature scaling
                        T = 1.8528777092173991
                                   |
                         calibrated output

Routing is deterministic. It is not a learned router.

## Repository Contents

    configs/
        final_risk.yaml

    docs/
        MODEL_CARD.md
        REPRODUCIBILITY.md

    examples/
        example.jsonl

    metadata/
        deployment_temperature.json
        weights_manifest.tsv

    model_assets/
        common/
        tokenizer_short/
        tokenizer_long/

    src/risk_final/
        __init__.py
        system.py

    scripts/
        verify_weights.py

    inference.py

## Installation

Create a Python environment and install the requirements:

    pip install -r requirements.txt

GPU-specific PyTorch installation may require using the wheel or
package index appropriate for the local CUDA environment.

## Model Weights

The final system requires 20 fold-level model weight files.

Weights are deliberately excluded from ordinary Git history.

The exact canonical filenames, byte sizes, and SHA256 values are
listed in:

    metadata/weights_manifest.tsv

Before inference, verify the files:

    python scripts/verify_weights.py \
      --weights-dir /path/to/weights

## Inference

Single text:

    python inference.py \
      --text "Synthetic example text." \
      --weights-dir /path/to/weights \
      --device auto

JSONL:

    python inference.py \
      --input-jsonl examples/example.jsonl \
      --weights-dir /path/to/weights \
      --device auto

Use --verify-weights if SHA256 verification should be performed
automatically before inference.

## Reproducibility

The standalone release implementation was verified against the frozen
Final-S1 deployment pipeline on all 378 internal leaderboard rows.

The golden reproduction audit produced:

    token-length mismatch rows = 0
    routing mismatch rows = 0
    final prediction mismatch rows = 0
    temperature argmax changes = 0

Branch-level and final-system probabilities agreed within numerical
floating-point tolerance.

See:

    docs/REPRODUCIBILITY.md

## Data

The competition dataset is not distributed in this repository.

Do not add competition post text, training data, leaderboard data,
test data, submission files, or derived files containing source text
to this repository.

## Intended Use and Safety

This software is intended for research, reproducibility, and
benchmarking.

It is not a clinical diagnostic system and must not be used as the
sole basis for diagnosis, treatment, emergency intervention, or
individual-level suicide-risk decisions.

See:

    docs/MODEL_CARD.md

for additional limitations.

## License Status

A project-level open-source license has not yet been assigned.

Model-weight redistribution and project licensing should be finalized
only after applicable project, team, competition, institutional, and
third-party requirements have been reviewed.

See:

    NOTICE.md
