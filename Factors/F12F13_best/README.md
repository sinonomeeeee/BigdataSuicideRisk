# F12 + F13 Best Factor Model

Reproducible package for the best Factors submission used in the IEEE BIGDATA project.

## Final result

Historical submission:

    SatAndSeriousTokyoScript_F12F13_w60_40_safe_LB6627.csv

Reference SHA256:

    10dfc9af089a1a4b2fbe93ffa0aef348e831a2e2246c1220fbed92ad42b5a99a

The final submission has been reproduced byte-for-byte.

Verified result:

    rows: 378
    exact factor strings: 378 / 378
    factor-set equality: 378 / 378
    reference sha256: 10dfc9af089a1a4b2fbe93ffa0aef348e831a2e2246c1220fbed92ad42b5a99a
    reproduced sha256: 10dfc9af089a1a4b2fbe93ffa0aef348e831a2e2246c1220fbed92ad42b5a99a

## Final model

The final submission combines F12 and F13 probabilities:

    prob = 0.6 * prob_f12 + 0.4 * prob_f13

The per-factor threshold is:

    threshold = 0.6 * threshold_f12 + 0.4 * threshold_f13

For selected safe-floor labels:

    threshold = max(threshold, 0.60)

A global floor is then applied:

    threshold = max(threshold, 0.50)

The prediction rule is:

    prediction = prob >= threshold

Safe-floor factors:

- meaning in life
- sense of responsibility
- exposure to others' suicide
- poor school performance
- sexual orientation related issues
- cognitive deficits
- substance use

## Reproduction provenance

F12 folds 1-4 can be reproduced from their surviving checkpoints.

F13 folds 0-4 can be reproduced from their surviving checkpoints.

The original F12 fold0 checkpoint used for the historical leaderboard submission was overwritten during later experimentation and is no longer recoverable.

Its original leaderboard probability output was preserved as:

    reference/F12_fold0_historical_post_probs_long.csv

Exact historical reconstruction therefore uses:

    historical F12 fold0 probabilities
    + reproducible F12 folds 1-4
    + reproducible F13 folds 0-4
    + original 60/40 fusion
    + original threshold and safe-floor logic
    + canonical factor ordering

This reproduces the final LB6627 CSV byte-for-byte.

## Package contents

    F12F13_best/
    ├── README.md
    ├── reproduce.sh
    ├── src/
    │   ├── infer_f12.py
    │   ├── infer_f13.py
    │   └── reproduce_lb6627.py
    ├── configs/
    │   └── reproduction.json
    ├── checkpoints/
    │   └── README.md
    ├── reference/
    │   ├── F12_fold0_historical_post_probs_long.csv
    │   └── SatAndSeriousTokyoScript_F12F13_w60_40_safe_LB6627.csv
    ├── results/
    │   └── SatAndSeriousTokyoScript_F12F13_w60_40_safe_REPRODUCED.csv
    ├── requirements.txt
    ├── MANIFEST.sha256
    └── .gitignore

## Data requirements

Leaderboard inference requires the original project files:

    data/leaderboard.csv
    submissions/SatAndSeriousTokyoScript.csv

Training additionally requires:

    data/train.csv
    outputs/factor_deberta_5fold_balanced/fold_assignments.csv

Raw data should only be redistributed if permitted by the dataset or competition terms.

## Checkpoints

The surviving DeBERTa checkpoints are approximately 700+ MB each and are intentionally not included as ordinary Git blobs.

See:

    checkpoints/README.md

for the expected checkpoint layout and F12 fold0 provenance.

## Reproduce

From the project root, run:

    bash Factors/F12F13_best/reproduce.sh

The reproduction script verifies the generated submission against the historical reference SHA256.

## Reproducibility scope

Verified:

- F12 folds 1-4 inference
- F13 folds 0-4 inference
- F12 five-fold reconstruction using preserved historical fold0 probabilities
- F12/F13 60/40 probability fusion
- per-factor threshold reconstruction
- safe-floor logic
- canonical factor ordering
- final CSV serialization
- byte-for-byte SHA256 equality

Not independently recoverable:

- original historical F12 fold0 checkpoint

The historical leaderboard score encoded by the filename is LB6627.

Leaderboard ground truth is not available locally, so that leaderboard metric itself is not independently recomputed here.
