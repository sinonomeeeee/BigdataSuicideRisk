# Checkpoint requirements

The model checkpoints are intentionally not included as ordinary Git files because each checkpoint is approximately 700+ MB.

## F12 checkpoints

Expected layout:

    outputs/factor/F12_taxonomy_verifier/checkpoints/
      F12_taxonomy_verifier_fold1_best.pt
      F12_taxonomy_verifier_fold2_best.pt
      F12_taxonomy_verifier_fold3_best.pt
      F12_taxonomy_verifier_fold4_best.pt

## F13 checkpoints

Expected layout:

    outputs/factor/F13_evidence_taxonomy_verifier/checkpoints/
      F13_evidence_taxonomy_verifier_fold0_best.pt
      F13_evidence_taxonomy_verifier_fold1_best.pt
      F13_evidence_taxonomy_verifier_fold2_best.pt
      F13_evidence_taxonomy_verifier_fold3_best.pt
      F13_evidence_taxonomy_verifier_fold4_best.pt

## F12 fold0 provenance

The original F12 fold0 checkpoint used for the historical best leaderboard submission was overwritten during later experimentation.

Its original leaderboard probability artifact is preserved as:

    reference/F12_fold0_historical_post_probs_long.csv

This historical artifact is required for exact reconstruction of LB6627.

Later surviving F12 fold0 checkpoints were tested and did not reproduce the historical fold0 leaderboard probabilities, so they must not be substituted as if they were the original model.

## Large file storage

If checkpoints are later distributed with this project, use Git LFS or external model storage rather than ordinary Git blobs.
