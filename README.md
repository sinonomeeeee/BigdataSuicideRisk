# IEEE BigData 2026 — Suicide Risk Analysis System

This repository contains the implementation of our submission, organized into three components:

- Risk
- Evidence
- Factors

## Risk

## Evidence

## Factors

The **Factors** component identifies suicide-risk-related factors expressed in social-media posts.

Our final Factors system uses an ensemble of two DeBERTa-v3-base verifier models:

- Taxonomy Verifier
- Evidence-aware Taxonomy Verifier

Each verifier uses five-fold inference. Their post-level probabilities are combined using a weighted ensemble with factor-specific decision thresholds.

The standalone Factors package accepts a CSV containing `row_id` and `post`, and produces `row_id` and `factors`.

### Inference

    python Factors/factor_ensemble/predict.py \
      --input test.csv \
      --output predictions.csv

For detailed information about installation, model architecture, checkpoints, factor taxonomy, thresholds, input/output format, and reproduction instructions, see:

**[Factors — Detailed Documentation](Factors/factor_ensemble/README.md)**
