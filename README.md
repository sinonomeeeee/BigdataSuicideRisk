# IEEE BigData 2026 — Suicide Risk Analysis System

This repository contains the implementation of our submission, organized into three components:

- Risk
- Evidence
- Factors

## Risk

## Evidence

The **Evidence** component extracts suicide-related phrases from social-media posts.

Our Evidence extraction system is based on DeBERTa-v3-base, and follows the procedure:

- Candidate Generation
- Binary Classification
- Evidence Aggregation

The Evidence inference accepts a csv containing columns 'row_id' and 'post'.
(Default: test.csv)
Additionally, a csv containing previously predicted 'suicide risk' is required.
(Default: SatAndSeriousTokyoScript.csv)

### Execution
    python Evidence/run_evidence_pipeline_local.py \
        --test test.csv \
        --risk-file SatAndSeriousTokyoScript.csv \
        --output-dir Evidence/evidence_local_run
        
Final output is documentet as:
Evidence/evidence_local_run/test_inference/LB_aggregated_evidence.csv

See Evidence/Readme_Ev.md for detailed instructions.

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
