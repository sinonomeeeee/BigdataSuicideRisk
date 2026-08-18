# Suicide Risk Factor Classification

This directory contains the standalone factor-classification component of our system.

The model predicts suicide-risk-related factors from social-media posts using an ensemble of two DeBERTa-v3-base classifiers:

- Taxonomy Verifier
- Evidence-aware Taxonomy Verifier

Each verifier uses five independently trained folds. Their probabilities are averaged within each verifier and then combined to generate the final factor predictions.

## Quick start

From the repository root:

    python Factors/factor_ensemble/predict.py \
      --input /path/to/test.csv \
      --output predictions.csv

The input CSV must contain:

    row_id
    post

The output CSV contains:

    row_id
    factors

A CUDA-capable GPU is strongly recommended.

## Installation

Create or activate a Python environment and install the required packages:

    pip install -r Factors/factor_ensemble/requirements.txt

The main dependencies are:

- PyTorch
- Transformers
- pandas
- NumPy
- scikit-learn
- sentencepiece
- protobuf

The inference code was tested with Python 3.10.

The model uses the Hugging Face base encoder:

    microsoft/deberta-v3-base

Internet access may be required the first time the tokenizer and base encoder configuration are downloaded from Hugging Face.

## Input format

The model expects one post per row.

Required columns:

    row_id
    post

Example:

    row_id,post
    TEST001,"I feel hopeless about my future."
    TEST002,"My friends and family have been supportive recently."

Requirements:

- `row_id` must be unique.
- `row_id` must not be missing.
- `post` must not be missing.
- Additional columns are allowed and are ignored during inference.

The competition data may also contain metadata columns such as:

    anon_user_id
    post_id
    post_word_len
    post_char_len

These columns are not required by the model.

## Running inference

Run:

    python Factors/factor_ensemble/predict.py \
      --input test.csv \
      --output predictions.csv

Optional batch-size control:

    python Factors/factor_ensemble/predict.py \
      --input test.csv \
      --output predictions.csv \
      --batch-size 96

The prediction pipeline performs:

1. input validation;
2. text-unit construction;
3. five-fold inference with the Taxonomy Verifier;
4. five-fold inference with the Evidence-aware Taxonomy Verifier;
5. probability averaging across folds;
6. probability-level ensembling;
7. factor-specific thresholding;
8. deterministic factor ordering;
9. final CSV generation.

For an input file containing N rows, the output contains exactly N rows.

## Output format

The output contains two columns:

    row_id
    factors

Example:

    row_id,factors
    TEST001,"['hopelessness', 'mental health issues']"
    TEST002,"['social support']"

The `factors` field is a Python-style list of predicted factor labels.

The order of predicted factors follows the fixed taxonomy order.

## Model architecture

Both verifier models use:

    microsoft/deberta-v3-base

For each post, the system evaluates a taxonomy of 24 suicide-risk-related factors.

Longer posts are split into text units. Each text unit is paired with each candidate factor and processed by the verifier.

For a given post and factor, unit-level probabilities are aggregated to produce a post-level probability.

Each verifier contains five folds:

    fold0
    fold1
    fold2
    fold3
    fold4

Within each verifier:

    verifier_probability =
        mean(fold0, fold1, fold2, fold3, fold4)

The final probability is:

    final_probability =
        0.60 * taxonomy_probability
        + 0.40 * evidence_probability

Each label has a validation-derived threshold.

The combined threshold is:

    threshold =
        0.60 * taxonomy_threshold
        + 0.40 * evidence_threshold

A global minimum threshold of 0.50 is applied.

For the following labels, the minimum threshold is 0.60:

- meaning in life
- sense of responsibility
- exposure to others' suicide
- poor school performance
- sexual orientation related issues
- cognitive deficits
- substance use

A factor is predicted when:

    final_probability >= threshold

## Factor taxonomy

The model predicts the following 24 factors:

1. hopelessness
2. low self-esteem
3. emotion dysregulation
4. poor social support
5. coping strategy
6. psychological capital
7. suicide means (with access)
8. interpersonal difficulty
9. mental health issues
10. prior self-harm or suicidal thought/attempt
11. stressful life event
12. dysfunctional family
13. social support
14. interpersonal violence
15. physical health/characteristic
16. sense of responsibility
17. low socio-economic status
18. traumatic experience
19. meaning in life
20. cognitive deficits
21. substance use
22. exposure to others' suicide
23. poor school performance
24. sexual orientation related issues

## Validation performance

### Taxonomy Verifier

Five-fold pooled out-of-fold performance:

    Macro-F1:     0.61298
    Micro-F1:     0.69182
    Samples-F1:   0.59882
    Weighted-F1:  0.70033
    Exact Match:  0.22752

Average gold factors per post:

    2.9187

Average predicted factors per post:

    3.5651

### Evidence-aware Taxonomy Verifier

Five-fold pooled out-of-fold performance:

    Macro-F1:     0.60704
    Micro-F1:     0.67763
    Samples-F1:   0.59735
    Weighted-F1:  0.68915
    Exact Match:  0.17982

Average gold factors per post:

    2.9187

Average predicted factors per post:

    3.8489

The final system uses probability-level ensembling of the two verifier models.

## Model files

The model contains five checkpoints for each verifier:

    models/
    ├── taxonomy/
    │   ├── fold0.pt
    │   ├── fold1.pt
    │   ├── fold2.pt
    │   ├── fold3.pt
    │   ├── fold4.pt
    │   ├── fold0_summary.json
    │   ├── fold1_summary.json
    │   ├── fold2_summary.json
    │   ├── fold3_summary.json
    │   └── fold4_summary.json
    └── evidence/
        ├── fold0.pt
        ├── fold1.pt
        ├── fold2.pt
        ├── fold3.pt
        ├── fold4.pt
        ├── fold0_summary.json
        ├── fold1_summary.json
        ├── fold2_summary.json
        ├── fold3_summary.json
        └── fold4_summary.json

Checkpoint files are stored using Git Large File Storage (Git LFS).

After cloning the repository, Git LFS should automatically download the model files if Git LFS is installed.

If necessary:

    git lfs install
    git lfs pull

## Checkpoint integrity

SHA256 hashes for all model checkpoints are stored in:

    Factors/factor_ensemble/MODEL_SHA256SUMS

Verify them from the repository root with:

    sha256sum -c Factors/factor_ensemble/MODEL_SHA256SUMS

All ten entries should report:

    OK

## Package structure

    Factors/factor_ensemble/
    ├── README.md
    ├── predict.py
    ├── requirements.txt
    ├── MODEL_SHA256SUMS
    ├── examples/
    │   └── example_input.csv
    ├── models/
    │   ├── taxonomy/
    │   └── evidence/
    └── src/
        ├── taxonomy_verifier.py
        ├── evidence_verifier.py
        └── ensemble.py

## Reproducibility checks

The standalone inference package was verified against the previous inference pipeline.

For the same input posts:

    same row identifiers: True
    same factor strings: True
    exact dataframe equality: True

The package was also tested after replacing local symbolic links with the actual checkpoint files. The resulting prediction CSV was byte-identical.

## Notes

- GPU inference is strongly recommended.
- Additional input metadata columns are allowed.
- The model does not use `row_id` as a predictive feature.
- Predictions depend on the post text and the trained model checkpoints.
- Temporary intermediate probability files are automatically removed after prediction.
