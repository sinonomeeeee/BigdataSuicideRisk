# Evidence Pipeline

This README documents the final evidence-extraction pipeline implemented by `run_evidence_pipeline_local.py`.

## Pipeline overview

The production workflow is:

1. Retrieve the previously obtained suicide-risk predictions from the risk pipeline.
2. Reuse an existing completed evidence model when available.
3. If a completed model is not available, train the evidence model from `train_none.xlsx` plus the human-reviewed `hard_negatives_modified.csv`.
4. Apply the trained evidence model to the **test file supplied/run by the IEEE competition officials**.
5. Aggregate candidate-level evidence scores into one evidence prediction per test row.
6. Add the aggregated evidence prediction to the competition result file.

`run_evidence_pipeline_local.py` is the canonical/final script for this pipeline.

---

## Requirements

Typical Python dependencies used by the script are:

```bash
pip install "transformers>=4.46,<5" "datasets>=3.1,<4" \
  "accelerate>=1.1,<2" sentencepiece scikit-learn pandas openpyxl torch
```

For NVIDIA GPU execution, install a CUDA-enabled PyTorch build appropriate for the machine.

The default model is:

```text
microsoft/deberta-v3-base
```

---

## Required files

### Training data

- `train_none.xlsx` — source training posts and gold evidence annotations.
- `hard_negatives_modified.csv` — reviewed candidate labels/hard negatives used to correct and strengthen the training set.

These are controlled by `--posts` and `--reviewed-negatives`.

### Risk-pipeline results

The evidence pipeline does **not** independently recreate the competition risk prediction. Supply the previously obtained risk-pipeline output with `--risk-file`.

Default:

```text
SatAndSeriousTokyoScript.csv
```

The risk file must contain the row identifier (default `row_id`) and a suicide-risk prediction column. The script automatically detects common risk-column names such as `risk_level` or `suicide risk`.

The risk file must cover the test rows. Duplicate `row_id` values in the risk file are rejected, and inference fails if a test row has no corresponding risk value.

### Test file — IMPORTANT

The test dataset supplied/run by the IEEE competition officials is passed with:

```text
--leaderboard TEST_FILE
```

Despite the argument name `--leaderboard`, **this is the test-file argument for final competition inference**. The name is retained because the v18 script calls the official inference dataset the “leaderboard” dataset internally.

Default:

```text
leaderboard.xlsx
```

For example, if the competition officials provide `test.xlsx`:

```bash
--leaderboard test.xlsx
```

If they provide a CSV:

```bash
--leaderboard test.csv
```

The loader supports the table formats accepted by the script (including CSV/XLSX).

#### Required test-file columns

The preferred test-file structure is:

```text
row_id | post_y
```

- `row_id` — unique identifier used to join the test row to the risk-pipeline result.
- `post_y` — post text used to generate evidence candidates and run evidence inference.

By default the script uses:

```text
--row-id-col row_id
--post-col post_y
```

If `row_id` is absent, v18 creates row IDs from the input DataFrame index. For official competition inference, however, the test file should retain the official `row_id` so that evidence predictions can be joined reliably to the risk results and final result file.

If `post_y` is absent, the inference code falls back to a column named `post` or `post_y`. If neither exists, inference stops with an error.

Optional identifier columns such as `anon_user_id`, `user_id`, and `post_id` are preserved in generated candidates when present.

**Critical:** `row_id` values in the test file and risk file must identify the same rows. v18 normalizes row IDs for joining (for example, `10` and `10.0` can match), but the safest procedure is to preserve the official IDs unchanged in both files.

---

## Recommended final run

With the standard filenames:

```bash
python scripts/run_evidence_pipeline_local.py \
  --posts train_none.xlsx \
  --reviewed-negatives hard_negatives_modified.csv \
  --leaderboard leaderboard.xlsx \
  --risk-file SatAndSeriousTokyoScript.csv \
  --output-dir evidence_local_run \
  --device auto
```

For an official test file named `test.xlsx` and a risk result named `risk_results.csv`:

```bash
python scripts/run_evidence_pipeline_local.py \
  --posts train_none.xlsx \
  --reviewed-negatives hard_negatives_modified.csv \
  --leaderboard test.xlsx \
  --risk-file risk_results.csv \
  --output-dir evidence_local_run \
  --device cuda \
  --fp16
```

The second command is the important competition pattern: **`--leaderboard` points to the IEEE official test file, while `--risk-file` points to the already-obtained risk predictions for those same test rows.**

---

## Model reuse and training behavior

The model directory is:

```text
evidence_local_run/best_model/
```

If a completed model exists there and `--force` is not supplied, v18 reuses the saved model instead of retraining it.

If no completed model exists, the script constructs/reuses candidates and training pairs, applies the reviewed labels from `hard_negatives_modified.csv`, creates grouped train/validation/test splits, and trains the model. Incomplete training can resume from the latest checkpoint unless `--no-resume` is specified.

Therefore the normal production behavior is:

```text
completed best_model exists
    -> reuse model

completed best_model missing
    -> train/resume model from training data
```

Do **not** use `--force` for normal final inference when the validated model should be reused. `--force` deliberately ignores reusable outputs and rebuilds/retrains stages.

---

## Test inference and evidence aggregation

For the file supplied through `--leaderboard`, v18:

1. Loads the official test posts.
2. Generates E1-style evidence candidates from each post.
3. Joins candidates to the previously obtained risk prediction by `row_id`.
4. Builds model input as:

```text
Candidate: <candidate>
Post: <post>
```

5. Scores each non-`none` candidate with the evidence model.
6. Aggregates candidate scores to row-level evidence predictions.
7. Applies the risk-based override: a row whose risk is `Indicator` produces evidence `none`.
8. Merges adjacent surviving evidence candidates in the original post.
9. Writes one aggregated prediction per test `row_id`.

The final inference aggregation uses an overlap threshold fixed at `0.0`. If `metrics/best_aggregation_settings.json` exists, its tuned threshold and maximum-phrase setting are reused; otherwise the configured defaults are used.

---

## Final evidence output

The main row-level output is:

```text
evidence_local_run/leaderboard_inference/LB_aggregated_evidence.csv
```

Important columns include:

- `row_id`
- `predicted` — final aggregated evidence phrase(s), or `none`
- `prob_evidence`
- `suicide_risk`
- `indicator_override`
- `post`

The candidate-level inference output is:

```text
evidence_local_run/leaderboard_inference/LB_candidate_scores.csv
```

Generated test candidates are saved as:

```text
evidence_local_run/leaderboard_inference/LB_E1_phrase_candidates.csv
```

An inference summary is saved as:

```text
evidence_local_run/leaderboard_inference/LB_inference_summary.json
```

---

## Adding evidence to the competition result file

Use `LB_aggregated_evidence.csv` as the evidence source for the final competition result. Join it to the official result/submission rows by `row_id` and copy the row-level `predicted` value into the submission's evidence field.

The final result must preserve the official test/result row IDs and ordering. Do not join evidence by row position when an official `row_id` is available.

Conceptually:

```text
official result/test row_id
        +
previous risk prediction for row_id
        +
LB_aggregated_evidence.predicted for row_id
        -> final competition result row
```

The exact name of the evidence column in the competition result file should follow the official IEEE submission schema.

---

## Command-line arguments

### Data and inference inputs

| Argument | Default | Purpose |
|---|---|---|
| `--posts` | `train_none.xlsx` | Training posts/gold evidence source. |
| `--reviewed-negatives` | `hard_negatives_modified.csv` | Human-reviewed labels/hard negatives. |
| `--training-pairs` | none | Optional prebuilt training-pairs CSV. |
| `--candidates` | none | Optional prebuilt training candidate CSV. |
| `--leaderboard` | `leaderboard.xlsx` | **Official test file for final evidence inference.** |
| `--risk-file` | `SatAndSeriousTokyoScript.csv` | Previously obtained risk-pipeline predictions for the same test rows. |
| `--skip-leaderboard` | false | Skip official/test inference. Do not use for a final competition run. |
| `--output-dir` | `evidence_local_run` | Pipeline output/model directory. |

### Column names

| Argument | Default | Purpose |
|---|---|---|
| `--row-id-col` | `row_id` | Row identifier used throughout and for the test/risk join. |
| `--post-col` | `post_y` | Post-text column. In inference, `post`/`post_y` fallback is supported. |
| `--candidate-col` | `candidate` | Candidate evidence column. |
| `--label-col` | `is_match` | Training label column. |
| `--gold-col` | `gold_phrases` | Gold evidence column. |
| `--risk-col` | auto | Training-side risk-column override/detection setting. Final risk-file inference detects its risk column automatically. |

### Model/training

| Argument | Default |
|---|---:|
| `--model-name` | `microsoft/deberta-v3-base` |
| `--max-length` | `256` |
| `--learning-rate` | `1e-5` |
| `--weight-decay` | `0.01` |
| `--epochs` | `5.0` |
| `--train-batch-size` | `8` |
| `--eval-batch-size` | `16` |
| `--gradient-accumulation-steps` | `2` |
| `--warmup-ratio` | `0.1` |
| `--early-stopping-patience` | `2` |
| `--test-size` | `0.20` |
| `--validation-size-from-remaining` | `0.125` |
| `--seed` | `42` |

### Runtime/reuse

| Argument | Default | Purpose |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--fp16` | false | Enable FP16 where supported. |
| `--bf16` | false | Enable BF16 where supported. |
| `--force` | false | Ignore reusable outputs and rebuild/retrain stages. |
| `--no-resume` | false | If training is incomplete, do not resume from the latest checkpoint. |
| `--expected-candidate-count` | `79478` | Training-candidate sanity check; `0` disables it. It is not imposed on the official test file. |

### Reviewed-label behavior

| Argument | Default behavior |
|---|---|
| `--hard-negative-extra-copies` | `1`; adds one extra training-only copy of each confirmed true-negative pair. |
| `--do-not-promote-annotation-errors` | Off; by default reviewed annotation errors are promoted to positive label `1`. |
| `--keep-ambiguous` | Off; by default ambiguous reviewed pairs are removed. |

### Aggregation

| Argument | Default |
|---|---:|
| `--no-tune-aggregation` | false |
| `--threshold-grid` | `0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90` |
| `--overlap-grid` | `0.0` |
| `--max-phrases-grid` | `0,1,2,3,4` (`0` = unlimited) |
| `--default-threshold` | `0.70` |
| `--default-overlap-threshold` | `0.0` |
| `--default-max-phrases` | `0` (unlimited) |
| `--min-match-word-ratio` | `1/3` |
| `--max-match-word-ratio` | `3.0` |

---

## Final competition checklist

- [ ] Use `run_evidence_pipeline_local.py`.
- [ ] Place/provide `train_none.xlsx`.
- [ ] Place/provide `hard_negatives_modified.csv`.
- [ ] Retrieve the completed risk-pipeline predictions for the official test rows.
- [ ] Pass the **IEEE official test file** to `--leaderboard`.
- [ ] Pass the corresponding risk-pipeline output to `--risk-file`.
- [ ] Confirm test and risk files use matching official `row_id` values.
- [ ] Reuse `evidence_local_run/best_model/` when the completed validated model exists.
- [ ] Allow training/resume only when the completed model is unavailable.
- [ ] Run candidate generation, scoring, and aggregation.
- [ ] Check `leaderboard_inference/LB_aggregated_evidence.csv`.
- [ ] Join `predicted` evidence into the official competition result by `row_id`.
- [ ] Preserve the official result schema and row ordering.

---

## Canonical script

```text
run_evidence_pipeline_local.py
```

Later experimental script numbers are not part of this final documented workflow unless explicitly adopted in the future.
