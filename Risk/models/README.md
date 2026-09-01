# Risk Assessment Model Weights

The final Risk Assessment system uses 20 DeBERTa-v3-large checkpoints:

- ML384: seeds 514, 9, and 42, with five folds per seed (15 checkpoints)
- ML640: seed 514, with five folds (5 checkpoints)

The large model weights are distributed separately from the source-code
package.

## 1. Download the model weights

Download all 20 `.safetensors` files from:

https://github.com/DawnMoon-P-C/BigdataSuicideRisk-Risk/releases/tag/risk-final-s1-v1

The release contains:

```text
ml384_seed514_fold0.safetensors
...
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
```

Place all 20 downloaded files in one directory, for example:

```text
downloaded_weights/
```

Do not rename the files.

## 2. Prepare the Hugging Face checkpoint directories

The source package contains:

```text
models/
├── prepare_models.py
├── weights.sha256
└── runtime/
    ├── ml384/
    │   ├── config.json
    │   ├── tokenizer_config.json
    │   └── tokenizer.json
    └── ml640/
        ├── config.json
        ├── tokenizer_config.json
        └── tokenizer.json
```

Run:

```bash
python models/prepare_models.py \
  --weights-dir downloaded_weights \
  --model-root models
```

The script:

1. checks that all 20 model files are present;
2. verifies all model weights against `models/weights.sha256`;
3. verifies the bundled ML384 and ML640 runtime metadata;
4. creates the Hugging Face checkpoint structure expected by
   `run_inference.py`.

The resulting structure is:

```text
models/
├── ml384/
│   ├── seed514/
│   │   ├── fold0/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   ├── fold3/
│   │   └── fold4/
│   ├── seed9/
│   │   ├── fold0/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   ├── fold3/
│   │   └── fold4/
│   └── seed42/
│       ├── fold0/
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       └── fold4/
└── ml640/
    └── seed514/
        ├── fold0/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        └── fold4/
```

Each fold directory contains:

```text
config.json
model.safetensors
tokenizer_config.json
tokenizer.json
```

`training_args.bin` is not required for inference.

## 3. Run final Risk inference

After preparing the models, return to the root of the Risk source package:

```bash
python run_inference.py \
  --input input.csv \
  --output output.csv \
  --model-root models
```

Optional probability diagnostics:

```bash
python run_inference.py \
  --input input.csv \
  --output output.csv \
  --model-root models \
  --probabilities-output probabilities.csv
```

## 4. Final system definition

The final Risk system uses:

- ML384: equal probability averaging across seeds 514, 9, and 42;
- five-fold probability averaging within each seed;
- ML640: seed 514 with five-fold probability averaging;
- deterministic routing using full untruncated token count including
  special tokens;
- `token_count_with_specials <= 384` -> ML384;
- `token_count_with_specials > 384` -> ML640;
- scalar temperature scaling with `T = 1.8528777092173991`.

The external package reconstruction workflow was validated using all 20
release-style model files and reproduced all 378 final Risk predictions of
the frozen reference system.
