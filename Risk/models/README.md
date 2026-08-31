# Risk Assessment Model Weights

The trained model weights are distributed separately because the final
Risk Assessment ensemble contains 20 DeBERTa-v3-large checkpoints.

Model download URL:

https://github.com/sinonomeeeee/BigdataSuicideRisk/releases/tag/risk-final-s1-v1

Expected directory structure:

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

Each fold directory should contain at least:

- config.json
- model.safetensors
- tokenizer_config.json
- tokenizer.json

The final system uses:

- ML384: equal probability average across seeds 514, 9, and 42
- ML640: seed 514
- five-fold probability averaging within each seed
- deterministic routing at token_count_with_specials > 384
- scalar temperature scaling with T = 1.8528777092173991
