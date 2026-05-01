# Twibot_Detection_Research

This repository studies how well bot-detection models transfer across Twitter/X datasets from different time periods.

## Logistic Regression Baseline

The repo now includes a self-contained logistic regression training pipeline that only depends on `numpy` and `pandas`.

Run it with:

```bash
/Users/hsla/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/train_logistic_regression.py \
  --dataset cresci_2017
```

Optional flags:

- `--data-root /path/to/data/processed`
- `--max-rows 50000`
- `--output-dir artifacts/logistic_regression`

Outputs include:

- `metrics.json`
- `coefficients.csv`
- `training_history.csv`
- `test_predictions.csv`
- `model.pkl`

To generate a paper-ready summary after running experiments:

```bash
/Users/hsla/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/summarize_logistic_results.py
```
