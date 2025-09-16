# PR Merge Predictor

This project refactors your single script into a small, readable Python package with clear responsibilities.

## Quick Start

```bash
pip install -r requirements.txt

# Run with curated subset (default)
python -m src.main --use-curated

# Or run with 'all_' datasets (no commit_details/review_comments support)
python -m src.main --no-use-curated
```

## Project Layout

```
src/
  config.py              # global flags, paths, seed
  data_io.py             # loading parquet + basic normalization
  features.py            # feature engineering & modeling table build
  model.py               # preprocessing, training, coef extraction
  evaluation.py          # metrics, plots, artifact saving
  main.py                # end-to-end orchestration + CLI
results/
  figs/                  # saved figures (created at runtime)
  tables/                # saved CSVs (created at runtime)
```



##Overview

This project addresses the task of predicting whether a GitHub Pull Request (PR) will be merged or closed without merging.

The pipeline covers:

• Dataset loading (curated PR data from HuggingFace).

• Feature engineering (text, time, patch statistics, repository metadata).

• Model training with Logistic Regression.

• Evaluation and visualization of results.

##Dataset

We use the AIDev dataset (hao-li/AIDev) hosted on HuggingFace.
Two variants are supported:

• Curated subset (default): includes commit details and review comments.

• Full dataset: larger but without commit/review features.

Data paths are configurable in src/config.py
