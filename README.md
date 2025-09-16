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
