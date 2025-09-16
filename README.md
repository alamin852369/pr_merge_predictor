# PR Merge Predictor (Modularized)

This project refactors your single script into a small, readable Python package with clear responsibilities.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
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

Key entry points:
- `src/main.py` is the CLI. See `--help` for options like `--top-n` and `--seed`.
- Figures and CSVs are written under `results/` (same as your original script).

> Note: The Hugging Face parquet URIs are preserved. If an auth token is required,
> log in before running:
>
> ```python
> from huggingface_hub import login
> login()  # paste your token
> ```
