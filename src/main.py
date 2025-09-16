from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Paths, PathsAll, DEFAULT_SEED
from .data_io import load_pr_repo, load_commit_details
from .features import build_base_table, build_patch_features, assemble_model_table
from .model import make_preprocessor, make_logit_pipeline, extract_coefficients
from .evaluation import (
    save_class_balance_tables, save_test_barplot, evaluate_and_save, save_coef_plot
)

def parse_args():
    ap = argparse.ArgumentParser(description="Train a simple PR merge predictor (logistic regression).")
    ap.add_argument("--use-curated", action="store_true", default=True, help="Use curated subset (default: True)")
    ap.add_argument("--no-use-curated", action="store_false", dest="use_curated", help="Use 'all_' datasets")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N coefficients to plot")
    ap.add_argument("--results-dir", type=str, default="results", help="Directory for figs/tables outputs")
    return ap.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    paths = Paths() if args.use_curated else PathsAll()

    # 1) Load data
    pr, repo = load_pr_repo(paths.pr_path, paths.repo_path)
    cd = load_commit_details(paths.commit_details_path)

    # 2) Features
    df_base = build_base_table(pr)
    patch_feats = build_patch_features(cd)
    model_df, numeric_cols, categorical_cols = assemble_model_table(df_base, repo, patch_feats)

    # 3) Train/test split
    X = model_df[numeric_cols + categorical_cols]
    y = model_df['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # 4) Diagnostics + save class balance
    out_figs = os.path.join(args.results_dir, "figs")
    out_tables = os.path.join(args.results_dir, "tables")

    full_balance, train_balance, test_balance = save_class_balance_tables(y, y_train, y_test, out_tables)
    save_test_barplot(test_balance, out_figs)

    # 5) Model
    preproc = make_preprocessor(numeric_cols, categorical_cols)
    model = make_logit_pipeline(preproc, seed=args.seed)
    model.fit(X_train, y_train)

    # 6) Evaluate + save artifacts
    report, auc, pred, proba = evaluate_and_save(model, X_test, y_test, out_tables, out_figs)

    # 7) Coefficients
    coef_df_sorted = extract_coefficients(model, numeric_cols, categorical_cols)
    coef_df_sorted.to_csv(os.path.join(out_tables, "logit_feature_importance.csv"), index=False)
    save_coef_plot(coef_df_sorted, args.top_n, out_figs)

    # 8) Save a compact modeling table sample
    sample = model_df.sample(min(1000, len(model_df)), random_state=args.seed)
    sample.to_csv(os.path.join(out_tables, "sample_model_table.csv"), index=False)

    # 9) Human-readable prints
    print("=== Logistic Regression ===")
    # compact metrics
    from sklearn.metrics import classification_report, roc_auc_score
    print(classification_report(y_test, (proba >= 0.5).astype(int), digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    print()
    print("Artifacts saved under ./results/")

if __name__ == "__main__":
    main()
