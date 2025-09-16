from __future__ import annotations
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

def class_balance(series: pd.Series) -> pd.DataFrame:
    vc = series.value_counts().sort_index()
    total = len(series)
    rates = (vc / total).rename("rate")
    return pd.concat([vc.rename("count"), rates], axis=1)

def save_class_balance_tables(y, y_train, y_test, outdir_tables: str):
    full_balance = class_balance(y)
    train_balance = class_balance(y_train)
    test_balance  = class_balance(y_test)

    os.makedirs(outdir_tables, exist_ok=True)
    full_balance.to_csv(os.path.join(outdir_tables, "class_balance_full.csv"))
    train_balance.to_csv(os.path.join(outdir_tables, "class_balance_train.csv"))
    test_balance.to_csv(os.path.join(outdir_tables, "class_balance_test.csv"))
    return full_balance, train_balance, test_balance

def save_test_barplot(test_balance: pd.DataFrame, outdir_figs: str):
    os.makedirs(outdir_figs, exist_ok=True)
    plt.figure()
    (test_balance["count"]).plot(kind="bar")
    plt.title("Class Counts – Test Split")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_figs, "class_balance_test.png"), bbox_inches="tight")
    plt.close()

def evaluate_and_save(model, X_test, y_test, outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    # Metrics
    report = classification_report(y_test, pred, digits=3, output_dict=True)
    auc = roc_auc_score(y_test, proba)

    # Save predictions
    pd.DataFrame({
        "y_true": y_test,
        "y_prob": proba,
        "y_pred": pred
    }).to_csv(os.path.join(outdir_tables, "test_predictions_logit.csv"), index=False)

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Logistic Regression – ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_figs, "roc_curve_logit.png"), bbox_inches="tight")
    plt.close()

    return report, auc, pred, proba

def save_coef_plot(coef_df_sorted: pd.DataFrame, top_n: int, outdir_figs: str):
    import matplotlib.pyplot as plt
    os.makedirs(outdir_figs, exist_ok=True)
    sub = coef_df_sorted.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n*0.4)))
    plt.barh(sub['feature'], sub['coef'])
    plt.xlabel('Coefficient (log-odds)')
    plt.title(f'Logistic Regression – Top {top_n} Feature Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_figs, "logit_top_coefs.png"), bbox_inches="tight")
    plt.close()
