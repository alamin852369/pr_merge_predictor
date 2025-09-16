from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def make_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

def make_logit_pipeline(preprocessor, seed: int):
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
    ])

def extract_coefficients(fitted_pipeline, numeric_cols, categorical_cols) -> pd.DataFrame:
    prep = fitted_pipeline.named_steps['prep']
    clf = fitted_pipeline.named_steps['clf']

    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = list(numeric_cols)
        ohe = None
        for name, trans, cols in prep.transformers_:
            if name == 'cat':
                ohe = trans
                cat_cols = cols
                break
        if hasattr(ohe, 'categories_'):
            for base, cats in zip(cat_cols, ohe.categories_):
                feature_names.extend([f"{base}_{c}" for c in cats])

    feature_names = [re.sub(r'^(num|cat)__', '', f) for f in feature_names]
    coefs = clf.coef_.ravel()
    coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df['odds_ratio'] = np.exp(coef_df['coef'])
    return coef_df.sort_values('abs_coef', ascending=False)
