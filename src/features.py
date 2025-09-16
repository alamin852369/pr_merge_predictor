from __future__ import annotations
import re
import numpy as np
import pandas as pd

TEST_RE = r'(^|/)(tests?|__tests__|spec)/|(_test\.(py|js|ts|go|rb|java)|\.spec\.(js|ts))$'

def _finalize_labelled_prs(pr: pd.DataFrame) -> pd.DataFrame:
    # Keep only PRs with a final state (merged or closed)
    mask_merged = pr.get('state', '').eq('merged') | pr.get('merged_at', pd.Series(index=pr.index)).notna()
    mask_closed = pr.get('state', '').eq('closed') & ~mask_merged
    df = pr.loc[mask_merged | mask_closed].copy()
    df['label'] = np.where(mask_merged.loc[df.index], 1, 0)
    return df

def _basic_text_time_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['title', 'body']:
        if col not in df.columns:
            df[col] = ""
    df['title_len'] = df['title'].fillna("").str.len()
    df['body_len']  = df['body'].fillna("").str.len()

    if 'created_at' in df.columns:
        created = pd.to_datetime(df['created_at'], errors='coerce')
        df['created_hour'] = created.dt.hour
        df['created_wday'] = created.dt.weekday
    else:
        df['created_hour'] = -1
        df['created_wday'] = -1
    return df

def _task_agent_defaults(df: pd.DataFrame) -> pd.DataFrame:
    if 'task_type' in df.columns:
        df['task_type'] = df['task_type'].fillna('unknown').astype(str)
    else:
        df['task_type'] = 'unknown'
    if 'agent' not in df.columns:
        df['agent'] = 'unknown'
    return df

def build_base_table(pr: pd.DataFrame) -> pd.DataFrame:
    df = _finalize_labelled_prs(pr)
    df = _basic_text_time_features(df)
    df = _task_agent_defaults(df)
    base_cols = ['id', 'repo_id', 'title_len', 'body_len', 'created_hour', 'created_wday',
                 'task_type', 'agent', 'label']
    return df[base_cols].copy()

def build_patch_features(cd: pd.DataFrame | None) -> pd.DataFrame | None:
    if cd is None:
        return None
    patch_feats = (
        cd.groupby('pr_id')
          .agg(loc_added=('additions','sum'),
               loc_deleted=('deletions','sum'),
               files_changed=('filename','nunique'))
          .reset_index()
    )
    cd = cd.copy()
    cd['is_test'] = cd['filename'].astype(str).str.contains(TEST_RE, regex=True, na=False)
    has_tests = (cd.groupby('pr_id')['is_test'].max().astype(int).rename('has_tests').reset_index())
    patch_feats = patch_feats.merge(has_tests, on='pr_id', how='left')
    patch_feats = patch_feats.rename(columns={'pr_id':'id'})
    patch_feats['has_tests'] = patch_feats['has_tests'].fillna(0).astype(int)
    return patch_feats

def assemble_model_table(df_base: pd.DataFrame, repo: pd.DataFrame, patch_feats: pd.DataFrame | None):
    Xy = df_base.copy()
    if patch_feats is not None:
        Xy = Xy.merge(patch_feats, on='id', how='left')

    for c in ['loc_added','loc_deleted','files_changed','has_tests']:
        if c not in Xy.columns:
            Xy[c] = 0
        else:
            Xy[c] = pd.to_numeric(Xy[c], errors='coerce').fillna(0)

    # Repo stars proxy
    star_col = None
    for c in ['stars','stargazers_count','watchers','watchers_count']:
        if c in repo.columns:
            star_col = c; break
    if star_col and 'id' in repo.columns:
        repo_min = repo[['id', star_col]].rename(columns={'id':'repo_id', star_col:'repo_stars'})
        Xy = Xy.merge(repo_min, on='repo_id', how='left')
        Xy['repo_stars'] = Xy['repo_stars'].fillna(0)
    else:
        Xy['repo_stars'] = 0

    numeric_cols = [c for c in ['title_len','body_len','created_hour','created_wday',
                                'loc_added','loc_deleted','files_changed','has_tests','repo_stars']
                    if c in Xy.columns]
    categorical_cols = [c for c in ['task_type','agent'] if c in Xy.columns]

    features = numeric_cols + categorical_cols
    model_df = Xy[features + ['label']].dropna().copy()
    return model_df, numeric_cols, categorical_cols
