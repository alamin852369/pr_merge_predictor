import pandas as pd

def read_parquet_safe(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_pr_repo(pr_path: str, repo_path: str):
    pr = normalize_columns(read_parquet_safe(pr_path))
    repo = normalize_columns(read_parquet_safe(repo_path))
    return pr, repo

def load_commit_details(commit_details_path: str | None):
    if not commit_details_path:
        return None
    cd = read_parquet_safe(commit_details_path)
    # keep only used columns; normalize types
    cols = ['pr_id','filename','additions','deletions']
    cd = cd[cols].copy()
    cd['additions'] = pd.to_numeric(cd['additions'], errors='coerce').fillna(0)
    cd['deletions'] = pd.to_numeric(cd['deletions'], errors='coerce').fillna(0)
    return cd
