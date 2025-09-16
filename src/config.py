from dataclasses import dataclass

DEFAULT_SEED = 42

@dataclass
class Paths:
    # Curated (richer) subset
    pr_path: str = "hf://datasets/hao-li/AIDev/pull_request.parquet"
    repo_path: str = "hf://datasets/hao-li/AIDev/repository.parquet"
    commit_details_path: str | None = "hf://datasets/hao-li/AIDev/pr_commit_details.parquet"
    review_comments_path: str | None = "hf://datasets/hao-li/AIDev/pr_review_comments.parquet"

@dataclass
class PathsAll:
    # 'all_' version (no commit_details/review_comments)
    pr_path: str = "hf://datasets/hao-li/AIDev/all_pull_request.parquet"
    repo_path: str = "hf://datasets/hao-li/AIDev/all_repository.parquet"
    commit_details_path: str | None = None
    review_comments_path: str | None = None
