"""Serve GitHub stats from the S3 CSV snapshot instead of the live API.

The deployed app holds no GitHub token: scripts/collect_github_stats.py and
scripts/upload_github_stats.py produce the snapshot locally, and this module
reads it back with the same method signatures GitHubStats exposed, so
src/chat.py's tool dispatch is unchanged.
"""

import csv
import io
import os
from functools import lru_cache

import boto3

REPO_STATS_KEY = "github_repo_stats.csv"
USER_STATS_KEY = "github_user_stats.csv"

_USER_INT_COLUMNS = ["lookback_days", "total_commits", "total_pull_requests"]
_REPO_INT_COLUMNS = ["lookback_days", "commits", "pull_requests", "total_code_changes"]


def _fetch_rows(s3, bucket: str, key: str, int_columns: list[str]) -> list[dict]:
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"GitHub stats snapshot unavailable (s3://{bucket}/{key}): {e}"
        ) from e
    rows = list(csv.DictReader(io.StringIO(body)))
    if not rows:
        raise RuntimeError(f"GitHub stats snapshot is empty (s3://{bucket}/{key})")
    for row in rows:
        for column in int_columns:
            row[column] = int(row[column])
    return rows


@lru_cache(maxsize=1)
def _load_stats(bucket: str, prefix: str) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Fetch and parse both CSVs once per process; failures are not cached."""
    s3 = boto3.client("s3")
    user_rows = _fetch_rows(s3, bucket, prefix + USER_STATS_KEY, _USER_INT_COLUMNS)
    repo_rows = _fetch_rows(s3, bucket, prefix + REPO_STATS_KEY, _REPO_INT_COLUMNS)
    return tuple(user_rows), tuple(repo_rows)


def _nearest_window(available: list[int], requested: int) -> int:
    # The tool schema enum should prevent unknown values; snap defensively
    # and report the window actually used rather than failing.
    return min(available, key=lambda w: abs(w - requested))


class CachedGitHubStats:
    """Drop-in replacement for GitHubStats backed by the S3 snapshot."""

    def __init__(self):
        self.bucket = os.getenv("GITHUB_STATS_BUCKET")
        if not self.bucket:
            raise ValueError(
                "GITHUB_STATS_BUCKET not set. Point it at the S3 bucket "
                "holding the stats snapshot (see README)."
            )
        self.prefix = os.getenv("GITHUB_STATS_PREFIX", "github-stats/")

    def get_user_stats(self, lookback_days: int = 365) -> dict:
        user_rows, _ = _load_stats(self.bucket, self.prefix)
        window = _nearest_window([r["lookback_days"] for r in user_rows], lookback_days)
        return dict(next(r for r in user_rows if r["lookback_days"] == window))

    def get_repo_stats(self, lookback_days: int = 365, intent_category_name: str = None) -> dict:
        _, repo_rows = _load_stats(self.bucket, self.prefix)
        window = _nearest_window(
            sorted({r["lookback_days"] for r in repo_rows}), lookback_days
        )
        return {
            "as_of_date": repo_rows[0]["as_of_date"],
            "lookback_days": window,
            "repos": [
                {
                    "repo_name": r["repo_name"],
                    "commits": r["commits"],
                    "pull_requests": r["pull_requests"],
                    "total_code_changes": r["total_code_changes"],
                }
                for r in repo_rows
                if r["lookback_days"] == window
                and r["category"] == intent_category_name
            ],
        }
