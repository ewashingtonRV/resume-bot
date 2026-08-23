import io
from unittest.mock import patch

import pytest

from src import stats_provider
from src.stats_provider import CachedGitHubStats

USER_CSV = """as_of_date,lookback_days,total_commits,total_pull_requests
2026-08-22,30,12,5
2026-08-22,365,140,52
"""

REPO_CSV = """as_of_date,lookback_days,category,repo_name,commits,pull_requests,total_code_changes
2026-08-22,30,raas,optum-now-core-graphql-api,4,2,1250
2026-08-22,365,raas,optum-now-core-graphql-api,40,21,15000
2026-08-22,365,ds-lead,rvo-eval-sdk,7,3,900
"""


def _s3_stub(bodies):
    """Return an object mimicking boto3's s3 client for the given key->csv map."""
    class Stub:
        def get_object(self, Bucket, Key):
            if Key not in bodies:
                raise RuntimeError(f"NoSuchKey: {Key}")
            return {"Body": io.BytesIO(bodies[Key].encode("utf-8"))}
    return Stub()


@pytest.fixture(autouse=True)
def stats_env(monkeypatch):
    monkeypatch.setenv("GITHUB_STATS_BUCKET", "test-bucket")
    monkeypatch.delenv("GITHUB_STATS_PREFIX", raising=False)
    stats_provider._load_stats.cache_clear()
    yield
    stats_provider._load_stats.cache_clear()


def _patched_client(bodies=None):
    if bodies is None:
        bodies = {
            "github-stats/github_user_stats.csv": USER_CSV,
            "github-stats/github_repo_stats.csv": REPO_CSV,
        }
    return patch("src.stats_provider.boto3.client", return_value=_s3_stub(bodies))


def test_missing_bucket_env_raises(monkeypatch):
    monkeypatch.delenv("GITHUB_STATS_BUCKET", raising=False)
    with pytest.raises(ValueError, match="GITHUB_STATS_BUCKET"):
        CachedGitHubStats()


def test_user_stats_exact_window():
    with _patched_client():
        stats = CachedGitHubStats().get_user_stats(lookback_days=30)
    assert stats == {
        "as_of_date": "2026-08-22", "lookback_days": 30,
        "total_commits": 12, "total_pull_requests": 5,
    }


def test_user_stats_snaps_to_nearest_window():
    with _patched_client():
        stats = CachedGitHubStats().get_user_stats(lookback_days=400)
    assert stats["lookback_days"] == 365
    assert stats["total_commits"] == 140


def test_repo_stats_filters_category_and_window():
    with _patched_client():
        result = CachedGitHubStats().get_repo_stats(
            lookback_days=365, intent_category_name="raas"
        )
    assert result["as_of_date"] == "2026-08-22"
    assert result["lookback_days"] == 365
    assert result["repos"] == [{
        "repo_name": "optum-now-core-graphql-api",
        "commits": 40, "pull_requests": 21, "total_code_changes": 15000,
    }]


def test_repo_stats_unknown_category_returns_empty_repos():
    with _patched_client():
        result = CachedGitHubStats().get_repo_stats(
            lookback_days=30, intent_category_name="not-a-category"
        )
    assert result["repos"] == []


def test_s3_failure_raises_with_bucket_and_key():
    with _patched_client(bodies={}):
        with pytest.raises(RuntimeError, match=r"s3://test-bucket/github-stats/"):
            CachedGitHubStats().get_user_stats()


def test_empty_csv_raises():
    header_only = USER_CSV.splitlines()[0] + "\n"
    with _patched_client(bodies={
        "github-stats/github_user_stats.csv": header_only,
        "github-stats/github_repo_stats.csv": REPO_CSV,
    }):
        with pytest.raises(RuntimeError, match="empty"):
            CachedGitHubStats().get_user_stats()
