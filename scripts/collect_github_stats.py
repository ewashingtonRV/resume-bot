"""Collect GitHub stats into local CSV snapshots.

Run locally (needs GITHUB_TOKEN in .env). Produces the two CSVs the deployed
app serves via S3, one row set per precomputed lookback window:

    data/github_stats/github_repo_stats.csv
    data/github_stats/github_user_stats.csv

The windows written here must match GITHUB_LOOKBACK_WINDOWS in src/chat.py.

Usage:
    uv run python scripts/collect_github_stats.py [--windows 7 30 90 365] [--out-dir data/github_stats]
"""

import argparse
import asyncio
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import GitHubStats

REPO_STATS_FILENAME = "github_repo_stats.csv"
USER_STATS_FILENAME = "github_user_stats.csv"

REPO_STATS_COLUMNS = [
    "as_of_date", "lookback_days", "category", "repo_name",
    "commits", "pull_requests", "total_code_changes",
]
USER_STATS_COLUMNS = [
    "as_of_date", "lookback_days", "total_commits", "total_pull_requests",
]


def sum_user_stats(window_rows: list[dict]) -> dict:
    """Aggregate user totals from the per-repo rows of one window.

    Derived from the same REST data as the repo stats (the GraphQL
    contributionsCollection API returns zeros for org-private contributions
    under a fine-grained PAT). Repos can appear in more than one category,
    so dedupe by repo name before summing.
    """
    by_repo = {row["repo_name"]: row for row in window_rows}
    return {
        "total_commits": sum(r["commits"] for r in by_repo.values()),
        "total_pull_requests": sum(r["pull_requests"] for r in by_repo.values()),
    }


async def collect(windows: list[int]) -> tuple[list[dict], list[dict]]:
    stats = GitHubStats()
    as_of = date.today().isoformat()
    categories = [entry["category_name"] for entry in stats.get_repos()]

    repo_rows, user_rows = [], []
    for window in windows:
        window_rows = []
        for category in categories:
            for repo in await stats.get_repo_stats_async(
                lookback_days=window, intent_category_name=category
            ):
                window_rows.append({
                    "as_of_date": as_of,
                    "lookback_days": window,
                    "category": category,
                    **repo,
                })
        repo_rows.extend(window_rows)
        user_rows.append({
            "as_of_date": as_of,
            "lookback_days": window,
            **sum_user_stats(window_rows),
        })
        print(f"window {window}d: {len(window_rows)} repo rows")

    return repo_rows, user_rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="+", default=[7, 30, 90, 365])
    parser.add_argument("--out-dir", type=Path, default=Path("data/github_stats"))
    args = parser.parse_args()

    repo_rows, user_rows = asyncio.run(collect(args.windows))

    # GitHubStats logs per-repo fetch errors (e.g. 404 for repos the token
    # can't see) and returns empty lists, so a permissions problem would
    # otherwise produce an empty-but-"successful" snapshot.
    if not repo_rows:
        raise SystemExit(
            "No repo stats were collected for any window. Check the ERROR "
            "logs above — this usually means the token can't see the repos "
            "(pending org approval or repos not granted to the token)."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / REPO_STATS_FILENAME, REPO_STATS_COLUMNS, repo_rows)
    write_csv(args.out_dir / USER_STATS_FILENAME, USER_STATS_COLUMNS, user_rows)


if __name__ == "__main__":
    main()
