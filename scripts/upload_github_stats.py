"""Upload the local GitHub stats CSVs to S3.

Run locally after scripts/collect_github_stats.py. Uploads to fixed keys
({prefix}github_repo_stats.csv, {prefix}github_user_stats.csv), overwriting
in place so the app always reads the latest snapshot at stable keys.

AWS credentials come from the default boto3 chain (env vars, ~/.aws, SSO).

Usage:
    uv run python scripts/upload_github_stats.py [--bucket <name>] [--prefix github-stats/] [--dir data/github_stats]
"""

import argparse
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

CSV_FILENAMES = ["github_repo_stats.csv", "github_user_stats.csv"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=os.getenv("GITHUB_STATS_BUCKET"))
    parser.add_argument("--prefix", default=os.getenv("GITHUB_STATS_PREFIX", "github-stats/"))
    parser.add_argument("--dir", type=Path, default=Path("data/github_stats"))
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("No bucket given: pass --bucket or set GITHUB_STATS_BUCKET in .env")
    prefix = args.prefix if args.prefix.endswith("/") or not args.prefix else args.prefix + "/"

    missing = [name for name in CSV_FILENAMES if not (args.dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {args.dir}/ — run scripts/collect_github_stats.py first."
        )

    s3 = boto3.client("s3")
    for name in CSV_FILENAMES:
        key = f"{prefix}{name}"
        s3.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=(args.dir / name).read_bytes(),
            ContentType="text/csv",
        )
        print(f"uploaded s3://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
