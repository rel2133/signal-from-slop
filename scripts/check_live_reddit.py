from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signal_from_the_slop.reddit_client import RedditClient, build_subreddit_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test live Reddit collection.")
    parser.add_argument("--subreddit", default="stocks", help="Subreddit name to scrape.")
    parser.add_argument("--days", type=int, default=1, help="Lookback window in whole days.")
    parser.add_argument("--max-posts", type=int, default=3, help="Maximum posts to fetch.")
    parser.add_argument("--max-comments", type=int, default=2, help="Maximum comments per thread to fetch.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    source = build_subreddit_source(args.subreddit)
    selected_sources = [{**asdict(source), "source_id": 1}]
    client = RedditClient(ROOT / "data" / "fake_reddit_data.json")
    window_end = datetime.now(UTC)
    window_start = window_end - timedelta(days=max(args.days, 1))
    items = client.collect_live_items(
        selected_sources=selected_sources,
        window_start=window_start,
        window_end=window_end,
        max_posts_per_source=max(args.max_posts, 1),
        max_comments_per_thread=max(args.max_comments, 0),
        include_comments=args.max_comments > 0,
    )

    posts = [item for item in items if item["item_type"] == "post"]
    comments = [item for item in items if item["item_type"] == "comment"]
    print(f"Collected {len(items)} items from r/{args.subreddit} in the last {max(args.days, 1)} day(s).")
    print(f"Posts: {len(posts)} | Comments: {len(comments)}")
    if posts:
        latest = posts[-1]
        print(f"Latest post: {latest['thread_title'][:120]}")
        print(f"Latest created_time: {latest['created_time']}")
        print(f"Latest permalink: {latest['permalink']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
