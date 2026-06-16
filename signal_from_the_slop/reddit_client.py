from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from math import ceil
from typing import Any
from urllib.parse import urlparse

import requests


SUBREDDIT_PATTERN = re.compile(r"^(?:/?r/)?([A-Za-z0-9_]+)$")
THREAD_URL_PATTERN = re.compile(r"^/r/([A-Za-z0-9_]+)/comments/([A-Za-z0-9]+)/?.*$")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_REDDIT_USER_AGENT = "script:signal-from-the-slop:0.1"
REDDIT_REQUEST_DELAY_SECONDS = 0.25
REDDIT_SUBREDDIT_RSS_LIMIT = 100
REDDIT_RATE_LIMIT_BUFFER_SECONDS = 2.0
REDDIT_RATE_LIMIT_RETRY_ATTEMPTS = 3


@dataclass(frozen=True)
class SourceCandidate:
    source_key: str
    source_type: str
    display_name: str
    normalized_value: str
    url: str


class RedditClient:
    """Reddit RSS collection layer with source-aware filtering."""

    def __init__(self) -> None:
        self._next_request_not_before = 0.0

    def collect_live_items(
        self,
        *,
        selected_sources: list[dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
        max_posts_per_source: int,
        max_comments_per_thread: int,
        include_comments: bool,
        run_type: str = "discovery",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        coverage_callback: Callable[[dict[str, Any]], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        session = self._build_scraper_session()
        deduped_items: dict[str, dict[str, Any]] = {}
        scrape_mode = "posts_plus_comments" if include_comments and max_comments_per_thread else "posts_only"
        canonical_trend_run = str(run_type).lower() == "measurement" and scrape_mode == "posts_only"

        ordered_sources = sorted(
            selected_sources,
            key=lambda source: 0 if source["source_type"] == "thread_url" else 1,
        )
        total_sources = len(ordered_sources)
        for source_index, source in enumerate(ordered_sources, start=1):
            if checkpoint is not None:
                checkpoint()
            if progress_callback is not None:
                progress_callback(
                    {
                        "message": f"Scraping source {source_index}/{total_sources}: {source['display_name']}",
                        "current": source_index - 1,
                        "total": total_sources,
                        "unit": "sources",
                    }
                )
            try:
                if source["source_type"] == "thread_url":
                    items, coverage = self._collect_scraped_thread(
                        session=session,
                        source=source,
                        window_start=window_start,
                        window_end=window_end,
                        max_comments_per_thread=max_comments_per_thread,
                        include_comments=include_comments,
                        scrape_mode=scrape_mode,
                        canonical_trend_run=canonical_trend_run,
                    )
                elif source["source_type"] == "subreddit":
                    items, coverage = self._collect_scraped_subreddit(
                        session=session,
                        source=source,
                        window_start=window_start,
                        window_end=window_end,
                        max_posts_per_source=max_posts_per_source,
                        max_comments_per_thread=max_comments_per_thread,
                        include_comments=include_comments,
                        scrape_mode=scrape_mode,
                        canonical_trend_run=canonical_trend_run,
                        progress_callback=progress_callback,
                        checkpoint=checkpoint,
                        source_index=source_index,
                        total_sources=total_sources,
                    )
                else:
                    items = []
                    coverage = self._build_source_coverage(
                        source=source,
                        window_start=window_start,
                        window_end=window_end,
                        scrape_mode=scrape_mode,
                        posts_requested=0,
                        comments_requested=0,
                        items=[],
                        canonical_trend_run=canonical_trend_run,
                        collection_completed=True,
                        collection_error="Unsupported source type.",
                    )
            except Exception as exc:
                if coverage_callback is not None:
                    coverage_callback(
                        self._build_source_coverage(
                            source=source,
                            window_start=window_start,
                            window_end=window_end,
                            scrape_mode=scrape_mode,
                            posts_requested=max_posts_per_source,
                            comments_requested=0,
                            items=[],
                            canonical_trend_run=canonical_trend_run,
                            collection_completed=False,
                            collection_error=str(exc),
                            rate_limited="rate-limit" in str(exc).lower() or "rate limited" in str(exc).lower(),
                        )
                    )
                raise

            for item in items:
                deduped_items.setdefault(item["item_id"], item)
            if coverage_callback is not None:
                coverage_callback(coverage)
            if progress_callback is not None:
                progress_callback(
                    {
                        "message": f"Finished source {source_index}/{total_sources}: {source['display_name']} ({len(items)} items)",
                        "current": source_index,
                        "total": total_sources,
                        "unit": "sources",
                    }
                )

        return sorted(deduped_items.values(), key=lambda row: row["created_time"])

    def _build_scraper_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": os.getenv("REDDIT_USER_AGENT", DEFAULT_REDDIT_USER_AGENT).strip()
                or DEFAULT_REDDIT_USER_AGENT,
                "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        return session

    def _collect_scraped_subreddit(
        self,
        *,
        session: requests.Session,
        source: dict[str, Any],
        window_start: datetime,
        window_end: datetime,
        max_posts_per_source: int,
        max_comments_per_thread: int,
        include_comments: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
        source_index: int = 1,
        total_sources: int = 1,
        scrape_mode: str = "posts_only",
        canonical_trend_run: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        subreddit = normalize_subreddit_name(str(source["normalized_value"]))
        feed_url = f"https://www.reddit.com/r/{subreddit}/new/.rss?limit={REDDIT_SUBREDDIT_RSS_LIMIT}"
        entries = self._fetch_rss_entries(session, feed_url)
        items: list[dict[str, Any]] = []
        feed_posts: list[tuple[ET.Element, datetime]] = [
            (entry, self._entry_timestamp(entry))
            for entry in entries
            if self._entry_id(entry).startswith("t3_")
        ]
        window_post_entries: list[ET.Element] = []

        for entry, created_at in feed_posts:
            if created_at > window_end:
                continue
            if created_at < window_start:
                break
            window_post_entries.append(entry)

        selected_post_entries = (
            window_post_entries[:max_posts_per_source]
            if max_posts_per_source
            else window_post_entries
        )
        posts = [
            self._rss_entry_to_item(entry, source, item_type="post")
            for entry in selected_post_entries
        ]
        items.extend(posts)

        if include_comments and max_comments_per_thread:
            for post_index, post in enumerate(posts, start=1):
                if checkpoint is not None:
                    checkpoint()
                if progress_callback is not None:
                    progress_callback(
                        {
                            "message": f"Fetching comments {post_index}/{len(posts)} for {source['display_name']}",
                            "current": source_index - 1,
                            "total": total_sources,
                            "unit": "sources",
                        }
                    )
                comments = self._collect_scraped_thread_comments(
                    session=session,
                    source=source,
                    thread_url=str(post["thread_url"]),
                    window_start=window_start,
                    window_end=window_end,
                    max_comments_per_thread=max_comments_per_thread,
                )
                items.extend(comments)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "message": f"Fetched comments {post_index}/{len(posts)} for {source['display_name']}",
                            "current": source_index - 1,
                            "total": total_sources,
                            "unit": "sources",
                        }
                    )

        oldest_feed_post = min((created_at for _, created_at in feed_posts), default=None)
        hit_rss_cap = (
            len(feed_posts) >= REDDIT_SUBREDDIT_RSS_LIMIT
            and oldest_feed_post is not None
            and oldest_feed_post > window_start
        )
        hit_post_cap = bool(max_posts_per_source and len(window_post_entries) > len(selected_post_entries))
        comments_requested = len(posts) * max_comments_per_thread if include_comments and max_comments_per_thread else 0
        coverage = self._build_source_coverage(
            source=source,
            window_start=window_start,
            window_end=window_end,
            scrape_mode=scrape_mode,
            posts_requested=max_posts_per_source,
            comments_requested=comments_requested,
            items=items,
            hit_post_cap=hit_post_cap,
            hit_rss_cap=hit_rss_cap,
            canonical_trend_run=canonical_trend_run,
            collection_completed=True,
        )
        return items, coverage

    def _collect_scraped_thread(
        self,
        *,
        session: requests.Session,
        source: dict[str, Any],
        window_start: datetime,
        window_end: datetime,
        max_comments_per_thread: int,
        include_comments: bool,
        scrape_mode: str = "posts_only",
        canonical_trend_run: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entries = self._fetch_rss_entries(session, self._thread_rss_url(str(source["url"])))
        items: list[dict[str, Any]] = []
        for entry in entries:
            entry_id = self._entry_id(entry)
            created_at = self._entry_timestamp(entry)
            if entry_id.startswith("t3_") and window_start <= created_at <= window_end:
                items.append(self._rss_entry_to_item(entry, source, item_type="post"))

        if include_comments and max_comments_per_thread:
            comments = [
                self._rss_entry_to_item(entry, source, item_type="comment")
                for entry in entries
                if self._entry_id(entry).startswith("t1_")
                and window_start <= self._entry_timestamp(entry) <= window_end
            ]
            comments.sort(key=lambda row: row["created_time"], reverse=True)
            items.extend(comments[:max_comments_per_thread])

        comments_requested = max_comments_per_thread if include_comments and max_comments_per_thread else 0
        coverage = self._build_source_coverage(
            source=source,
            window_start=window_start,
            window_end=window_end,
            scrape_mode=scrape_mode,
            posts_requested=1,
            comments_requested=comments_requested,
            items=items,
            hit_rss_cap=len(entries) >= 100 and include_comments,
            canonical_trend_run=canonical_trend_run,
            collection_completed=True,
        )
        return items, coverage

    def _collect_scraped_thread_comments(
        self,
        *,
        session: requests.Session,
        source: dict[str, Any],
        thread_url: str,
        window_start: datetime,
        window_end: datetime,
        max_comments_per_thread: int,
    ) -> list[dict[str, Any]]:
        comments = [
            self._rss_entry_to_item(entry, source, item_type="comment")
            for entry in self._fetch_rss_entries(session, self._thread_rss_url(thread_url))
            if self._entry_id(entry).startswith("t1_")
            and window_start <= self._entry_timestamp(entry) <= window_end
        ]
        comments.sort(key=lambda row: row["created_time"], reverse=True)
        return comments[:max_comments_per_thread]

    def _fetch_rss_entries(self, session: requests.Session, url: str) -> list[ET.Element]:
        for attempt in range(1, REDDIT_RATE_LIMIT_RETRY_ATTEMPTS + 1):
            self._wait_for_request_window()
            response = session.get(url, timeout=20)
            wait_seconds = self._update_request_window(response)

            if response.status_code == 429:
                if attempt >= REDDIT_RATE_LIMIT_RETRY_ATTEMPTS:
                    retry_hint = (
                        f" Reddit asked this client to wait about {ceil(wait_seconds)} seconds before the next RSS request."
                        if wait_seconds > 0
                        else ""
                    )
                    raise RuntimeError(
                        "Reddit rate-limited the public scraper."
                        f"{retry_hint} This is a temporary Reddit cooldown, not an app weekly quota."
                    )
                continue

            if response.status_code == 403:
                raise RuntimeError(
                    "Reddit blocked the public scraper request. Try again later or lower the run size."
                )

            response.raise_for_status()
            root = ET.fromstring(response.text)
            return root.findall("atom:entry", ATOM_NS)

        raise RuntimeError("Reddit rate-limited the public scraper. Please retry later.")

    def _wait_for_request_window(self) -> None:
        wait_seconds = max(REDDIT_REQUEST_DELAY_SECONDS, self._next_request_not_before - time.monotonic(), 0.0)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _update_request_window(self, response: requests.Response) -> float:
        retry_after_seconds = self._parse_float_header(response.headers.get("retry-after"))
        remaining = self._parse_float_header(response.headers.get("x-ratelimit-remaining"), default=None)
        reset_seconds = self._parse_float_header(response.headers.get("x-ratelimit-reset"))
        wait_seconds = max(retry_after_seconds, 0.0)

        if wait_seconds <= 0 and (
            response.status_code == 429 or (remaining is not None and remaining <= 0 and reset_seconds > 0)
        ):
            wait_seconds = reset_seconds

        if wait_seconds > 0:
            self._next_request_not_before = max(
                self._next_request_not_before,
                time.monotonic() + wait_seconds + REDDIT_RATE_LIMIT_BUFFER_SECONDS,
            )
        return wait_seconds

    def _parse_float_header(self, value: str | None, *, default: float | None = 0.0) -> float | None:
        if value is None:
            return default
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _rss_entry_to_item(self, entry: ET.Element, source: dict[str, Any], *, item_type: str) -> dict[str, Any]:
        entry_id = self._entry_id(entry)
        bare_id = entry_id.split("_", 1)[1] if "_" in entry_id else entry_id
        link = self._entry_link(entry)
        subreddit, thread_id, comment_id = self._parse_reddit_permalink(link)
        title = self._entry_text(entry, "title")
        body_text = self._html_to_text(self._entry_text(entry, "content"))
        created_time = self._entry_timestamp(entry).isoformat()
        thread_title = title
        if item_type == "comment" and " on " in title:
            thread_title = title.split(" on ", 1)[1]

        return self._normalize_item(
            {
                "id": bare_id,
                "item_id": bare_id,
                "item_type": item_type,
                "subreddit": subreddit,
                "thread_id": thread_id or bare_id,
                "thread_title": thread_title,
                "body_text": body_text,
                "author": self._normalize_author(self._entry_author(entry)),
                "score": 0,
                "created_time": created_time,
                "permalink": link,
                "thread_url": canonicalize_thread_url(link),
                "comment_id": comment_id if item_type == "comment" else None,
                "parent_id": None,
                "comment_depth": 0 if item_type == "post" else 1,
                "source_id": int(source["source_id"]),
                "source_type": str(source["source_type"]),
                "source_name": str(source["display_name"]),
            }
        )

    def _entry_id(self, entry: ET.Element) -> str:
        return self._entry_text(entry, "id")

    def _entry_link(self, entry: ET.Element) -> str:
        link = entry.find("atom:link", ATOM_NS)
        return str(link.attrib.get("href", "")) if link is not None else ""

    def _entry_author(self, entry: ET.Element) -> str:
        author = entry.find("atom:author/atom:name", ATOM_NS)
        if author is None:
            return "[deleted]"
        return author.text or "[deleted]"

    def _entry_timestamp(self, entry: ET.Element) -> datetime:
        raw_value = self._entry_text(entry, "published") or self._entry_text(entry, "updated")
        return _parse_iso_datetime(raw_value)

    def _entry_text(self, entry: ET.Element, tag: str) -> str:
        element = entry.find(f"atom:{tag}", ATOM_NS)
        if element is None:
            return ""
        return element.text or ""

    def _thread_rss_url(self, thread_url: str) -> str:
        return canonicalize_thread_url(thread_url).rstrip("/") + "/.rss?limit=100"

    def _parse_reddit_permalink(self, value: str) -> tuple[str, str, str | None]:
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[0].lower() == "r" and parts[2].lower() == "comments":
            comment_id = parts[5] if len(parts) >= 6 else None
            return parts[1], parts[3], comment_id
        return "", "", None

    def _normalize_author(self, value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("/u/"):
            return cleaned[3:]
        if cleaned.startswith("u/"):
            return cleaned[2:]
        return cleaned or "[deleted]"

    def _html_to_text(self, value: str) -> str:
        text = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
        text = re.sub(r"<br\\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        normalized.setdefault("id", normalized.get("item_id"))
        normalized.setdefault("item_id", normalized["id"])
        normalized.setdefault("item_type", normalized.get("source_type", "post"))
        normalized.setdefault("subreddit", "")
        normalized.setdefault("thread_id", normalized.get("id"))
        normalized.setdefault("thread_title", "")
        normalized.setdefault("body_text", "")
        normalized.setdefault("author", "[deleted]")
        normalized.setdefault("score", 0)
        normalized.setdefault("created_time", normalized.get("created_utc", ""))
        normalized.setdefault("created_utc", normalized["created_time"])
        normalized.setdefault("permalink", "")
        normalized.setdefault("thread_url", self._derive_thread_url(normalized))
        normalized.setdefault("comment_id", normalized["id"] if normalized["item_type"] == "comment" else None)
        normalized.setdefault("parent_id", None)
        normalized.setdefault("comment_depth", 0)
        normalized.setdefault("source_name", normalized.get("subreddit", ""))
        if not normalized.get("id"):
            raise ValueError("Each Reddit item must include an id.")
        return normalized

    def _apply_limits(
        self,
        items: list[dict[str, Any]],
        *,
        max_posts_per_source: int,
        max_comments_per_thread: int,
        include_comments: bool,
    ) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        posts_per_source: dict[int, int] = {}
        comments_per_thread: dict[str, int] = {}

        for item in items:
            if item["item_type"] == "post":
                source_id = int(item["source_id"])
                current = posts_per_source.get(source_id, 0)
                if max_posts_per_source and current >= max_posts_per_source:
                    continue
                posts_per_source[source_id] = current + 1
                kept.append(item)
                continue

            if include_comments and item["item_type"] == "comment":
                thread_id = str(item["thread_id"])
                current = comments_per_thread.get(thread_id, 0)
                if max_comments_per_thread and current >= max_comments_per_thread:
                    continue
                comments_per_thread[thread_id] = current + 1
                kept.append(item)

        kept.sort(key=lambda row: row["created_time"])
        return kept

    def _match_source(self, item: dict[str, Any], selected_sources: list[dict[str, Any]]) -> dict[str, Any] | None:
        thread_url = canonicalize_thread_url(item.get("thread_url") or self._derive_thread_url(item))
        subreddit = normalize_subreddit_name(item.get("subreddit", ""))

        for source in selected_sources:
            if source["source_type"] == "thread_url":
                if source["normalized_value"] == thread_url:
                    return source
        for source in selected_sources:
            if source["source_type"] == "subreddit":
                if source["normalized_value"] == subreddit:
                    return source
        return None

    def _derive_thread_url(self, item: dict[str, Any]) -> str:
        permalink = item.get("permalink", "")
        if not permalink:
            subreddit = normalize_subreddit_name(item.get("subreddit", ""))
            thread_id = item.get("thread_id", item.get("id", ""))
            title_slug = slugify(item.get("thread_title", "")) or "thread"
            return f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}/{title_slug}/" if subreddit else ""
        parsed = urlparse(permalink)
        if parsed.scheme and parsed.netloc:
            path = parsed.path
        else:
            path = permalink
        match = THREAD_URL_PATTERN.match(path)
        if not match:
            return permalink
        subreddit = match.group(1)
        thread_id = match.group(2)
        title_slug = slugify(item.get("thread_title", "")) or "thread"
        return f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}/{title_slug}/"

    def _parse_timestamp(self, value: str) -> datetime:
        return _parse_iso_datetime(value)

    def _build_source_coverage(
        self,
        *,
        source: dict[str, Any],
        window_start: datetime,
        window_end: datetime,
        scrape_mode: str,
        posts_requested: int,
        comments_requested: int,
        items: list[dict[str, Any]],
        hit_post_cap: bool = False,
        hit_rss_cap: bool = False,
        rate_limited: bool = False,
        collection_error: str = "",
        collection_completed: bool = True,
        canonical_trend_run: bool = False,
    ) -> dict[str, Any]:
        posts_returned = sum(1 for item in items if item.get("item_type") == "post")
        comments_returned = sum(1 for item in items if item.get("item_type") == "comment")
        threads_returned = len({str(item.get("thread_id", "")) for item in items if item.get("thread_id")})
        authors_returned = len({str(item.get("author", "")) for item in items if item.get("author")})
        timestamps = [
            _parse_iso_datetime(str(item.get("created_time", "")))
            for item in items
            if str(item.get("created_time", "")).strip()
        ]
        oldest_seen_at = min(timestamps).isoformat() if timestamps else ""
        newest_seen_at = max(timestamps).isoformat() if timestamps else ""
        reliability, label, reasons = calculate_coverage_reliability(
            source_type=str(source.get("source_type", "")),
            scrape_mode=scrape_mode,
            window_start=window_start,
            window_end=window_end,
            oldest_seen_at=oldest_seen_at,
            posts_returned=posts_returned,
            hit_post_cap=hit_post_cap,
            hit_rss_cap=hit_rss_cap,
            rate_limited=rate_limited,
            collection_completed=collection_completed,
            collection_error=collection_error,
            canonical_trend_run=canonical_trend_run,
        )
        return {
            "source_id": int(source["source_id"]) if source.get("source_id") is not None else None,
            "source_name": str(source.get("display_name", source.get("source_name", ""))),
            "source_type": str(source.get("source_type", "")),
            "scrape_mode": scrape_mode,
            "lookback_hours": round(max((window_end - window_start).total_seconds() / 3600, 0), 2),
            "posts_requested": int(posts_requested or 0),
            "posts_returned": int(posts_returned),
            "comments_requested": int(comments_requested or 0),
            "comments_returned": int(comments_returned),
            "threads_returned": int(threads_returned),
            "authors_returned": int(authors_returned),
            "oldest_seen_at": oldest_seen_at,
            "newest_seen_at": newest_seen_at,
            "hit_post_cap": bool(hit_post_cap),
            "hit_rss_cap": bool(hit_rss_cap),
            "rate_limited": bool(rate_limited),
            "collection_error": str(collection_error or ""),
            "collection_completed": bool(collection_completed and not collection_error),
            "canonical_trend_run": bool(canonical_trend_run),
            "coverage_reliability": reliability,
            "reliability_label": label,
            "reliability_reason": reasons,
        }


def parse_source_input(raw_value: str) -> SourceCandidate | None:
    candidate = (raw_value or "").strip()
    if not candidate:
        return None

    if candidate.startswith("http://") or candidate.startswith("https://"):
        parsed = urlparse(candidate)
        thread_match = THREAD_URL_PATTERN.match(parsed.path)
        if thread_match:
            canonical = canonicalize_thread_url(candidate)
            title_bits = [bit for bit in parsed.path.split("/") if bit]
            display_tail = title_bits[4] if len(title_bits) > 4 else thread_match.group(2)
            return SourceCandidate(
                source_key=f"thread:{canonical}",
                source_type="thread_url",
                display_name=f"Thread: {display_tail.replace('-', ' ')}",
                normalized_value=canonical,
                url=canonical,
            )

        subreddit_name = subreddit_name_from_url(candidate)
        if subreddit_name:
            return build_subreddit_source(subreddit_name)
        return None

    subreddit_match = SUBREDDIT_PATTERN.match(candidate.strip("/"))
    if subreddit_match:
        return build_subreddit_source(subreddit_match.group(1))
    return None


def build_subreddit_source(subreddit_name: str) -> SourceCandidate:
    normalized = normalize_subreddit_name(subreddit_name)
    display_name = f"r/{subreddit_name.strip().strip('/')}".replace("//", "/")
    return SourceCandidate(
        source_key=f"subreddit:{normalized}",
        source_type="subreddit",
        display_name=display_name if display_name.startswith("r/") else f"r/{display_name}",
        normalized_value=normalized,
        url=f"https://www.reddit.com/r/{normalized}/",
    )


def subreddit_name_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0].lower() == "r":
        return parts[1]
    return None


def normalize_subreddit_name(value: str) -> str:
    cleaned = value.strip().strip("/")
    if cleaned.lower().startswith("r/"):
        cleaned = cleaned[2:]
    return cleaned.lower()


def canonicalize_thread_url(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path if parsed.scheme else value
    match = THREAD_URL_PATTERN.match(path)
    if not match:
        return value.rstrip("/") + "/"
    subreddit = match.group(1)
    thread_id = match.group(2)
    title_bits = [bit for bit in path.split("/") if bit]
    title_slug = title_bits[4] if len(title_bits) > 4 else "thread"
    return f"https://www.reddit.com/r/{subreddit}/comments/{thread_id}/{title_slug}/"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "thread"


def calculate_coverage_reliability(
    *,
    source_type: str,
    scrape_mode: str,
    window_start: datetime,
    window_end: datetime,
    oldest_seen_at: str,
    posts_returned: int,
    hit_post_cap: bool,
    hit_rss_cap: bool,
    rate_limited: bool,
    collection_completed: bool,
    collection_error: str,
    canonical_trend_run: bool,
) -> tuple[float, str, list[str]]:
    reasons: list[str] = []
    if not collection_completed or collection_error:
        return 0.0, "Low", [collection_error or "Collection did not complete."]

    score = 1.0
    if not canonical_trend_run:
        score *= 0.65
        reasons.append("Discovery or comment-inclusive run; useful for evidence, weaker for acceleration.")
    if source_type == "thread_url":
        score *= 0.7
        reasons.append("Single-thread source; not a broad subreddit sample.")
    if rate_limited:
        score *= 0.2
        reasons.append("Reddit rate limit affected collection.")
    if hit_rss_cap:
        score *= 0.45
        reasons.append("RSS cap was hit before the requested window was fully visible.")
    if hit_post_cap:
        score *= 0.75
        reasons.append("Configured post cap stopped collection before the feed was exhausted.")
    if posts_returned == 0:
        score = 0.0
        reasons.append("No posts returned for this source.")
    elif posts_returned < 10:
        score *= 0.55
        reasons.append("Very small post sample.")
    elif posts_returned < 30:
        score *= 0.78
        reasons.append("Post sample is below the preferred 30-post trend minimum.")

    if oldest_seen_at:
        try:
            oldest = _parse_iso_datetime(oldest_seen_at)
            window_seconds = max((window_end - window_start).total_seconds(), 1)
            missed_fraction = max((oldest - window_start).total_seconds(), 0) / window_seconds
            if missed_fraction > 0.35:
                score *= 0.75
                reasons.append("Oldest returned post does not cover much of the requested window.")
        except (TypeError, ValueError):
            reasons.append("Could not parse oldest returned timestamp.")

    score = round(max(min(score, 1.0), 0.0), 3)
    if score >= 0.8:
        label = "High"
    elif score >= 0.5:
        label = "Medium"
    else:
        label = "Low"
    if not reasons:
        reasons.append("Comparable posts-only source sample.")
    return score, label, reasons


def _parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC)
