from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import urlparse

import requests


SUBREDDIT_PATTERN = re.compile(r"^(?:/?r/)?([A-Za-z0-9_]+)$")
THREAD_URL_PATTERN = re.compile(r"^/r/([A-Za-z0-9_]+)/comments/([A-Za-z0-9]+)/?.*$")
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_REDDIT_USER_AGENT = "script:signal-from-the-slop:0.1"
REDDIT_REQUEST_DELAY_SECONDS = 0.25
REDDIT_SUBREDDIT_RSS_LIMIT = 100


@dataclass(frozen=True)
class SourceCandidate:
    source_key: str
    source_type: str
    display_name: str
    normalized_value: str
    url: str


class RedditClient:
    """Reddit RSS collection layer with source-aware filtering."""

    def collect_live_items(
        self,
        *,
        selected_sources: list[dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
        max_posts_per_source: int,
        max_comments_per_thread: int,
        include_comments: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        session = self._build_scraper_session()
        deduped_items: dict[str, dict[str, Any]] = {}

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
            if source["source_type"] == "thread_url":
                items = self._collect_scraped_thread(
                    session=session,
                    source=source,
                    window_start=window_start,
                    window_end=window_end,
                    max_comments_per_thread=max_comments_per_thread,
                    include_comments=include_comments,
                )
            elif source["source_type"] == "subreddit":
                items = self._collect_scraped_subreddit(
                    session=session,
                    source=source,
                    window_start=window_start,
                    window_end=window_end,
                    max_posts_per_source=max_posts_per_source,
                    max_comments_per_thread=max_comments_per_thread,
                    include_comments=include_comments,
                    progress_callback=progress_callback,
                    checkpoint=checkpoint,
                    source_index=source_index,
                    total_sources=total_sources,
                )
            else:
                items = []

            for item in items:
                deduped_items.setdefault(item["item_id"], item)
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
    ) -> list[dict[str, Any]]:
        subreddit = normalize_subreddit_name(str(source["normalized_value"]))
        feed_url = f"https://www.reddit.com/r/{subreddit}/new/.rss?limit={REDDIT_SUBREDDIT_RSS_LIMIT}"
        entries = self._fetch_rss_entries(session, feed_url)
        items: list[dict[str, Any]] = []
        posts: list[dict[str, Any]] = []

        for entry in entries:
            if not self._entry_id(entry).startswith("t3_"):
                continue

            created_at = self._entry_timestamp(entry)
            if created_at > window_end:
                continue
            if created_at < window_start:
                break

            post = self._rss_entry_to_item(entry, source, item_type="post")
            posts.append(post)
            items.append(post)
            if max_posts_per_source and len(posts) >= max_posts_per_source:
                break

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
                items.extend(
                    self._collect_scraped_thread_comments(
                        session=session,
                        source=source,
                        thread_url=str(post["thread_url"]),
                        window_start=window_start,
                        window_end=window_end,
                        max_comments_per_thread=max_comments_per_thread,
                    )
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "message": f"Fetched comments {post_index}/{len(posts)} for {source['display_name']}",
                            "current": source_index - 1,
                            "total": total_sources,
                            "unit": "sources",
                        }
                    )

        return items

    def _collect_scraped_thread(
        self,
        *,
        session: requests.Session,
        source: dict[str, Any],
        window_start: datetime,
        window_end: datetime,
        max_comments_per_thread: int,
        include_comments: bool,
    ) -> list[dict[str, Any]]:
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

        return items

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
        time.sleep(REDDIT_REQUEST_DELAY_SECONDS)
        response = session.get(url, timeout=20)
        if response.status_code == 429:
            raise RuntimeError("Reddit rate-limited the public scraper. Wait a few minutes and try a smaller run.")
        if response.status_code == 403:
            raise RuntimeError(
                "Reddit blocked the public scraper request. Try again later or lower the run size."
            )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        return root.findall("atom:entry", ATOM_NS)

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


def _parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC)
