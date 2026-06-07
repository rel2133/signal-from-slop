from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SUBREDDIT_PATTERN = re.compile(r"^(?:/?r/)?([A-Za-z0-9_]+)$")
THREAD_URL_PATTERN = re.compile(r"^/r/([A-Za-z0-9_]+)/comments/([A-Za-z0-9]+)/?.*$")


@dataclass(frozen=True)
class SourceCandidate:
    source_key: str
    source_type: str
    display_name: str
    normalized_value: str
    url: str


class RedditClient:
    """Fake Reddit data access layer with source-aware filtering."""

    def __init__(self, fake_data_path: str | Path) -> None:
        self.fake_data_path = Path(fake_data_path)

    def load_fake_data(self) -> list[dict[str, Any]]:
        payload = json.loads(self.fake_data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Fake Reddit data must be a JSON list.")
        return [self._normalize_item(item) for item in payload]

    def latest_available_timestamp(self) -> datetime:
        items = self.load_fake_data()
        return max(self._parse_timestamp(item["created_time"]) for item in items)

    def collect_fake_items(
        self,
        *,
        selected_sources: list[dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
        max_posts_per_source: int,
        max_comments_per_thread: int,
        include_comments: bool,
    ) -> list[dict[str, Any]]:
        items = self.load_fake_data()
        matched_items: list[dict[str, Any]] = []
        for item in items:
            matched_source = self._match_source(item, selected_sources)
            if not matched_source:
                continue
            item_time = self._parse_timestamp(item["created_time"])
            if item_time < window_start or item_time > window_end:
                continue
            if not include_comments and item["item_type"] == "comment":
                continue
            enriched = dict(item)
            enriched["source_id"] = int(matched_source["source_id"])
            enriched["source_type"] = str(matched_source["source_type"])
            enriched["source_name"] = str(matched_source["display_name"])
            matched_items.append(enriched)

        matched_items.sort(key=lambda row: row["created_time"], reverse=True)
        return self._apply_limits(
            matched_items,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
        )

    def fetch_live(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "Live Reddit collection is not implemented in this MVP. "
            "Use the bundled fake dataset for now."
        )

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
