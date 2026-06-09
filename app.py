from __future__ import annotations

import json
import logging
import os
from html import escape
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from signal_from_the_slop.analytics import (
    apply_ticker_flags_to_mentions,
    author_hash,
    build_classification_record,
    build_mention_records,
    build_run_summary,
    build_ticker_summaries,
    build_time_bucket_summary,
    parse_created_time,
)
from signal_from_the_slop.database import (
    add_source,
    complete_analysis_run,
    create_analysis_run,
    db_overview,
    delete_sources,
    fail_analysis_run,
    init_db,
    latest_analysis_run_id,
    load_analysis_runs,
    load_full_results_records,
    load_run_classification_quality,
    load_historical_ticker_summaries,
    load_run_mentions,
    load_run_summary,
    load_run_source_activity,
    load_run_ticker_summaries,
    load_run_time_buckets,
    load_sources,
    replace_run_classifications,
    replace_run_mentions,
    replace_run_ticker_summaries,
    replace_run_time_buckets,
    update_sources,
    upsert_reddit_items,
)
from signal_from_the_slop.ollama_classifier import OllamaClassifier
from signal_from_the_slop.reddit_client import RedditClient, build_subreddit_source, parse_source_input
from signal_from_the_slop.ticker_extractor import TickerExtractor


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
load_dotenv()

ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_ROOT = Path(os.getenv("APP_STORAGE_DIR", "")).expanduser() if os.getenv("APP_STORAGE_DIR") else ROOT
DEFAULT_TICKER_PATH = ROOT / "data" / "tickers.csv"
DEFAULT_SOURCES_PATH = ROOT / "data" / "default_sources.json"
SCHEMA_PATH = ROOT / "schema.sql"
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TIME_WINDOW_OPTIONS = {
    "Last 24 hours": 1,
    "Last 3 days": 3,
    "Last 7 days": 7,
    "Last 14 days": 14,
    "Last 30 days": 30,
    "Since last completed run": "since_last_run",
    "Custom date range": None,
}
PAGE_STEPS = [
    {"page": "Sources", "label": "Sources", "step": "01", "context": "Reddit pages"},
    {"page": "Run Analysis", "label": "Scrape", "step": "02", "context": "Collect and classify"},
    {"page": "Results Dashboard", "label": "Results", "step": "03", "context": "Review mentions"},
    {"page": "Ticker Trends", "label": "Trends", "step": "04", "context": "Compare scrapes"},
    {"page": "Export Data", "label": "Export", "step": "05", "context": "Download data"},
    {"page": "Settings", "label": "Settings", "step": "06", "context": "Storage and model"},
]
PAGES = [step["page"] for step in PAGE_STEPS]
PAGE_LABELS = {step["page"]: f"{step['step']} {step['label']}" for step in PAGE_STEPS}
PAGE_CONTEXT = {step["page"]: step["context"] for step in PAGE_STEPS}

LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
FILTER_PRESETS = {
    "Balanced review": {
        "sentiment": ["bullish", "bearish", "neutral", "irrelevant"],
        "min_confidence": 0.0,
        "min_depth": 0,
        "min_alpha": 0,
        "max_hype": 10,
        "hide_hype": False,
        "only_deeper": False,
        "only_low_signal": False,
        "only_new": False,
    },
    "High signal": {
        "sentiment": ["bullish", "bearish", "neutral"],
        "min_confidence": 0.45,
        "min_depth": 4,
        "min_alpha": 65,
        "max_hype": 7,
        "hide_hype": True,
        "only_deeper": False,
        "only_low_signal": False,
        "only_new": False,
    },
    "Low-hype leads": {
        "sentiment": ["bullish", "bearish", "neutral"],
        "min_confidence": 0.35,
        "min_depth": 0,
        "min_alpha": 55,
        "max_hype": 4,
        "hide_hype": True,
        "only_deeper": False,
        "only_low_signal": True,
        "only_new": False,
    },
    "New tickers": {
        "sentiment": ["bullish", "bearish", "neutral", "irrelevant"],
        "min_confidence": 0.0,
        "min_depth": 0,
        "min_alpha": 0,
        "max_hype": 10,
        "hide_hype": False,
        "only_deeper": False,
        "only_low_signal": False,
        "only_new": True,
    },
    "Needs verification": {
        "sentiment": ["bullish", "bearish", "neutral"],
        "min_confidence": 0.0,
        "min_depth": 0,
        "min_alpha": 0,
        "max_hype": 10,
        "hide_hype": False,
        "only_deeper": True,
        "only_low_signal": False,
        "only_new": False,
    },
}
DEFAULT_UI_PREFERENCES = {
    "watchlist": [],
    "evidence_checks": {},
}
METRIC_GUIDE = [
    {
        "metric": "Mentions",
        "plain_english": "How many item/ticker rows matched this ticker in the selected run or bucket.",
        "how_it_is_determined": "Each classified Reddit post/comment can produce one row per detected ticker.",
    },
    {
        "metric": "Thread mentions",
        "plain_english": "How many matching ticker mentions came from the same Reddit thread.",
        "how_it_is_determined": "Results are grouped by thread so multi-comment discussions do not look like unrelated posts.",
    },
    {
        "metric": "Confidence",
        "plain_english": "How confident the classifier was about the sentiment/ticker interpretation.",
        "how_it_is_determined": "Returned by the model or fallback classifier as a 0 to 1 score.",
    },
    {
        "metric": "Alpha signal",
        "plain_english": "A quality score for potentially useful, researchable information.",
        "how_it_is_determined": "Combines evidence quality, specificity, depth, confidence, and Reddit context in the scoring module.",
    },
    {
        "metric": "Max alpha",
        "plain_english": "The strongest single alpha signal found inside a grouped thread.",
        "how_it_is_determined": "Highest alpha_signal_score among the mentions in that thread group.",
    },
    {
        "metric": "Hype",
        "plain_english": "How meme-like, repetitive, or promotional the text looks.",
        "how_it_is_determined": "Classifier score plus repetition/promotional language cues. Lower is generally cleaner.",
    },
    {
        "metric": "Hype-adjusted signal",
        "plain_english": "Signal after penalizing hype and repetition.",
        "how_it_is_determined": "Alpha signal minus hype/repetition penalties, capped to 0 to 100.",
    },
    {
        "metric": "Acceleration",
        "plain_english": "Whether mentions are rising compared with the previous bucket or scrape.",
        "how_it_is_determined": "Uses mention change, growth rate, unique author/thread growth, and hype penalties.",
    },
    {
        "metric": "Emerging ticker score",
        "plain_english": "A lead-ranking score for tickers that may deserve follow-up.",
        "how_it_is_determined": "Combines acceleration, source diversity, alpha score, evidence quality, low-volume bonus, and mega-cap penalty.",
    },
    {
        "metric": "Source diversity",
        "plain_english": "Whether discussion is coming from more than one source/subreddit.",
        "how_it_is_determined": "Scores unique source/subreddit count on a 0 to 10 scale.",
    },
    {
        "metric": "Controversy",
        "plain_english": "How split bullish and bearish mentions are.",
        "how_it_is_determined": "Higher when bullish and bearish counts are balanced; zero when all decisive mentions are one-sided.",
    },
    {
        "metric": "Trust",
        "plain_english": "A quick label for how much attention to give the row before reading it.",
        "how_it_is_determined": "Strong, Standard, Review, or Fallback based on confidence, evidence quality, hype, and classifier mode.",
    },
]


def resolve_config_path(env_name: str, default_name: str, *, base_dir: Path) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return base_dir / default_name
    configured = Path(raw_value).expanduser()
    return configured if configured.is_absolute() else base_dir / configured


def validate_storage_root(storage_root: Path) -> str | None:
    resolved = storage_root.expanduser()
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        volume_root = Path("/", parts[1], parts[2])
        if not volume_root.exists() or not volume_root.is_mount():
            return (
                f"Configured APP_STORAGE_DIR is on `{volume_root}`, but that external volume is not mounted. "
                "Plug in the SSD or update `.env` before running the app."
            )
    return None


DEFAULT_DB_PATH = resolve_config_path("SQLITE_PATH", "signal_from_the_slop.db", base_dir=DEFAULT_STORAGE_ROOT)
DEFAULT_ARTIFACTS_DIR = resolve_config_path("ARTIFACTS_DIR", "artifacts", base_dir=DEFAULT_STORAGE_ROOT)
DEFAULT_UI_PREFS_PATH = DEFAULT_ARTIFACTS_DIR / "ui_preferences.json"
STORAGE_ROOT_ERROR = validate_storage_root(DEFAULT_STORAGE_ROOT)


alt.data_transformers.disable_max_rows()


st.set_page_config(page_title="Signal from the Slop", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1480px;
            padding-top: 1.15rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px;
            padding: 0.45rem 0.6rem;
        }
        .app-shell-header {
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 0.75rem;
            padding-bottom: 0.75rem;
        }
        .app-shell-header h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0;
            letter-spacing: 0;
        }
        .app-shell-header p {
            margin: 0.25rem 0 0;
            opacity: 0.72;
        }
        .workflow-kicker {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            margin-top: 0.35rem;
            opacity: 0.72;
            text-transform: uppercase;
        }
        .insight-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            margin: 0.35rem 0 1rem;
        }
        .insight-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: rgba(128, 128, 128, 0.06);
        }
        .insight-label {
            font-size: 0.78rem;
            font-weight: 700;
            opacity: 0.7;
            text-transform: uppercase;
        }
        .insight-value {
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.25;
            margin-top: 0.3rem;
        }
        .insight-note {
            font-size: 0.86rem;
            margin-top: 0.25rem;
            opacity: 0.76;
        }
        .status-chip {
            border: 1px solid rgba(128, 128, 128, 0.26);
            border-radius: 999px;
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0 0.25rem 0.25rem 0;
            padding: 0.16rem 0.48rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="app-shell-header">
        <h1>Signal from the Slop</h1>
        <p>Research dashboard for Reddit stock discussion triage. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
if STORAGE_ROOT_ERROR:
    st.error(STORAGE_ROOT_ERROR)
    st.stop()


def discover_ollama_models(ollama_url: str) -> tuple[list[str], str | None]:
    try:
        return OllamaClassifier(model="", endpoint=ollama_url).list_available_models(), None
    except requests.RequestException as exc:
        return [], str(exc)


def is_localhost_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return (parsed.hostname or "").lower() in LOCALHOST_HOSTS


def normalize_ticker(value: str) -> str:
    return value.strip().upper().replace("$", "")


def normalize_ui_preferences(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    watchlist = sorted(
        {
            normalize_ticker(str(ticker))
            for ticker in value.get("watchlist", [])
            if normalize_ticker(str(ticker))
        }
    )
    evidence_checks = value.get("evidence_checks", {})
    if not isinstance(evidence_checks, dict):
        evidence_checks = {}
    return {
        "watchlist": watchlist,
        "evidence_checks": {str(key): bool(val) for key, val in evidence_checks.items()},
    }


def load_ui_preferences() -> dict[str, Any]:
    if "ui_preferences" in st.session_state:
        return normalize_ui_preferences(st.session_state["ui_preferences"])
    if DEFAULT_UI_PREFS_PATH.exists():
        try:
            prefs = json.loads(DEFAULT_UI_PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prefs = DEFAULT_UI_PREFERENCES
    else:
        prefs = DEFAULT_UI_PREFERENCES
    st.session_state["ui_preferences"] = normalize_ui_preferences(prefs)
    return st.session_state["ui_preferences"]


def save_ui_preferences(prefs: dict[str, Any]) -> None:
    normalized = normalize_ui_preferences(prefs)
    DEFAULT_UI_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_UI_PREFS_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
    st.session_state["ui_preferences"] = normalized


def update_watchlist(tickers: list[str]) -> None:
    prefs = load_ui_preferences()
    prefs["watchlist"] = sorted({normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)})
    save_ui_preferences(prefs)


def get_watchlist() -> list[str]:
    return load_ui_preferences().get("watchlist", [])


def format_delta(value: Any, *, decimals: int = 0) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if decimals:
        return f"{numeric:+.{decimals}f}"
    return f"{int(round(numeric)):+d}"


def format_metric_value(value: Any, *, decimals: int = 0) -> str:
    if value in (None, ""):
        return "None"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals:
        return f"{numeric:.{decimals}f}"
    return f"{int(round(numeric))}"


def render_insight_cards(cards: list[dict[str, str]]) -> None:
    if not cards:
        return
    card_html = []
    for card in cards:
        label = escape(card.get("label", ""))
        value = escape(card.get("value", ""))
        note = escape(card.get("note", ""))
        card_html.append(
            f"""
            <div class="insight-card">
                <div class="insight-label">{label}</div>
                <div class="insight-value">{value}</div>
                <div class="insight-note">{note}</div>
            </div>
            """
        )
    st.markdown(f'<div class="insight-grid">{"".join(card_html)}</div>', unsafe_allow_html=True)


def render_status_chips(labels: list[str]) -> None:
    chips = "".join(f'<span class="status-chip">{escape(label)}</span>' for label in labels if label)
    if chips:
        st.markdown(chips, unsafe_allow_html=True)


def render_copyable_ticker_chips(
    tickers: list[str],
    *,
    label: str = "Tickers",
    help_text: str = "Click a ticker to copy it.",
    max_items: int = 24,
) -> None:
    cleaned = [normalize_ticker(ticker) for ticker in tickers if normalize_ticker(ticker)]
    cleaned = list(dict.fromkeys(cleaned))[:max_items]
    if not cleaned:
        st.caption("No tickers to show.")
        return

    payload = json.dumps(cleaned)
    component_height = min(260, 62 + ((len(cleaned) + 7) // 8) * 34)
    html = f"""
    <div class="ticker-copy-shell">
      <div class="ticker-copy-head">
        <strong>{escape(label)}</strong>
        <span id="ticker-copy-status">{escape(help_text)}</span>
      </div>
      <div class="ticker-copy-grid" id="ticker-copy-grid"></div>
    </div>
    <style>
      .ticker-copy-shell {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 0.15rem 0 0.5rem;
      }}
      .ticker-copy-head {{
        align-items: center;
        display: flex;
        gap: 0.75rem;
        justify-content: space-between;
        margin-bottom: 0.45rem;
      }}
      .ticker-copy-head span {{
        color: rgba(127, 127, 127, 0.92);
        font-size: 0.82rem;
      }}
      .ticker-copy-grid {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
      }}
      .ticker-copy-button {{
        appearance: none;
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.30);
        border-radius: 999px;
        color: inherit;
        cursor: pointer;
        font: 700 0.86rem -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 0.34rem 0.62rem;
        transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
      }}
      .ticker-copy-button:hover {{
        background: rgba(57, 132, 255, 0.16);
        border-color: rgba(57, 132, 255, 0.55);
        transform: translateY(-1px);
      }}
      .ticker-copy-button.copied {{
        background: rgba(52, 199, 89, 0.18);
        border-color: rgba(52, 199, 89, 0.70);
      }}
    </style>
    <script>
      const tickers = {payload};
      const grid = document.getElementById("ticker-copy-grid");
      const status = document.getElementById("ticker-copy-status");
      tickers.forEach((ticker) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ticker-copy-button";
        button.textContent = ticker;
        button.title = `Copy ${{ticker}}`;
        button.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(ticker);
            status.textContent = `Copied ${{ticker}}`;
            button.classList.add("copied");
            setTimeout(() => button.classList.remove("copied"), 900);
          }} catch (error) {{
            status.textContent = `Select and copy ${{ticker}}`;
          }}
        }});
        grid.appendChild(button);
      }});
    </script>
    """
    components.html(html, height=component_height, scrolling=False)


def render_metric_guide(title: str = "How to read these metrics", *, expanded: bool = False) -> None:
    with st.expander(title, expanded=expanded):
        st.dataframe(
            pd.DataFrame(METRIC_GUIDE),
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric": st.column_config.TextColumn("Metric", width="medium"),
                "plain_english": st.column_config.TextColumn("Plain English", width="large"),
                "how_it_is_determined": st.column_config.TextColumn("How it is determined", width="large"),
            },
        )


def render_all_tickers_panel(summary_df: pd.DataFrame, mentions_df: pd.DataFrame) -> None:
    if summary_df.empty and mentions_df.empty:
        return
    tickers = []
    if not summary_df.empty:
        tickers.extend(summary_df["ticker"].dropna().astype(str).tolist())
    if not mentions_df.empty:
        tickers.extend(mentions_df["ticker"].dropna().astype(str).tolist())
    tickers = [ticker for ticker in dict.fromkeys(tickers) if ticker != "UNKNOWN"]
    if not tickers:
        return

    with st.expander(f"All mentioned tickers ({len(tickers)})", expanded=True):
        render_copyable_ticker_chips(
            tickers,
            label="All mentioned tickers",
            help_text="Click any ticker to copy it.",
            max_items=80,
        )
        if not summary_df.empty:
            compact_columns = [
                "ticker",
                "company_name",
                "total_mentions",
                "acceleration_score",
                "emerging_ticker_score",
                "average_alpha_signal_score",
                "average_hype_score",
            ]
            compact = summary_df[[column for column in compact_columns if column in summary_df.columns]].copy()
            st.dataframe(
                compact,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "total_mentions": st.column_config.NumberColumn("Mentions", format="%d"),
                    "acceleration_score": st.column_config.NumberColumn("Acceleration", format="%.1f"),
                    "emerging_ticker_score": st.column_config.NumberColumn("Emerging", format="%.1f"),
                    "average_alpha_signal_score": st.column_config.NumberColumn("Alpha", format="%.1f"),
                    "average_hype_score": st.column_config.NumberColumn("Hype", format="%.1f"),
                },
            )


def classifier_trust_label(row: pd.Series) -> str:
    if str(row.get("classifier_mode", "")).lower() == "fallback":
        return "Fallback"
    confidence = float(row.get("confidence", 0) or 0)
    evidence = str(row.get("evidence_quality", "low")).lower()
    hype = float(row.get("hype_score", 0) or 0)
    if confidence >= 0.7 and evidence in {"medium", "high"} and hype <= 6:
        return "Strong"
    if confidence < 0.35 or evidence == "low":
        return "Review"
    return "Standard"


def attach_quality_columns(long_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    working = long_df.copy()
    if not quality_df.empty:
        working = working.merge(quality_df, on="item_id", how="left")
    if "classifier_mode" not in working.columns:
        working["classifier_mode"] = "unknown"
    if "model_name" not in working.columns:
        working["model_name"] = ""
    working["classifier_mode"] = working["classifier_mode"].fillna("unknown")
    working["trust_status"] = working.apply(classifier_trust_label, axis=1)
    return working


def selected_run_source_ids(run_record: dict[str, Any]) -> list[int]:
    return [int(source_id) for source_id in run_record.get("selected_source_ids", [])]


def build_source_health_frame(
    activity_df: pd.DataFrame,
    run_record: dict[str, Any],
    source_lookup: dict[int, str],
) -> pd.DataFrame:
    selected_ids = selected_run_source_ids(run_record)
    records = [
        {
            "source_id": source_id,
            "source_name": source_lookup.get(source_id, f"source:{source_id}"),
        }
        for source_id in selected_ids
    ]
    if not records:
        return pd.DataFrame()

    base = pd.DataFrame(records)
    if activity_df.empty:
        base["items_analyzed"] = 0
        base["ticker_mentions"] = 0
        base["newest_item_time"] = ""
    else:
        activity = activity_df.copy()
        activity["source_id"] = activity["source_id"].fillna(-1).astype(int)
        base = base.merge(
            activity[["source_id", "items_analyzed", "ticker_mentions", "newest_item_time"]],
            on="source_id",
            how="left",
        )
        base["items_analyzed"] = base["items_analyzed"].fillna(0).astype(int)
        base["ticker_mentions"] = base["ticker_mentions"].fillna(0).astype(int)
        base["newest_item_time"] = base["newest_item_time"].fillna("")
    base["status"] = base.apply(
        lambda row: "No items" if int(row["items_analyzed"]) == 0 else (
            "No tickers" if int(row["ticker_mentions"]) == 0 else "Active"
        ),
        axis=1,
    )
    return base.sort_values(["status", "source_name"])


def render_source_health(
    *,
    db_path: Path,
    run_id: str,
    run_record: dict[str, Any],
    source_lookup: dict[int, str],
    expanded: bool = False,
) -> None:
    source_health = build_source_health_frame(
        load_run_source_activity(db_path, run_id),
        run_record,
        source_lookup,
    )
    if source_health.empty:
        return

    active_count = int((source_health["items_analyzed"] > 0).sum())
    ticker_count = int((source_health["ticker_mentions"] > 0).sum())
    total_count = len(source_health)
    with st.expander(
        f"Source coverage: {active_count}/{total_count} returned items, {ticker_count}/{total_count} produced ticker mentions",
        expanded=expanded,
    ):
        st.dataframe(
            source_health[["source_name", "status", "items_analyzed", "ticker_mentions", "newest_item_time"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_name": st.column_config.TextColumn("Source"),
                "items_analyzed": st.column_config.NumberColumn("Items", format="%d"),
                "ticker_mentions": st.column_config.NumberColumn("Ticker mentions", format="%d"),
                "newest_item_time": st.column_config.TextColumn("Newest item"),
            },
        )


def queue_completed_run_navigation(run_id: str) -> None:
    st.session_state["pending_selected_run_id"] = run_id
    st.session_state["completed_run_notice"] = f"Run `{run_id}` completed."
    st.session_state["pending_page"] = "Results Dashboard"


def latest_matching_run(
    db_path: Path,
    *,
    source_ids: list[int],
    data_mode: str,
) -> dict[str, Any] | None:
    runs = load_analysis_runs(db_path)
    if runs.empty:
        return None

    normalized_source_ids = sorted(int(source_id) for source_id in source_ids)
    for row in runs.to_dict(orient="records"):
        if row.get("status") != "completed" or row.get("data_mode") != data_mode:
            continue
        if int((row.get("summary_json") or {}).get("items_analyzed", 0)) <= 0:
            continue
        row_source_ids = sorted(int(source_id) for source_id in row.get("selected_source_ids", []))
        if row_source_ids == normalized_source_ids:
            return row
    return None


def save_run_artifacts(
    *,
    artifacts_dir: Path,
    run_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    items: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    mentions_df: pd.DataFrame,
    ticker_summary_df: pd.DataFrame,
    time_bucket_df: pd.DataFrame,
) -> None:
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "analysis_run_id": run_id,
        "config": config,
        "summary": summary,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "raw_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "classifications.json").write_text(
        json.dumps(classifications, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    mentions_df.to_csv(run_dir / "mentions.csv", index=False)
    ticker_summary_df.to_csv(run_dir / "ticker_summary.csv", index=False)
    time_bucket_df.to_csv(run_dir / "ticker_time_buckets.csv", index=False)


def build_run_display_label(run_record: dict[str, Any], source_lookup: dict[int, str]) -> str:
    timestamp = str(run_record.get("completed_at") or run_record.get("started_at") or "")[:19].replace("T", " ")
    source_ids = [int(source_id) for source_id in run_record.get("selected_source_ids", [])]
    source_names = [source_lookup.get(source_id, f"source:{source_id}") for source_id in source_ids]
    if len(source_names) == 1:
        source_label = source_names[0]
    elif source_names:
        source_label = f"{source_names[0]} +{len(source_names) - 1}"
    else:
        source_label = "No sources"
    summary = run_record.get("summary_json", {}) or {}
    items_analyzed = int(summary.get("items_analyzed", 0))
    status = str(run_record.get("status", "")).upper()
    status_prefix = f"{status} | " if status and status != "COMPLETED" else ""
    return f"{status_prefix}{timestamp} | {source_label} | {items_analyzed} items"


def run_has_analysed_items(run_record: dict[str, Any]) -> bool:
    return int((run_record.get("summary_json") or {}).get("items_analyzed", 0)) > 0


def render_workflow_navigation() -> str:
    st.markdown('<div class="workflow-kicker">Workflow</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workflow",
        PAGES,
        key="page",
        horizontal=True,
        format_func=lambda value: PAGE_LABELS.get(value, value),
        label_visibility="collapsed",
    )
    st.caption(f"{page}: {PAGE_CONTEXT.get(page, '')}")
    return page


def run_completed_label(run_record: dict[str, Any]) -> str:
    raw_timestamp = str(run_record.get("completed_at") or run_record.get("started_at") or "")
    if not raw_timestamp:
        return "Unstarted"
    return raw_timestamp[:19].replace("T", " ")


def format_source_names(run_record: dict[str, Any], source_lookup: dict[int, str]) -> str:
    source_names = [
        source_lookup.get(int(source_id), f"source:{int(source_id)}")
        for source_id in run_record.get("selected_source_ids", [])
    ]
    return ", ".join(source_names) if source_names else "No sources"


def run_elapsed_minutes(run_record: dict[str, Any]) -> int | None:
    started_at = str(run_record.get("started_at") or "")
    if not started_at:
        return None
    try:
        started = parse_created_time(started_at)
    except Exception:
        return None
    return max(int((datetime.now(UTC) - started).total_seconds() // 60), 0)


def run_elapsed_label(run_record: dict[str, Any]) -> str:
    minutes = run_elapsed_minutes(run_record)
    if minutes is None:
        started_at = str(run_record.get("started_at") or "")
        if not started_at:
            return "unknown age"
        return started_at[:19].replace("T", " ")
    if minutes < 60:
        return f"{minutes} min"
    hours, remainder = divmod(minutes, 60)
    return f"{hours}h {remainder}m"


def render_latest_run_status(runs: pd.DataFrame, source_lookup: dict[int, str]) -> None:
    if runs.empty:
        return
    latest = runs.iloc[0].to_dict()
    summary = latest.get("summary_json", {}) or {}
    status = str(latest.get("status", "unknown")).lower()
    items_analyzed = int(summary.get("items_analyzed", 0))
    items_collected = int(summary.get("items_collected", items_analyzed) or 0)
    run_label = f"`{latest.get('analysis_run_id')}`"
    source_label = format_source_names(latest, source_lookup)

    if status == "completed":
        st.success(f"Latest run complete: {run_label} with {items_analyzed} analysed item(s).")
    elif status == "running":
        st.warning(f"Latest run still running: {run_label} started {run_elapsed_label(latest)} ago.")
    elif status == "failed":
        st.error(f"Latest run failed: {run_label}.")
    else:
        st.info(f"Latest run status: {status} for {run_label}.")
    st.caption(f"Sources: {source_label}. Collected: {items_collected}. Completed: `{run_completed_label(latest)}`.")


def render_recent_run_statuses(runs: pd.DataFrame, source_lookup: dict[int, str]) -> None:
    if runs.empty:
        return
    with st.expander("Recent run status", expanded=False):
        for row in runs.head(6).to_dict(orient="records"):
            summary = row.get("summary_json", {}) or {}
            status = str(row.get("status", "unknown")).upper()
            run_id = str(row.get("analysis_run_id", ""))
            items_analyzed = int(summary.get("items_analyzed", 0))
            items_collected = int(summary.get("items_collected", items_analyzed) or 0)
            st.markdown(f"**{status}** `{run_id}`")
            st.caption(
                f"{format_source_names(row, source_lookup)} | started `{str(row.get('started_at', ''))[:19].replace('T', ' ')}` | "
                f"collected {items_collected} | analysed {items_analyzed}"
            )
            error_message = str(summary.get("error_message", "") or "")
            if error_message:
                st.caption(f"Error: {error_message[:180]}")


def render_watchlist_controls(db_path: Path, run_id: str | None) -> None:
    prefs = load_ui_preferences()
    watchlist = prefs.get("watchlist", [])
    ticker_options = set(watchlist)
    if run_id:
        summary_df = load_run_ticker_summaries(db_path, run_id)
        if not summary_df.empty:
            ticker_options.update(summary_df["ticker"].dropna().astype(str).tolist())

    with st.expander("Watchlist", expanded=bool(watchlist)):
        options = sorted(ticker_options)
        selected = st.multiselect(
            "Pinned tickers",
            options,
            default=[ticker for ticker in watchlist if ticker in options],
            key="watchlist_editor",
        )
        added_ticker = normalize_ticker(st.text_input("Add ticker", key="watchlist_add_ticker"))
        if st.button("Save watchlist", key="save_watchlist", use_container_width=True):
            update_watchlist([*selected, added_ticker] if added_ticker else selected)
            st.success("Watchlist saved.")
            st.rerun()


def render_sidebar_quick_search(db_path: Path, run_id: str | None) -> None:
    if not run_id:
        return

    query = st.text_input("Quick find", placeholder="Ticker, company, thread, claim", key="global_quick_find")
    query = query.strip()
    if len(query) < 2:
        return

    summary_df = load_run_ticker_summaries(db_path, run_id)
    mentions_df = load_run_mentions(db_path, run_id)
    lowered = query.lower()

    ticker_matches = pd.DataFrame()
    if not summary_df.empty:
        ticker_matches = summary_df[
            summary_df["ticker"].astype(str).str.lower().str.contains(lowered, regex=False)
            | summary_df["company_name"].astype(str).str.lower().str.contains(lowered, regex=False)
        ].head(4)

    mention_matches = pd.DataFrame()
    if not mentions_df.empty:
        searchable = (
            mentions_df["ticker"].astype(str)
            + " "
            + mentions_df["company_name"].astype(str)
            + " "
            + mentions_df["thread_title"].astype(str)
            + " "
            + mentions_df["summary"].astype(str)
            + " "
            + mentions_df["body_text"].astype(str)
        ).str.lower()
        mention_matches = mentions_df[searchable.str.contains(lowered, regex=False)].head(3)

    if ticker_matches.empty and mention_matches.empty:
        st.caption("No current-run matches.")
        return

    if not ticker_matches.empty:
        st.caption("Ticker matches")
        for row in ticker_matches.to_dict(orient="records"):
            ticker = str(row["ticker"])
            if st.button(
                f"{ticker} · {format_metric_value(row.get('emerging_ticker_score'), decimals=1)} emerging",
                key=f"focus_ticker_{run_id}_{ticker}",
                use_container_width=True,
            ):
                st.session_state["ticker_focus"] = ticker
                st.session_state["page"] = "Ticker Trends"
                st.rerun()

    if not mention_matches.empty:
        st.caption("Mention matches")
        for row in mention_matches.to_dict(orient="records"):
            st.caption(f"{row['ticker']} · {str(row['thread_title'])[:72]}")


def seed_sources_from_file(db_path: Path, sources_path: Path) -> int:
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    for source in sources:
        add_source(
            db_path,
            source_key=source["source_key"],
            source_type=source["source_type"],
            display_name=source["display_name"],
            normalized_value=source["normalized_value"],
            url=source["url"],
            notes=source.get("notes", ""),
            active=bool(source.get("active", True)),
        )
    return len(sources)


def ensure_seed_sources(db_path: Path, sources_path: Path) -> None:
    if sources_path.exists():
        seed_sources_from_file(db_path, sources_path)


def resolve_time_window(
    *,
    anchor: datetime,
    label: str,
    custom_range: tuple[date, date] | None,
    previous_run: dict[str, Any] | None,
) -> tuple[datetime, datetime]:
    if label == "Custom date range" and custom_range:
        start_date, end_date = custom_range
        start_dt = datetime.combine(start_date, time.min, tzinfo=UTC)
        end_dt = datetime.combine(end_date, time.max, tzinfo=UTC)
        return start_dt, end_dt

    if label == "Since last completed run" and previous_run:
        previous_end = parse_created_time(str(previous_run.get("completed_at") or previous_run["time_window_end"]))
        if previous_end >= anchor:
            start_dt = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = (previous_end + timedelta(seconds=1)).replace(microsecond=0)
        end_dt = anchor.replace(microsecond=0)
        return start_dt, end_dt
    if label == "Since last completed run":
        label = "Last 7 days"

    days = TIME_WINDOW_OPTIONS[label]
    end_dt = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = (end_dt - timedelta(days=(days or 1) - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_dt, end_dt


def run_analysis_pipeline(
    *,
    ticker_path: Path,
    db_path: Path,
    artifacts_dir: Path,
    model_name: str,
    ollama_url: str,
    selected_sources: list[dict[str, Any]],
    time_window_label: str,
    time_window_start: datetime,
    time_window_end: datetime,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    include_comments: bool,
    run_deeper_analysis: bool,
    progress_bar: st.delta_generator.DeltaGenerator,
    status_box: st.delta_generator.DeltaGenerator,
) -> tuple[str, dict[str, Any]]:
    reddit_client = RedditClient()
    extractor = TickerExtractor(ticker_path)
    classifier = OllamaClassifier(model=model_name, endpoint=ollama_url)
    run_config = {
        "data_mode": "live",
        "selected_source_ids": [int(source["source_id"]) for source in selected_sources],
        "selected_sources": [
            {
                "source_id": int(source["source_id"]),
                "display_name": str(source["display_name"]),
                "source_type": str(source["source_type"]),
                "url": str(source["url"]),
            }
            for source in selected_sources
        ],
        "time_window_label": time_window_label,
        "time_window_start": time_window_start.isoformat(),
        "time_window_end": time_window_end.isoformat(),
        "max_posts_per_source": max_posts_per_source,
        "max_comments_per_thread": max_comments_per_thread,
        "include_comments": include_comments,
        "run_deeper_analysis": run_deeper_analysis,
        "ollama_model": model_name,
        "ollama_url": ollama_url,
    }

    run_id = create_analysis_run(db_path, run_config)

    item_rows: list[dict[str, Any]] = []
    item_lookup: dict[str, dict[str, Any]] = {}
    classification_rows: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    fallback_count = 0
    run_completed = False

    def failure_summary(stage: str, error: Exception | str) -> dict[str, Any]:
        return {
            "items_collected": len(items),
            "items_analyzed": len(classification_rows),
            "ticker_mentions": 0,
            "tickers_found": 0,
            "bullish_items": 0,
            "bearish_items": 0,
            "neutral_items": 0,
            "irrelevant_items": 0,
            "top_mentioned_tickers": [],
            "top_accelerating_tickers": [],
            "high_depth_posts_found": 0,
            "low_mentions_high_signal_count": 0,
            "newly_detected_tickers_count": 0,
            "fallback_count": fallback_count,
            "failed_stage": stage,
            "error_message": str(error),
        }

    try:
        progress_bar.progress(0.02, text="Scraping Reddit sources...")
        status_box.info(
            f"Scraping {len(selected_sources)} source(s) from "
            f"{time_window_start.date().isoformat()} to {time_window_end.date().isoformat()}."
        )
        items = reddit_client.collect_live_items(
            selected_sources=selected_sources,
            window_start=time_window_start,
            window_end=time_window_end,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
        )
        if not items:
            source_names = ", ".join(str(source.get("display_name", source.get("source_id"))) for source in selected_sources)
            raise RuntimeError(
                "No Reddit items matched this source selection and time window. "
                "The run was saved as failed so it remains visible in Run health. "
                f"Sources: {source_names}. Window: {time_window_start.date().isoformat()} to {time_window_end.date().isoformat()}."
            )

        total_items = max(len(items), 1)
        for index, item in enumerate(items, start=1):
            status_box.info(f"Classifying {index}/{len(items)}: {item['thread_title']}")
            progress_bar.progress(index / total_items, text=f"Analysed {index}/{len(items)} items")

            normalized_item = dict(item)
            normalized_item["author_hash"] = author_hash(item.get("author", ""))
            item_rows.append(
                {
                    "item_id": normalized_item["item_id"],
                    "source_id": normalized_item.get("source_id"),
                    "source_type": normalized_item.get("source_type", ""),
                    "source_name": normalized_item.get("source_name", ""),
                    "subreddit": normalized_item.get("subreddit", ""),
                    "thread_id": normalized_item.get("thread_id", normalized_item["item_id"]),
                    "thread_title": normalized_item.get("thread_title", ""),
                    "thread_url": normalized_item.get("thread_url", ""),
                    "comment_id": normalized_item.get("comment_id"),
                    "parent_id": normalized_item.get("parent_id"),
                    "item_type": normalized_item.get("item_type", "post"),
                    "body_text": normalized_item.get("body_text", ""),
                    "author": normalized_item.get("author", ""),
                    "author_hash": normalized_item["author_hash"],
                    "score": int(normalized_item.get("score", 0)),
                    "created_time": normalized_item.get("created_time", ""),
                    "permalink": normalized_item.get("permalink", ""),
                    "comment_depth": int(normalized_item.get("comment_depth", 0)),
                    "raw_json": normalized_item,
                }
            )
            item_lookup[normalized_item["item_id"]] = normalized_item

            extraction = extractor.extract(" ".join([normalized_item.get("thread_title", ""), normalized_item.get("body_text", "")]))
            extracted = {"tickers": extraction.tickers, "company_names": extraction.company_names}
            stage_one = classifier.classify(normalized_item, extracted)
            if stage_one.mode == "fallback":
                fallback_count += 1
            classification_rows[normalized_item["item_id"]] = build_classification_record(
                normalized_item,
                stage_one.payload,
                classifier_mode=stage_one.mode,
                model_name=model_name,
            )

        temp_mentions = [
            mention
            for item_id, classification_record in classification_rows.items()
            for mention in build_mention_records(item_lookup[item_id], classification_record)
        ]
        temp_long_df = pd.DataFrame(temp_mentions)
        temp_summary_df = build_ticker_summaries(temp_long_df) if not temp_long_df.empty else pd.DataFrame()

        if run_deeper_analysis and not temp_long_df.empty:
            low_signal_tickers = set(
                temp_summary_df.loc[temp_summary_df["low_mentions_high_signal"], "ticker"].tolist()
            ) if not temp_summary_df.empty else set()
            eligible_item_ids = {
                item_id
                for item_id, row in classification_rows.items()
                if row["depth_score"] >= 7 or row["evidence_quality"] == "high" or row["needs_deeper_analysis"]
            }
            if low_signal_tickers:
                eligible_item_ids.update(temp_long_df.loc[temp_long_df["ticker"].isin(low_signal_tickers), "item_id"].tolist())

            for item_id in sorted(eligible_item_ids):
                deeper_result = classifier.deep_analyze(item_lookup[item_id], classification_rows[item_id])
                classification_rows[item_id]["deeper_analysis_json"] = deeper_result.payload

        final_mentions = [
            mention
            for item_id, classification_record in classification_rows.items()
            for mention in build_mention_records(item_lookup[item_id], classification_record)
        ]
        long_df = pd.DataFrame(final_mentions)
        ticker_summary_df = build_ticker_summaries(long_df) if not long_df.empty else pd.DataFrame()
        long_df = apply_ticker_flags_to_mentions(long_df, ticker_summary_df) if not long_df.empty else long_df
        time_bucket_df = build_time_bucket_summary(long_df) if not long_df.empty else pd.DataFrame()
        summary = build_run_summary(long_df, ticker_summary_df)
        item_sentiments = [row["sentiment"] for row in classification_rows.values()]
        summary["items_collected"] = len(items)
        summary["items_analyzed"] = len(classification_rows)
        summary["bullish_items"] = item_sentiments.count("bullish")
        summary["bearish_items"] = item_sentiments.count("bearish")
        summary["neutral_items"] = item_sentiments.count("neutral")
        summary["irrelevant_items"] = item_sentiments.count("irrelevant")
        summary["fallback_count"] = fallback_count

        replaceable_mentions = long_df.to_dict(orient="records") if not long_df.empty else []
        replaceable_summaries = ticker_summary_df.to_dict(orient="records") if not ticker_summary_df.empty else []
        replaceable_buckets = time_bucket_df.to_dict(orient="records") if not time_bucket_df.empty else []
        upsert_reddit_items(db_path, item_rows)
        replace_run_classifications(db_path, run_id, list(classification_rows.values()))
        replace_run_mentions(db_path, run_id, replaceable_mentions)
        replace_run_ticker_summaries(db_path, run_id, replaceable_summaries)
        replace_run_time_buckets(db_path, run_id, replaceable_buckets)
        save_run_artifacts(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            config=run_config,
            summary=summary,
            items=item_rows,
            classifications=list(classification_rows.values()),
            mentions_df=long_df,
            ticker_summary_df=ticker_summary_df,
            time_bucket_df=time_bucket_df,
        )
        complete_analysis_run(db_path, run_id, summary)
        run_completed = True

        progress_bar.empty()
        status_box.success(f"Analysis complete. Run `{run_id}` saved with {len(classification_rows)} analysed item(s).")
        return run_id, summary
    except Exception as exc:
        if not run_completed:
            try:
                fail_analysis_run(db_path, run_id, failure_summary("collection_or_analysis", exc))
            except Exception:
                logging.exception("Failed to mark analysis run %s as failed", run_id)
        raise


def build_sidebar_state(db_path: Path) -> tuple[str, str | None, pd.DataFrame, pd.DataFrame]:
    sources = load_sources(db_path)
    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
    pending_page = st.session_state.pop("pending_page", None)
    if pending_page in PAGES:
        st.session_state["page"] = pending_page
    pending_run_id = st.session_state.pop("pending_selected_run_id", None)
    if pending_run_id and not runs.empty and pending_run_id in runs["analysis_run_id"].tolist():
        st.session_state["selected_run_id"] = pending_run_id

    page = render_workflow_navigation()

    completed_runs = runs[
        (runs["status"] == "completed")
        & (runs.apply(lambda row: run_has_analysed_items(row.to_dict()), axis=1))
    ].copy() if not runs.empty else runs
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}

    with st.sidebar:
        st.subheader("Run context")
        if not runs.empty:
            overview_cols = st.columns(2)
            overview_cols[0].metric("Completed", len(completed_runs))
            overview_cols[1].metric("All runs", len(runs))
            render_latest_run_status(runs, source_lookup)
            render_recent_run_statuses(runs, source_lookup)

        show_all_runs = st.toggle(
            "Show all runs",
            value=False,
            help="Includes running, failed, and zero-item runs that are hidden from the default results picker.",
            disabled=runs.empty,
        )
        selectable_runs = runs if show_all_runs else (completed_runs if not completed_runs.empty else runs)
        run_label_lookup = {
            str(row["analysis_run_id"]): build_run_display_label(row, source_lookup)
            for row in selectable_runs.to_dict(orient="records")
        }

        run_id = None
        if not selectable_runs.empty:
            run_search = st.text_input("Find run", placeholder="Search ID, date, or source")
            filtered_runs = selectable_runs.copy()
            if run_search.strip():
                query = run_search.strip().lower()
                filtered_runs = selectable_runs[
                    selectable_runs["analysis_run_id"].astype(str).str.lower().str.contains(query)
                    | selectable_runs["status"].astype(str).str.lower().str.contains(query)
                    | selectable_runs["data_mode"].astype(str).str.lower().str.contains(query)
                    | selectable_runs["analysis_run_id"].astype(str).map(
                        lambda value: run_label_lookup.get(str(value), str(value)).lower()
                    ).str.contains(query)
                ].copy()

            if filtered_runs.empty:
                st.info("No runs matched that search.")
                filtered_runs = selectable_runs

            filtered_run_options = filtered_runs["analysis_run_id"].tolist()
            default_run_id = st.session_state.get("selected_run_id")
            if default_run_id not in filtered_run_options:
                default_run_id = str(filtered_run_options[0])
            run_index = filtered_run_options.index(default_run_id) if default_run_id in filtered_run_options else 0
            run_id = st.selectbox(
                "Analysis run",
                filtered_run_options,
                index=run_index,
                key="selected_run_id",
                format_func=lambda value: run_label_lookup.get(str(value), str(value)),
            )
            selected_rows = selectable_runs[selectable_runs["analysis_run_id"] == run_id]
            if not selected_rows.empty:
                selected_record = selected_rows.iloc[0].to_dict()
                st.caption(f"Sources: {format_source_names(selected_record, source_lookup)}")
                st.caption(f"Completed: `{run_completed_label(selected_record)}`")
            if not completed_runs.empty and len(completed_runs) != len(runs):
                st.caption("Default view shows completed runs with analysed items. Turn on 'Show all runs' to inspect failed, running, or zero-item runs.")
        else:
            st.info("No analysis runs saved yet.")

        if run_id:
            render_watchlist_controls(db_path, run_id)
            render_sidebar_quick_search(db_path, run_id)

        st.divider()
        st.caption(f"Current step: `{page}`")
        with st.expander("Storage paths", expanded=False):
            st.caption(f"SQLite: `{db_path}`")

    return page, run_id, sources, runs


def render_sources_page(db_path: Path) -> None:
    st.header("Sources")
    st.write("Add subreddit sources or Reddit thread URLs. Inputs are normalized before saving.")
    st.caption("Sources are the watchlist. To scrape Reddit and analyse posts, open `02 Scrape` in the workflow.")

    if st.button("Go to Run Analysis", type="primary"):
        st.session_state["pending_page"] = "Run Analysis"
        st.rerun()

    with st.form("add_source_form", clear_on_submit=True):
        source_input = st.text_input("Paste a subreddit name or Reddit thread URL")
        source_notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Add source", type="primary")
        if submitted:
            parsed = parse_source_input(source_input)
            if not parsed:
                st.error("Input was not recognized as a subreddit or Reddit thread URL.")
            else:
                add_source(
                    db_path,
                    source_key=parsed.source_key,
                    source_type=parsed.source_type,
                    display_name=parsed.display_name,
                    normalized_value=parsed.normalized_value,
                    url=parsed.url,
                    notes=source_notes,
                )
                st.success(f"Saved source `{parsed.display_name}`.")
                st.rerun()

    sources = load_sources(db_path)
    if sources.empty:
        st.info("No sources saved yet.")
        return

    st.subheader("Saved sources")
    editable = sources[["source_id", "source_type", "display_name", "url", "active", "notes", "date_added"]].copy()
    edited = st.data_editor(
        editable,
        hide_index=True,
        use_container_width=True,
        disabled=["source_id", "source_type", "display_name", "url", "date_added"],
        column_config={
            "active": st.column_config.CheckboxColumn("active"),
            "url": st.column_config.LinkColumn("url"),
        },
    )
    save_col, delete_col = st.columns([1, 1])
    with save_col:
        if st.button("Save source changes"):
            update_sources(db_path, edited.to_dict(orient="records"))
            st.success("Source changes saved.")
            st.rerun()
    with delete_col:
        delete_ids = st.multiselect(
            "Delete sources",
            sources["source_id"].tolist(),
            format_func=lambda source_id: f"{int(source_id)} · {sources.loc[sources['source_id'] == source_id, 'display_name'].iloc[0]}",
        )
        if st.button("Delete selected sources"):
            delete_sources(db_path, [int(source_id) for source_id in delete_ids])
            st.success("Selected sources deleted.")
            st.rerun()


def render_run_analysis_page(
    *,
    db_path: Path,
    artifacts_dir: Path,
    ticker_path: Path,
) -> None:
    st.header("Run Analysis")
    sources = load_sources(db_path)
    if sources.empty:
        st.warning("Add at least one source before running analysis.")
        return

    active_sources = sources[sources["active"]]
    if active_sources.empty:
        st.warning("No active sources are enabled.")
        return

    data_mode = "live"
    st.info(
        "Reddit scraping uses public Reddit RSS feeds and stores the collected run locally. "
        "No Reddit API keys are required, but large runs may be rate-limited."
    )

    selected_source_ids = st.multiselect(
        "Active sources to include",
        active_sources["source_id"].tolist(),
        default=active_sources["source_id"].tolist(),
        format_func=lambda source_id: active_sources.loc[active_sources["source_id"] == source_id, "display_name"].iloc[0],
    )
    anchor = datetime.now(UTC)
    previous_run = latest_matching_run(
        db_path,
        source_ids=[int(source_id) for source_id in selected_source_ids],
        data_mode=data_mode,
    ) if selected_source_ids else None
    use_last_run_window = st.toggle(
        "Use last completed run -> now",
        value=False,
        disabled=not previous_run,
        help="If enabled, the analysis window starts immediately after the most recent completed run for the current source selection.",
    )
    if previous_run:
        st.caption(
            "Most recent matching completed run: "
            f"`{previous_run['analysis_run_id']}` at "
            f"`{previous_run.get('completed_at') or previous_run.get('started_at')}`."
        )
    else:
        st.caption("No matching completed run exists yet for the current source selection.")
    manual_time_window_options = [
        option for option in TIME_WINDOW_OPTIONS.keys() if option != "Since last completed run"
    ]
    time_window_index = 2
    time_window_label = st.selectbox(
        "Time window",
        manual_time_window_options,
        index=time_window_index,
        disabled=use_last_run_window,
    )
    effective_time_window_label = "Since last completed run" if use_last_run_window else time_window_label
    custom_range = None
    if effective_time_window_label == "Custom date range":
        start_default = anchor.date() - timedelta(days=13)
        custom_range = st.date_input("Custom date range", value=(start_default, anchor.date()))
        if isinstance(custom_range, date):
            custom_range = (custom_range, custom_range)
    elif effective_time_window_label == "Since last completed run" and not previous_run:
        st.warning("No matching completed run was found for the current source selection, so choose another time window.")

    max_posts_per_source = st.slider("Max posts per source", min_value=1, max_value=20, value=8)
    max_comments_per_thread = st.slider("Max comments per thread", min_value=0, max_value=20, value=4)
    item_scope = st.radio("Analyse scope", ["Posts + comments", "Posts only"], horizontal=True)
    include_comments = item_scope == "Posts + comments"
    run_deeper_analysis = st.toggle("Run deeper analysis for qualifying items", value=True)
    ollama_url = st.text_input("Ollama URL", DEFAULT_OLLAMA_URL)
    if is_localhost_url(ollama_url):
        st.caption(
            "`localhost` only works when this Streamlit app runs on the same machine as Ollama. "
            "On Streamlit Community Cloud, `localhost` points to the cloud container, not your Mac."
        )
    available_models, model_error = discover_ollama_models(ollama_url)
    if available_models:
        default_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
        model_name = st.selectbox("Ollama model", available_models, index=available_models.index(default_model))
        if DEFAULT_MODEL not in available_models:
            st.caption(f"Configured model `{DEFAULT_MODEL}` is not installed locally. Using `{default_model}` by default.")
    else:
        model_name = st.text_input("Ollama model", DEFAULT_MODEL)
        if model_error:
            st.caption(f"Could not list local Ollama models: {model_error}")

    start_dt, end_dt = resolve_time_window(
        anchor=anchor,
        label=effective_time_window_label,
        custom_range=custom_range,
        previous_run=previous_run,
    )
    st.caption(f"Window resolved against current time: `{start_dt.date().isoformat()}` to `{end_dt.date().isoformat()}`.")
    if use_last_run_window and previous_run:
        st.caption(
            f"Most recent matching completed run completed at `{str(previous_run.get('completed_at') or previous_run['time_window_end'])[:19]}` "
            "for this source set."
        )

    run_disabled = False
    if st.button("Scrape Reddit and Run Analysis", type="primary", use_container_width=True, disabled=run_disabled):
        if not selected_source_ids:
            st.error("Select at least one source.")
            return
        if effective_time_window_label == "Since last completed run" and not previous_run:
            st.error("A matching completed run is required to use the 'Since last completed run' option.")
            return

        selected_sources = sources[sources["source_id"].isin(selected_source_ids)].to_dict(orient="records")
        progress_bar = st.progress(0.0, text="Preparing analysis...")
        status_box = st.empty()
        try:
            run_id, _summary = run_analysis_pipeline(
                ticker_path=ticker_path,
                db_path=db_path,
                artifacts_dir=artifacts_dir,
                model_name=model_name,
                ollama_url=ollama_url,
                selected_sources=selected_sources,
                time_window_label=effective_time_window_label,
                time_window_start=start_dt,
                time_window_end=end_dt,
                max_posts_per_source=max_posts_per_source,
                max_comments_per_thread=max_comments_per_thread,
                include_comments=include_comments,
                run_deeper_analysis=run_deeper_analysis,
                progress_bar=progress_bar,
                status_box=status_box,
            )
            queue_completed_run_navigation(run_id)
            st.rerun()
        except RuntimeError as exc:
            progress_bar.empty()
            status_box.empty()
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            progress_bar.empty()
            status_box.empty()
            st.exception(exc)


def render_run_summary_cards(summary: dict[str, Any]) -> None:
    metrics = st.columns(6)
    metrics[0].metric("Items analysed", summary.get("items_analyzed", 0))
    metrics[1].metric("Ticker mentions", summary.get("ticker_mentions", 0))
    metrics[2].metric("Bullish items", summary.get("bullish_items", 0))
    metrics[3].metric("Bearish items", summary.get("bearish_items", 0))
    metrics[4].metric("High-depth posts", summary.get("high_depth_posts_found", 0))
    metrics[5].metric("Fallback items", summary.get("fallback_count", 0))

    extra = st.columns(4)
    extra[0].metric("Neutral items", summary.get("neutral_items", 0))
    extra[1].metric("Irrelevant items", summary.get("irrelevant_items", 0))
    extra[2].metric("Low-mention high-signal", summary.get("low_mentions_high_signal_count", 0))
    extra[3].metric("Newly detected tickers", summary.get("newly_detected_tickers_count", 0))

    top_cols = st.columns(2)
    with top_cols[0]:
        top_mentioned = [
            row for row in summary.get("top_mentioned_tickers", [])
            if row.get("ticker") != "UNKNOWN"
        ]
        render_copyable_ticker_chips(
            [str(row.get("ticker", "")) for row in top_mentioned],
            label="Top mentioned tickers",
            help_text="Click to copy ticker.",
            max_items=8,
        )
    with top_cols[1]:
        top_accelerating = [
            row for row in summary.get("top_accelerating_tickers", [])
            if row.get("ticker") != "UNKNOWN"
        ]
        render_copyable_ticker_chips(
            [str(row.get("ticker", "")) for row in top_accelerating],
            label="Top accelerating tickers",
            help_text="Click to copy ticker.",
            max_items=8,
        )

    if summary.get("fallback_count", 0):
        st.warning(
            "Some items used heuristic fallback classification because the Ollama endpoint was unavailable "
            "or returned invalid JSON."
        )


def filter_results_dataframe(long_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    working = long_df.copy()
    st.subheader("Review controls")

    preset_names = list(FILTER_PRESETS.keys())
    preset_name = st.selectbox(
        "Preset",
        preset_names,
        index=0,
        key=f"results_preset_{run_id}",
        help="Start from the view you need most often, then refine only when necessary.",
    )
    preset = FILTER_PRESETS[preset_name]
    preset_slug = preset_name.lower().replace(" ", "_").replace("-", "_")

    primary_cols = st.columns([1.4, 1.4, 1])
    ticker_filter = primary_cols[0].multiselect(
        "Ticker",
        sorted(working["ticker"].dropna().unique()),
        key=f"ticker_filter_{run_id}_{preset_slug}",
    )
    source_filter = primary_cols[1].multiselect(
        "Source",
        sorted(working["source_name"].dropna().unique()),
        key=f"source_filter_{run_id}_{preset_slug}",
    )
    watchlist = get_watchlist()
    only_watchlist = primary_cols[2].checkbox(
        "Watchlist only",
        value=False,
        disabled=not watchlist,
        key=f"watchlist_filter_{run_id}_{preset_slug}",
    )

    with st.expander("Advanced filters", expanded=False):
        filter_cols = st.columns(3)
        sentiment_filter = filter_cols[0].multiselect(
            "Sentiment",
            ["bullish", "bearish", "neutral", "irrelevant"],
            default=preset["sentiment"],
            key=f"sentiment_filter_{run_id}_{preset_slug}",
        )
        min_confidence = filter_cols[1].slider(
            "Min confidence",
            0.0,
            1.0,
            float(preset["min_confidence"]),
            0.05,
            key=f"min_confidence_{run_id}_{preset_slug}",
        )
        min_depth = filter_cols[2].slider(
            "Min depth score",
            0,
            10,
            int(preset["min_depth"]),
            key=f"min_depth_{run_id}_{preset_slug}",
        )

        numeric_cols = st.columns(4)
        min_alpha = numeric_cols[0].slider(
            "Min alpha signal",
            0,
            100,
            int(preset["min_alpha"]),
            key=f"min_alpha_{run_id}_{preset_slug}",
        )
        max_hype = numeric_cols[1].slider(
            "Max hype",
            0,
            10,
            int(preset["max_hype"]),
            key=f"max_hype_{run_id}_{preset_slug}",
        )
        hide_hype = numeric_cols[2].checkbox(
            "Hide hype/meme",
            value=bool(preset["hide_hype"]),
            key=f"hide_hype_{run_id}_{preset_slug}",
        )
        only_deeper = numeric_cols[3].checkbox(
            "Needs verification",
            value=bool(preset["only_deeper"]),
            key=f"only_deeper_{run_id}_{preset_slug}",
        )

        flag_cols = st.columns(2)
        only_low_signal = flag_cols[0].checkbox(
            "Low-mention high-signal",
            value=bool(preset["only_low_signal"]),
            key=f"only_low_signal_{run_id}_{preset_slug}",
        )
        only_new = flag_cols[1].checkbox(
            "Newly detected",
            value=bool(preset["only_new"]),
            key=f"only_new_{run_id}_{preset_slug}",
        )

    if "sentiment_filter" not in locals():
        sentiment_filter = preset["sentiment"]
        min_confidence = float(preset["min_confidence"])
        min_depth = int(preset["min_depth"])
        min_alpha = int(preset["min_alpha"])
        max_hype = int(preset["max_hype"])
        hide_hype = bool(preset["hide_hype"])
        only_deeper = bool(preset["only_deeper"])
        only_low_signal = bool(preset["only_low_signal"])
        only_new = bool(preset["only_new"])

    if ticker_filter:
        working = working[working["ticker"].isin(ticker_filter)]
    if source_filter:
        working = working[working["source_name"].isin(source_filter)]
    if only_watchlist:
        working = working[working["ticker"].isin(watchlist)]
    working = working[working["sentiment"].isin(sentiment_filter)]
    working = working[working["confidence"] >= min_confidence]
    working = working[working["depth_score"] >= min_depth]
    working = working[working["alpha_signal_score"] >= min_alpha]
    working = working[working["hype_score"] <= max_hype]
    if hide_hype:
        working = working[~((working["hype_score"] >= 7) | ((working["evidence_quality"] == "low") & (working["claim_specificity_score"] <= 3)))]
    if only_deeper:
        working = working[working["needs_deeper_analysis"]]
    if only_low_signal:
        working = working[working["low_mentions_high_signal"]]
    if only_new:
        working = working[working["new_ticker_detected"]]

    return working


def render_filtered_empty_state(original_df: pd.DataFrame) -> None:
    st.info("No mentions matched the current filters.")
    if original_df.empty:
        return
    st.caption(
        "Try `Balanced review`, clear ticker/source selections, or lower the confidence and alpha thresholds. "
        "If the run itself is sparse, check Source coverage to see whether the scrape returned enough items."
    )


def mode_value(series: pd.Series) -> str:
    if series.empty:
        return ""
    modes = series.mode(dropna=True)
    return str(modes.iloc[0]) if not modes.empty else str(series.iloc[0])


def build_thread_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    group_columns = ["thread_id", "thread_title", "source_name"]
    for keys, group in filtered_df.groupby(group_columns, dropna=False):
        thread_id, thread_title, source_name = keys
        group = group.sort_values(["alpha_signal_score", "confidence"], ascending=False)
        top = group.iloc[0]
        tickers = sorted(str(ticker) for ticker in group["ticker"].dropna().unique() if str(ticker) != "UNKNOWN")
        companies = sorted(str(company) for company in group["company_name"].dropna().unique() if str(company))
        records.append(
            {
                "tickers": ", ".join(tickers),
                "companies": ", ".join(companies[:4]),
                "source": source_name,
                "thread_title": thread_title,
                "mentions": int(len(group)),
                "ticker_count": len(tickers),
                "unique_authors": int(group["author_hash"].nunique()),
                "sentiment": mode_value(group["sentiment"]),
                "trust": mode_value(group["trust_status"]) if "trust_status" in group.columns else "",
                "max_alpha": round(float(group["alpha_signal_score"].max()), 1),
                "avg_confidence": round(float(group["confidence"].mean()), 2),
                "avg_hype": round(float(group["hype_score"].mean()), 1),
                "top_summary": top.get("summary", ""),
                "permalink": top.get("permalink", ""),
                "latest": str(group["created_time"].max())[:19],
            }
        )
    return pd.DataFrame(records).sort_values(
        ["max_alpha", "mentions", "avg_confidence"],
        ascending=[False, False, False],
    )


def render_mention_details(filtered_df: pd.DataFrame) -> None:
    st.subheader("Details")
    thread_tickers = {
        thread_id: sorted(str(ticker) for ticker in group["ticker"].dropna().unique() if str(ticker) != "UNKNOWN")
        for thread_id, group in filtered_df.groupby("thread_id", dropna=False)
    }
    for row in filtered_df.head(12).to_dict(orient="records"):
        expander_title = f"{row['created_time'][:10]} · {row['ticker']} · {row['source_name']} · {row['sentiment']}"
        with st.expander(expander_title):
            detail_cols = st.columns(5)
            detail_cols[0].metric("confidence", f"{row['confidence']:.2f}")
            detail_cols[1].metric("alpha", f"{row['alpha_signal_score']:.1f}")
            detail_cols[2].metric("hype-adjusted", f"{row['hype_adjusted_signal_score']:.1f}")
            detail_cols[3].metric("controversy", f"{row['controversy_score']:.1f}")
            detail_cols[4].metric("trust", row.get("trust_status", ""))
            render_status_chips(
                [
                    str(row.get("classifier_mode", "unknown")),
                    f"evidence: {row.get('evidence_quality', 'unknown')}",
                    "needs verification" if row.get("needs_deeper_analysis") else "",
                ]
            )
            render_copyable_ticker_chips(
                thread_tickers.get(row.get("thread_id"), [row["ticker"]]),
                label="Tickers in this thread",
                help_text="Click to copy.",
                max_items=12,
            )
            st.write("Original text")
            st.code(row["body_text"])
            st.write("Bull case")
            st.write(row["bull_case"] or "None")
            st.write("Bear case")
            st.write(row["bear_case"] or "None")
            st.write("Red flags")
            st.write(", ".join(row["red_flags"]) if row["red_flags"] else "None")
            st.write("Claims to verify")
            st.write(row["claims_to_verify"] if row["claims_to_verify"] else "None")
            deeper = row.get("deeper_analysis_json") or {}
            if deeper:
                st.write("Deeper analysis")
                st.json(deeper)


def render_run_metadata(db_path: Path, run_id: str) -> None:
    runs = load_analysis_runs(db_path)
    run_rows = runs[runs["analysis_run_id"] == run_id]
    if run_rows.empty:
        return

    run_record = run_rows.iloc[0].to_dict()
    sources = load_sources(db_path)
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    }
    source_names = [
        source_lookup.get(int(source_id), f"source:{int(source_id)}")
        for source_id in run_record.get("selected_source_ids", [])
    ]

    with st.expander("Run metadata", expanded=False):
        st.write(f"Collected window: `{run_record['time_window_start']}` to `{run_record['time_window_end']}`")
        st.write(f"Started at: `{run_record['started_at']}`")
        if run_record.get("completed_at"):
            st.write(f"Completed at: `{run_record['completed_at']}`")
        st.write(f"Sources: {', '.join(source_names) if source_names else 'None'}")


ZERO_FILL_TREND_METRICS = {
    "total_mentions",
    "bullish_mentions",
    "bearish_mentions",
    "neutral_mentions",
    "unique_authors",
    "unique_threads",
    "acceleration_score",
    "emerging_ticker_score",
}
WITHIN_RUN_METRICS = {
    "total_mentions": "Mentions",
    "acceleration_score": "Acceleration score",
    "emerging_ticker_score": "Emerging ticker score",
    "average_alpha_signal_score": "Average alpha signal",
    "average_hype_adjusted_signal_score": "Average hype-adjusted signal",
    "average_hype_score": "Average hype",
    "source_diversity_score": "Source diversity",
}
HISTORY_METRICS = {
    "total_mentions": "Mentions per scrape",
    "mentions_this_period": "Mentions in scrape window",
    "mention_change": "Mention delta inside scrape",
    "acceleration_score": "Acceleration score",
    "emerging_ticker_score": "Emerging ticker score",
    "average_alpha_signal_score": "Average alpha signal",
    "source_diversity_score": "Source diversity",
}
METRIC_DOMAINS = {
    "acceleration_score": [0, 100],
    "emerging_ticker_score": [0, 100],
    "average_alpha_signal_score": [0, 100],
    "average_hype_adjusted_signal_score": [0, 100],
    "average_hype_score": [0, 10],
    "average_depth_score": [0, 10],
    "average_claim_specificity_score": [0, 10],
    "source_diversity_score": [0, 10],
    "controversy_score": [0, 10],
    "net_sentiment_score": [-1, 1],
}
COMPARISON_COLUMNS = [
    "total_mentions",
    "mentions_this_period",
    "mention_change",
    "acceleration_score",
    "emerging_ticker_score",
    "average_alpha_signal_score",
    "source_diversity_score",
    "average_hype_score",
    "net_sentiment_score",
]


def metric_scale(metric_name: str) -> alt.Scale:
    domain = METRIC_DOMAINS.get(metric_name)
    if domain:
        return alt.Scale(domain=domain, nice=True)
    return alt.Scale(zero=True, nice=True)


def build_dense_bucket_frame(bucket_df: pd.DataFrame, tickers: list[str], metric_name: str) -> pd.DataFrame:
    if bucket_df.empty or not tickers or metric_name not in bucket_df.columns:
        return pd.DataFrame()

    working = bucket_df[bucket_df["ticker"].isin(tickers)].copy()
    working["bucket_start_dt"] = pd.to_datetime(working["bucket_start"], errors="coerce", utc=True)
    all_buckets = (
        pd.Series(pd.to_datetime(bucket_df["bucket_start"], errors="coerce", utc=True).dropna().unique())
        .sort_values()
        .tolist()
    )
    if not all_buckets:
        return pd.DataFrame()

    grid = pd.MultiIndex.from_product([tickers, all_buckets], names=["ticker", "bucket_start_dt"]).to_frame(index=False)
    values = working[["ticker", "bucket_start_dt", metric_name]].dropna(subset=["bucket_start_dt"])
    dense = grid.merge(values, on=["ticker", "bucket_start_dt"], how="left")
    if metric_name in ZERO_FILL_TREND_METRICS:
        dense[metric_name] = dense[metric_name].fillna(0)
    return dense.sort_values(["bucket_start_dt", "ticker"])


def prepare_historical_summary_frame(
    historical_summary_df: pd.DataFrame,
    run_label_lookup: dict[str, str],
) -> pd.DataFrame:
    if historical_summary_df.empty:
        return historical_summary_df

    working = historical_summary_df.copy()
    raw_completed = working["completed_at"].where(
        working["completed_at"].notna() & (working["completed_at"].astype(str) != ""),
        working["started_at"],
    )
    working["run_completed_at"] = pd.to_datetime(raw_completed, errors="coerce", utc=True)
    working["run_label"] = working["analysis_run_id"].astype(str).map(
        lambda value: run_label_lookup.get(value, value)
    )
    return working.dropna(subset=["run_completed_at"]).sort_values(["run_completed_at", "ticker"])


def build_run_comparison_frame(
    historical_summary_df: pd.DataFrame,
    *,
    baseline_run_id: str,
    comparison_run_id: str,
) -> pd.DataFrame:
    if historical_summary_df.empty:
        return pd.DataFrame()

    available_columns = [column for column in COMPARISON_COLUMNS if column in historical_summary_df.columns]
    baseline = historical_summary_df[historical_summary_df["analysis_run_id"] == baseline_run_id][
        ["ticker", "company_name", *available_columns]
    ].copy()
    comparison = historical_summary_df[historical_summary_df["analysis_run_id"] == comparison_run_id][
        ["ticker", "company_name", *available_columns]
    ].copy()
    if baseline.empty or comparison.empty:
        return pd.DataFrame()

    merged = comparison.merge(
        baseline,
        on="ticker",
        how="outer",
        suffixes=("_current", "_baseline"),
    )
    merged["company_name"] = merged["company_name_current"].fillna(merged["company_name_baseline"]).fillna("Unknown")
    for column in available_columns:
        current_column = f"{column}_current"
        baseline_column = f"{column}_baseline"
        merged[current_column] = merged[current_column].fillna(0)
        merged[baseline_column] = merged[baseline_column].fillna(0)
        merged[f"{column}_delta"] = merged[current_column] - merged[baseline_column]

    movement_columns = [
        column
        for column in ("total_mentions_delta", "acceleration_score_delta", "emerging_ticker_score_delta")
        if column in merged.columns
    ]
    merged["movement_score"] = merged[movement_columns].abs().sum(axis=1) if movement_columns else 0
    if "total_mentions_delta" in merged.columns and "acceleration_score_delta" in merged.columns:
        merged["movement"] = merged.apply(
            lambda row: f"{format_delta(row['total_mentions_delta'])} mentions / {format_delta(row['acceleration_score_delta'], decimals=1)} accel",
            axis=1,
        )
    return merged.sort_values(["movement_score", "ticker"], ascending=[False, True])


def previous_history_run_id(history_df: pd.DataFrame, current_run_id: str) -> str | None:
    if history_df.empty:
        return None
    history_runs = (
        history_df[["analysis_run_id", "run_completed_at"]]
        .drop_duplicates("analysis_run_id")
        .sort_values("run_completed_at")
    )
    run_ids = history_runs["analysis_run_id"].astype(str).tolist()
    if current_run_id not in run_ids:
        return run_ids[-1] if run_ids else None
    index = run_ids.index(current_run_id)
    if index <= 0:
        return None
    return run_ids[index - 1]


def render_change_summary(
    *,
    comparison_df: pd.DataFrame,
    previous_label: str | None,
) -> None:
    st.subheader("What changed")
    if comparison_df.empty or not previous_label:
        st.caption("Run one more scrape with the same source set to unlock a clean before/after summary.")
        return

    current_rows = comparison_df[comparison_df.get("total_mentions_current", 0) > 0].copy()
    new_rows = current_rows[current_rows.get("total_mentions_baseline", 0) <= 0]
    mention_jump = comparison_df.sort_values("total_mentions_delta", ascending=False).head(1)
    accel_jump = comparison_df.sort_values("acceleration_score_delta", ascending=False).head(1)
    emerging = current_rows.sort_values("emerging_ticker_score_current", ascending=False).head(1)

    cards: list[dict[str, str]] = []
    if not new_rows.empty:
        tickers = ", ".join(new_rows["ticker"].head(4).astype(str).tolist())
        cards.append({"label": "New tickers", "value": tickers, "note": f"{len(new_rows)} appeared vs previous scrape"})
    else:
        cards.append({"label": "New tickers", "value": "None", "note": "No new ticker symbols vs previous scrape"})

    if not mention_jump.empty:
        row = mention_jump.iloc[0]
        cards.append(
            {
                "label": "Mention jump",
                "value": str(row["ticker"]),
                "note": f"{format_delta(row.get('total_mentions_delta'))} mentions since {previous_label}",
            }
        )
    if not accel_jump.empty:
        row = accel_jump.iloc[0]
        cards.append(
            {
                "label": "Acceleration jump",
                "value": str(row["ticker"]),
                "note": f"{format_delta(row.get('acceleration_score_delta'), decimals=1)} acceleration",
            }
        )
    if not emerging.empty:
        row = emerging.iloc[0]
        cards.append(
            {
                "label": "Top current lead",
                "value": str(row["ticker"]),
                "note": f"{format_metric_value(row.get('emerging_ticker_score_current'), decimals=1)} emerging score",
            }
        )
    if "net_sentiment_score_delta" in comparison_df.columns:
        sentiment_flip = comparison_df.assign(
            sentiment_abs_delta=comparison_df["net_sentiment_score_delta"].abs()
        ).sort_values("sentiment_abs_delta", ascending=False).head(1)
        if not sentiment_flip.empty:
            row = sentiment_flip.iloc[0]
            cards.append(
                {
                    "label": "Sentiment move",
                    "value": str(row["ticker"]),
                    "note": f"{format_delta(row.get('net_sentiment_score_delta'), decimals=2)} net sentiment",
                }
            )
    render_insight_cards(cards)


def extract_claims_for_ticker(mentions_df: pd.DataFrame, ticker: str) -> list[str]:
    if mentions_df.empty:
        return []
    claims: list[str] = []
    for value in mentions_df.loc[mentions_df["ticker"] == ticker, "claims_to_verify"].tolist():
        if isinstance(value, list):
            claims.extend(str(item) for item in value if str(item).strip())
        elif str(value).strip():
            claims.append(str(value))
    return sorted(dict.fromkeys(claims))[:8]


def render_evidence_checklist(ticker: str, claims: list[str]) -> None:
    if not claims:
        st.caption("No claims to verify were extracted for this ticker.")
        return
    prefs = load_ui_preferences()
    checks = dict(prefs.get("evidence_checks", {}))
    updated_checks = dict(checks)
    for index, claim in enumerate(claims):
        check_key = f"{ticker}|{claim}"
        updated_checks[check_key] = st.checkbox(
            claim,
            value=bool(checks.get(check_key, False)),
            key=f"evidence_{ticker}_{index}",
        )
    if st.button("Save checklist", key=f"save_evidence_{ticker}"):
        prefs["evidence_checks"] = updated_checks
        save_ui_preferences(prefs)
        st.success("Checklist saved.")


def render_ticker_focus(
    *,
    run_id: str,
    summary_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    long_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    if summary_df.empty:
        return
    watchlist = get_watchlist()
    ticker_options = summary_df["ticker"].dropna().astype(str).tolist()
    focused = st.session_state.get("ticker_focus")
    if focused not in ticker_options:
        focused = next((ticker for ticker in watchlist if ticker in ticker_options), ticker_options[0])
    default_index = ticker_options.index(focused) if focused in ticker_options else 0

    with st.expander("Ticker focus", expanded=False):
        selected_ticker = st.selectbox(
            "Ticker",
            ticker_options,
            index=default_index,
            key=f"ticker_focus_select_{run_id}",
        )
        st.session_state["ticker_focus"] = selected_ticker
        ticker_row = summary_df[summary_df["ticker"] == selected_ticker].iloc[0].to_dict()
        chips = []
        if selected_ticker in watchlist:
            chips.append("watchlist")
        if ticker_row.get("low_mentions_high_signal"):
            chips.append("low-hype lead")
        if ticker_row.get("new_ticker_detected"):
            chips.append("new ticker")
        render_status_chips(chips)

        metric_cols = st.columns(5)
        metric_cols[0].metric("Mentions", int(ticker_row.get("total_mentions", 0)))
        metric_cols[1].metric("Emerging", f"{float(ticker_row.get('emerging_ticker_score', 0)):.1f}")
        metric_cols[2].metric("Acceleration", f"{float(ticker_row.get('acceleration_score', 0)):.1f}")
        metric_cols[3].metric("Alpha", f"{float(ticker_row.get('average_alpha_signal_score', 0)):.1f}")
        metric_cols[4].metric("Hype", f"{float(ticker_row.get('average_hype_score', 0)):.1f}")

        pin_cols = st.columns([1, 4])
        if selected_ticker in watchlist:
            if pin_cols[0].button("Unpin", key=f"unpin_{run_id}_{selected_ticker}"):
                update_watchlist([ticker for ticker in watchlist if ticker != selected_ticker])
                st.rerun()
        elif pin_cols[0].button("Pin", key=f"pin_{run_id}_{selected_ticker}"):
            update_watchlist([*watchlist, selected_ticker])
            st.rerun()
        pin_cols[1].caption("Pinned tickers stay available in the sidebar watchlist across sessions.")

        snapshot_tab, posts_tab, verification_tab = st.tabs(["Snapshot", "Posts", "Verification"])
        with snapshot_tab:
            if not bucket_df.empty:
                render_within_run_chart(bucket_df, [selected_ticker], "total_mentions")
            if not history_df.empty:
                render_history_chart(history_df, [selected_ticker], "total_mentions")

        with posts_tab:
            ticker_mentions = long_df[long_df["ticker"] == selected_ticker].copy()
            if ticker_mentions.empty:
                st.caption("No item-level mentions are available for this ticker.")
            else:
                top_posts = ticker_mentions.sort_values(
                    ["alpha_signal_score", "confidence"],
                    ascending=False,
                ).head(8)
                st.dataframe(
                    top_posts[
                        [
                            "created_time",
                            "source_name",
                            "sentiment",
                            "confidence",
                            "alpha_signal_score",
                            "hype_score",
                            "thread_title",
                            "summary",
                            "permalink",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    column_config={"permalink": st.column_config.LinkColumn("permalink")},
                )

        with verification_tab:
            render_evidence_checklist(selected_ticker, extract_claims_for_ticker(long_df, selected_ticker))


def build_spike_notes(bucket_df: pd.DataFrame, long_df: pd.DataFrame, selected_tickers: list[str]) -> pd.DataFrame:
    if bucket_df.empty or long_df.empty or not selected_tickers:
        return pd.DataFrame()
    top_buckets = (
        bucket_df[bucket_df["ticker"].isin(selected_tickers)]
        .sort_values(["total_mentions", "acceleration_score"], ascending=False)
        .head(5)
    )
    records: list[dict[str, Any]] = []
    for row in top_buckets.to_dict(orient="records"):
        bucket_mentions = long_df[
            (long_df["ticker"] == row["ticker"])
            & (long_df["time_bucket"] == row["time_bucket"])
        ].sort_values(["alpha_signal_score", "confidence"], ascending=False)
        top_mention = bucket_mentions.iloc[0].to_dict() if not bucket_mentions.empty else {}
        records.append(
            {
                "ticker": row["ticker"],
                "bucket_start": row["bucket_start"],
                "mentions": int(row["total_mentions"]),
                "acceleration": round(float(row["acceleration_score"]), 1),
                "likely_driver": top_mention.get("thread_title", ""),
                "summary": top_mention.get("summary", ""),
                "permalink": top_mention.get("permalink", ""),
            }
        )
    return pd.DataFrame(records)


def render_time_line_chart(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    color_column: str,
    y_title: str,
    x_title: str,
    tooltip_columns: list[str],
    height: int = 360,
) -> None:
    if frame.empty or y_column not in frame.columns:
        st.info("No chartable data for this selection.")
        return

    chart = (
        alt.Chart(frame)
        .mark_line(point={"filled": True, "size": 62}, strokeWidth=2)
        .encode(
            x=alt.X(
                f"{x_column}:T",
                title=x_title,
                axis=alt.Axis(format="%b %d", labelAngle=0, tickCount="week"),
            ),
            y=alt.Y(f"{y_column}:Q", title=y_title, scale=metric_scale(y_column)),
            color=alt.Color(f"{color_column}:N", title="Ticker"),
            tooltip=[
                alt.Tooltip(f"{column}:Q" if pd.api.types.is_numeric_dtype(frame[column]) else f"{column}:N")
                for column in tooltip_columns
                if column in frame.columns
            ],
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )
    st.altair_chart(chart, use_container_width=True)


def render_history_chart(history_df: pd.DataFrame, selected_tickers: list[str], metric_name: str) -> None:
    chart_df = history_df[history_df["ticker"].isin(selected_tickers)].copy()
    if chart_df.empty:
        st.info("No historical rows matched those tickers.")
        return

    render_time_line_chart(
        chart_df,
        x_column="run_completed_at",
        y_column=metric_name,
        color_column="ticker",
        y_title=HISTORY_METRICS.get(metric_name, metric_name),
        x_title="Completed scrape",
        tooltip_columns=["run_label", "ticker", metric_name, "total_mentions", "acceleration_score"],
        height=380,
    )


def render_within_run_chart(bucket_df: pd.DataFrame, selected_tickers: list[str], metric_name: str) -> None:
    chart_df = build_dense_bucket_frame(bucket_df, selected_tickers, metric_name)
    if chart_df.empty:
        st.info("No bucketed rows matched those tickers.")
        return

    render_time_line_chart(
        chart_df,
        x_column="bucket_start_dt",
        y_column=metric_name,
        color_column="ticker",
        y_title=WITHIN_RUN_METRICS.get(metric_name, metric_name),
        x_title="Bucket start",
        tooltip_columns=["ticker", metric_name],
        height=380,
    )


def render_signal_map(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        st.info("No ticker summary rows are available.")
        return

    plot_df = summary_df.copy()
    plot_df["net_sentiment_score"] = plot_df["net_sentiment_score"].fillna(0)
    plot_df["total_mentions"] = plot_df["total_mentions"].clip(lower=0)
    label_df = plot_df.sort_values("emerging_ticker_score", ascending=False).head(12)
    mention_scale = alt.Scale(type="sqrt", zero=True, nice=True) if plot_df["total_mentions"].max() > 20 else alt.Scale(zero=True, nice=True)

    points = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.78, stroke="#ffffff", strokeWidth=1)
        .encode(
            x=alt.X("total_mentions:Q", title="Total mentions", scale=mention_scale),
            y=alt.Y("emerging_ticker_score:Q", title="Emerging ticker score", scale=alt.Scale(domain=[0, 100], nice=True)),
            size=alt.Size("source_diversity_score:Q", title="Source diversity", scale=alt.Scale(range=[80, 520])),
            color=alt.Color(
                "net_sentiment_score:Q",
                title="Net sentiment",
                scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
            ),
            tooltip=[
                "ticker:N",
                "company_name:N",
                "total_mentions:Q",
                "emerging_ticker_score:Q",
                "acceleration_score:Q",
                "average_alpha_signal_score:Q",
                "source_diversity_score:Q",
                "net_sentiment_score:Q",
            ],
        )
    )
    labels = (
        alt.Chart(label_df)
        .mark_text(align="left", baseline="middle", dx=8, fontSize=12)
        .encode(
            x="total_mentions:Q",
            y="emerging_ticker_score:Q",
            text="ticker:N",
        )
    )
    st.altair_chart((points + labels).properties(height=410).interactive(), use_container_width=True)


def render_acceleration_bar(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    bar_df = summary_df.sort_values("acceleration_score", ascending=False).head(12).sort_values("acceleration_score")
    chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X("acceleration_score:Q", title="Acceleration score", scale=alt.Scale(domain=[0, 100], nice=True)),
            y=alt.Y("ticker:N", title=None, sort=None),
            color=alt.Color(
                "average_hype_score:Q",
                title="Avg hype",
                scale=alt.Scale(scheme="tealblues", domain=[0, 10]),
            ),
            tooltip=[
                "ticker:N",
                "company_name:N",
                "acceleration_score:Q",
                "mentions_this_period:Q",
                "mentions_previous_period:Q",
                "average_hype_score:Q",
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def render_ticker_summary_table(summary_df: pd.DataFrame) -> None:
    display_source = summary_df.copy()
    watchlist = set(get_watchlist())
    display_source["watched"] = display_source["ticker"].astype(str).isin(watchlist)
    display_columns = [
        "watched",
        "ticker",
        "company_name",
        "total_mentions",
        "mentions_this_period",
        "mentions_previous_period",
        "mention_change",
        "acceleration_score",
        "emerging_ticker_score",
        "average_alpha_signal_score",
        "average_hype_score",
        "source_diversity_score",
        "controversy_score",
        "low_mentions_high_signal",
        "new_ticker_detected",
    ]
    available_columns = [column for column in display_columns if column in display_source.columns]
    st.dataframe(
        display_source[available_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "watched": st.column_config.CheckboxColumn("Watch"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "acceleration_score": st.column_config.NumberColumn("Acceleration", format="%.1f"),
            "emerging_ticker_score": st.column_config.NumberColumn("Emerging", format="%.1f"),
            "average_alpha_signal_score": st.column_config.NumberColumn("Alpha", format="%.1f"),
            "average_hype_score": st.column_config.NumberColumn("Hype", format="%.1f"),
            "source_diversity_score": st.column_config.NumberColumn("Source diversity", format="%.1f"),
            "controversy_score": st.column_config.NumberColumn("Controversy", format="%.1f"),
        },
    )


def render_results_dashboard_page(db_path: Path, run_id: str | None) -> None:
    st.header("Results Dashboard")
    st.caption("Use this page to review the actual Reddit evidence: which threads mentioned which tickers, how strong the signal looks, and what needs verification.")
    if not run_id:
        st.info("Run an analysis first.")
        return

    completed_notice = st.session_state.pop("completed_run_notice", None)
    if completed_notice:
        st.success(completed_notice)

    summary = load_run_summary(db_path, run_id)
    if summary:
        render_run_summary_cards(summary)
    render_run_metadata(db_path, run_id)
    runs = load_analysis_runs(db_path)
    run_rows = runs[runs["analysis_run_id"] == run_id]
    sources = load_sources(db_path)
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}
    if not run_rows.empty:
        render_source_health(
            db_path=db_path,
            run_id=run_id,
            run_record=run_rows.iloc[0].to_dict(),
            source_lookup=source_lookup,
        )

    long_df = load_run_mentions(db_path, run_id)
    if long_df.empty:
        if summary and int(summary.get("items_analyzed", 0)) == 0:
            st.warning("This run completed, but no Reddit items matched the selected sources and time window.")
        elif summary:
            st.info(
                f"This run analysed {int(summary.get('items_analyzed', 0))} Reddit items, "
                "but no catalog tickers or company-name matches were found."
            )
        else:
            st.info("No item-level mention data saved for this run.")
        return

    long_df = attach_quality_columns(long_df, load_run_classification_quality(db_path, run_id))
    summary_df = load_run_ticker_summaries(db_path, run_id)
    render_all_tickers_panel(summary_df, long_df)
    render_metric_guide("Metric guide for Results", expanded=False)
    filtered = filter_results_dataframe(long_df, run_id)
    st.write(f"{len(filtered)} mention rows matched the current filters.")
    if filtered.empty:
        render_filtered_empty_state(long_df)
        return

    display_columns = [
        "created_time",
        "source_name",
        "ticker",
        "company_name",
        "sentiment",
        "trust_status",
        "classifier_mode",
        "confidence",
        "alpha_signal_score",
        "hype_adjusted_signal_score",
        "depth_score",
        "hype_score",
        "evidence_quality",
        "claim_specificity_score",
        "source_diversity_score",
        "controversy_score",
        "repetition_score",
        "summary",
        "permalink",
    ]
    display_columns = [column for column in display_columns if column in filtered.columns]
    thread_tab, mention_tab, detail_tab = st.tabs(["Threads", "Mention Rows", "Details"])
    with thread_tab:
        thread_summary = build_thread_summary(filtered)
        if thread_summary.empty:
            st.info("No thread-level rows matched the current filters.")
        else:
            st.dataframe(
                thread_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "permalink": st.column_config.LinkColumn("permalink"),
                    "tickers": st.column_config.TextColumn("Tickers", width="medium"),
                    "companies": st.column_config.TextColumn("Companies", width="medium"),
                    "ticker_count": st.column_config.NumberColumn("# tickers", format="%d"),
                    "thread_title": st.column_config.TextColumn("Thread", width="large"),
                    "top_summary": st.column_config.TextColumn("Top summary", width="large"),
                    "max_alpha": st.column_config.NumberColumn("Max alpha", format="%.1f"),
                    "avg_confidence": st.column_config.NumberColumn("Avg confidence", format="%.2f"),
                    "avg_hype": st.column_config.NumberColumn("Avg hype", format="%.1f"),
                },
            )

    with mention_tab:
        display_df = filtered[display_columns].rename(columns={"source_name": "source"})
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "permalink": st.column_config.LinkColumn("permalink"),
                "trust_status": st.column_config.TextColumn("trust"),
                "classifier_mode": st.column_config.TextColumn("mode"),
            },
        )

    with detail_tab:
        render_mention_details(filtered)


def render_ticker_trends_page(db_path: Path, run_id: str | None) -> None:
    st.header("Ticker Trends")
    st.caption("Use this page to answer what changed between scrapes, which tickers are accelerating, and which leads deserve a closer read.")
    if not run_id:
        st.info("Run an analysis first.")
        return

    runs = load_analysis_runs(db_path)
    run_rows = runs[runs["analysis_run_id"] == run_id]
    if run_rows.empty:
        st.info("Selected run was not found.")
        return
    run_record = run_rows.iloc[0].to_dict()
    summary_df = load_run_ticker_summaries(db_path, run_id)
    bucket_df = load_run_time_buckets(db_path, run_id)
    long_df = load_run_mentions(db_path, run_id)
    if summary_df.empty:
        st.info("No ticker summary data saved for this run.")
        return

    sources = load_sources(db_path)
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}
    run_label_lookup = {
        str(row["analysis_run_id"]): build_run_display_label(row, source_lookup)
        for row in runs.to_dict(orient="records")
    }

    st.caption(
        f"Selected scrape: `{run_completed_label(run_record)}` · "
        f"{format_source_names(run_record, source_lookup)}"
    )
    metric_cols = st.columns(5)
    metric_cols[0].metric("Tickers", int(summary_df["ticker"].nunique()))
    metric_cols[1].metric("Mentions", int(summary_df["total_mentions"].sum()))
    metric_cols[2].metric("Top emerging", str(summary_df.iloc[0]["ticker"]))
    metric_cols[3].metric("Best acceleration", f"{float(summary_df['acceleration_score'].max()):.1f}")
    metric_cols[4].metric("Low-hype leads", int(summary_df["low_mentions_high_signal"].sum()))
    render_all_tickers_panel(summary_df, long_df)
    render_metric_guide("Metric guide for Trends", expanded=False)

    historical_summary_df = load_historical_ticker_summaries(
        db_path,
        data_mode=str(run_record["data_mode"]),
        selected_source_ids_json=json.dumps(run_record.get("selected_source_ids", []), ensure_ascii=True),
    )
    history_df = prepare_historical_summary_frame(historical_summary_df, run_label_lookup)
    previous_run_id = previous_history_run_id(history_df, run_id)
    previous_label = run_label_lookup.get(str(previous_run_id), str(previous_run_id)) if previous_run_id else None
    current_comparison_df = build_run_comparison_frame(
        history_df,
        baseline_run_id=str(previous_run_id),
        comparison_run_id=run_id,
    ) if previous_run_id else pd.DataFrame()

    render_change_summary(
        comparison_df=current_comparison_df,
        previous_label=previous_label,
    )
    render_source_health(
        db_path=db_path,
        run_id=run_id,
        run_record=run_record,
        source_lookup=source_lookup,
    )
    render_ticker_focus(
        run_id=run_id,
        summary_df=summary_df,
        bucket_df=bucket_df,
        long_df=long_df,
        history_df=history_df,
    )

    compare_tab, timeline_tab, map_tab, table_tab = st.tabs(
        ["Compare Scrapes", "Timeline", "Signal Map", "Summary Table"]
    )

    with compare_tab:
        st.subheader("Run-over-run comparison")
        if history_df.empty:
            st.info("No completed ticker history exists yet for this same source selection.")
        else:
            history_runs = (
                history_df[["analysis_run_id", "run_completed_at", "run_label"]]
                .drop_duplicates("analysis_run_id")
                .sort_values("run_completed_at")
            )
            history_run_ids = history_runs["analysis_run_id"].astype(str).tolist()
            comparison_default = run_id if run_id in history_run_ids else history_run_ids[-1]
            comparison_index = history_run_ids.index(comparison_default)
            baseline_index = max(comparison_index - 1, 0)

            controls = st.columns([1.2, 1.2, 1])
            baseline_run_id = controls[0].selectbox(
                "Baseline scrape",
                history_run_ids,
                index=baseline_index,
                format_func=lambda value: run_label_lookup.get(str(value), str(value)),
                key=f"baseline_run_{run_id}",
            )
            comparison_run_id = controls[1].selectbox(
                "Compare scrape",
                history_run_ids,
                index=comparison_index,
                format_func=lambda value: run_label_lookup.get(str(value), str(value)),
                key=f"comparison_run_{run_id}",
            )
            history_metric = controls[2].selectbox(
                "Chart metric",
                list(HISTORY_METRICS.keys()),
                format_func=lambda value: HISTORY_METRICS[value],
                key=f"history_metric_{run_id}",
            )

            default_history_tickers = [
                ticker for ticker in summary_df["ticker"].head(6).tolist()
                if ticker in set(history_df["ticker"].tolist())
            ]
            history_tickers = sorted(history_df["ticker"].dropna().unique().tolist())
            selected_history_tickers = st.multiselect(
                "Tickers in comparison chart",
                history_tickers,
                default=default_history_tickers[: min(6, len(default_history_tickers))],
                key=f"history_tickers_{run_id}",
            )
            if selected_history_tickers:
                render_history_chart(history_df, selected_history_tickers, history_metric)
            else:
                st.info("Select at least one ticker to draw the comparison chart.")

            comparison_df = build_run_comparison_frame(
                history_df,
                baseline_run_id=str(baseline_run_id),
                comparison_run_id=str(comparison_run_id),
            )
            if len(history_run_ids) < 2:
                st.caption("Only one completed scrape exists for this source set. New scrapes will appear here automatically.")
            elif comparison_df.empty:
                st.info("No overlapping ticker rows were available for those two scrapes.")
            else:
                delta_cols = [
                    "ticker",
                    "movement",
                    "company_name",
                    "total_mentions_baseline",
                    "total_mentions_current",
                    "total_mentions_delta",
                    "acceleration_score_baseline",
                    "acceleration_score_current",
                    "acceleration_score_delta",
                    "emerging_ticker_score_current",
                    "emerging_ticker_score_delta",
                    "average_alpha_signal_score_current",
                    "average_alpha_signal_score_delta",
                ]
                available_delta_cols = [column for column in delta_cols if column in comparison_df.columns]
                st.dataframe(
                    comparison_df[available_delta_cols].head(25),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "movement": st.column_config.TextColumn("Movement", width="medium"),
                        "company_name": st.column_config.TextColumn("Company", width="medium"),
                        "total_mentions_baseline": st.column_config.NumberColumn("Base mentions", format="%d"),
                        "total_mentions_current": st.column_config.NumberColumn("Current mentions", format="%d"),
                        "total_mentions_delta": st.column_config.NumberColumn("Mention change", format="%+d"),
                        "acceleration_score_baseline": st.column_config.NumberColumn("Base accel.", format="%.1f"),
                        "acceleration_score_current": st.column_config.NumberColumn("Current accel.", format="%.1f"),
                        "acceleration_score_delta": st.column_config.NumberColumn("Accel. change", format="%+.1f"),
                        "emerging_ticker_score_current": st.column_config.NumberColumn("Current emerging", format="%.1f"),
                        "emerging_ticker_score_delta": st.column_config.NumberColumn("Emerging change", format="%+.1f"),
                        "average_alpha_signal_score_current": st.column_config.NumberColumn("Current alpha", format="%.1f"),
                        "average_alpha_signal_score_delta": st.column_config.NumberColumn("Alpha change", format="%+.1f"),
                    },
                )

    with timeline_tab:
        st.subheader("Selected scrape timeline")
        if bucket_df.empty:
            st.info("No time buckets are available for this run.")
        else:
            timeline_controls = st.columns([2, 1])
            selected_tickers = timeline_controls[0].multiselect(
                "Tickers",
                summary_df["ticker"].tolist(),
                default=summary_df["ticker"].head(min(6, len(summary_df))).tolist(),
                key=f"timeline_tickers_{run_id}",
            )
            timeline_metric = timeline_controls[1].selectbox(
                "Metric",
                list(WITHIN_RUN_METRICS.keys()),
                format_func=lambda value: WITHIN_RUN_METRICS[value],
                key=f"timeline_metric_{run_id}",
            )
            if selected_tickers:
                render_within_run_chart(bucket_df, selected_tickers, timeline_metric)
            else:
                st.info("Select at least one ticker to draw the timeline.")

            spike_notes = build_spike_notes(bucket_df, long_df, selected_tickers)
            if not spike_notes.empty:
                with st.expander("Spike notes", expanded=False):
                    st.dataframe(
                        spike_notes,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "permalink": st.column_config.LinkColumn("permalink"),
                            "likely_driver": st.column_config.TextColumn("Likely driver", width="large"),
                            "summary": st.column_config.TextColumn("Summary", width="large"),
                        },
                    )

            if not long_df.empty:
                sentiment_df = (
                    long_df[long_df["ticker"].isin(selected_tickers)]
                    .groupby(["ticker", "time_bucket"], as_index=False)["sentiment_numeric"]
                    .mean()
                    .rename(columns={"sentiment_numeric": "net_sentiment_score", "time_bucket": "bucket_start_dt"})
                )
                sentiment_df["bucket_start_dt"] = pd.to_datetime(sentiment_df["bucket_start_dt"], errors="coerce", utc=True)
                if not sentiment_df.empty:
                    st.write("Average sentiment by bucket")
                    render_time_line_chart(
                        sentiment_df,
                        x_column="bucket_start_dt",
                        y_column="net_sentiment_score",
                        color_column="ticker",
                        y_title="Net sentiment",
                        x_title="Bucket start",
                        tooltip_columns=["ticker", "net_sentiment_score"],
                        height=280,
                    )

    with map_tab:
        st.subheader("Ticker signal overview")
        map_cols = st.columns([1.65, 1])
        with map_cols[0]:
            render_signal_map(summary_df)
        with map_cols[1]:
            st.write("Top acceleration")
            render_acceleration_bar(summary_df)

        focus_df = summary_df[summary_df["low_mentions_high_signal"] | summary_df["new_ticker_detected"]]
        if not focus_df.empty:
            st.write("Emerging and low-hype watchlist")
            st.dataframe(
                focus_df[
                    [
                        "ticker",
                        "company_name",
                        "total_mentions",
                        "average_alpha_signal_score",
                        "average_hype_score",
                        "acceleration_score",
                        "emerging_ticker_score",
                        "low_mentions_high_signal",
                        "new_ticker_detected",
                    ]
                ].head(20),
                use_container_width=True,
                hide_index=True,
            )

    with table_tab:
        st.subheader("Ticker summary")
        render_ticker_summary_table(summary_df)


def render_export_page(db_path: Path, run_id: str | None) -> None:
    st.header("Export Data")
    if not run_id:
        st.info("Run an analysis first.")
        return

    long_df = load_run_mentions(db_path, run_id)
    summary_df = load_run_ticker_summaries(db_path, run_id)
    bucket_df = load_run_time_buckets(db_path, run_id)
    summary_json = load_run_summary(db_path, run_id)

    if long_df.empty:
        st.info("No data available for export in this run.")
        return

    run_artifact_dir = DEFAULT_ARTIFACTS_DIR / run_id
    if run_artifact_dir.exists():
        st.caption(f"Automatic run artifacts saved to `{run_artifact_dir}`.")

    st.write("Long format: one row per analysed item and ticker mention. Wide format: a pivoted table for statistical tools like JASP.")
    st.download_button(
        "Download raw item-level long format CSV",
        long_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_long_format.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download ticker-level summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_ticker_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download time-bucketed ticker trend CSV",
        bucket_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_ticker_trends.csv",
        mime="text/csv",
    )

    st.subheader("Wide format export")
    wide_cols = st.columns(3)
    index_var = wide_cols[0].selectbox("Index variable", ["ticker", "time_bucket", "source_name", "created_date"])
    columns_var = wide_cols[1].selectbox("Columns variable", ["sentiment", "subreddit", "source_name", "ticker"])
    numeric_candidates = [column for column in long_df.columns if pd.api.types.is_numeric_dtype(long_df[column])]
    values_var = wide_cols[2].selectbox("Values variable", numeric_candidates, index=numeric_candidates.index("alpha_signal_score"))
    wide_df = long_df.pivot_table(index=index_var, columns=columns_var, values=values_var, aggfunc="mean").reset_index()
    st.dataframe(wide_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download wide format CSV",
        wide_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_wide_format.csv",
        mime="text/csv",
    )

    full_payload = {
        "analysis_run_id": run_id,
        "run_summary": summary_json,
        "long_format": load_full_results_records(db_path, run_id),
        "ticker_summaries": summary_df.to_dict(orient="records"),
        "ticker_time_buckets": bucket_df.to_dict(orient="records"),
    }
    st.download_button(
        "Download JSON export of full classified results",
        json.dumps(full_payload, indent=2).encode("utf-8"),
        file_name=f"{run_id}_full_results.json",
        mime="application/json",
    )


def render_settings_page(db_path: Path, ticker_path: Path, artifacts_dir: Path) -> None:
    st.header("Settings")
    overview = db_overview(db_path)
    overview_cols = st.columns(4)
    overview_cols[0].metric("Sources", overview["sources"])
    overview_cols[1].metric("Analysis runs", overview["analysis_runs"])
    overview_cols[2].metric("Reddit items", overview["reddit_items"])
    overview_cols[3].metric("Mention rows", overview["item_ticker_mentions"])

    settings_tab, guide_tab, runs_tab, prefs_tab = st.tabs(["Diagnostics", "Metric Guide", "Runs", "Preferences"])

    with settings_tab:
        st.subheader("Environment")
        env_cols = st.columns(4)
        env_cols[0].metric("Streamlit", st.__version__)
        env_cols[1].metric("Altair", alt.__version__)
        env_cols[2].metric("Pandas", pd.__version__)
        env_cols[3].metric("Requests", requests.__version__)

        st.write("Collection uses public Reddit RSS feeds and stores runs locally.")

        st.subheader("Paths")
        st.code(
            "\n".join(
                [
                    f"db_path = {db_path}",
                    f"artifacts_dir = {artifacts_dir}",
                    f"ui_preferences = {DEFAULT_UI_PREFS_PATH}",
                    f"storage_root = {DEFAULT_STORAGE_ROOT}",
                    f"ticker_catalog_path = {ticker_path}",
                    f"ollama_url = {DEFAULT_OLLAMA_URL}",
                    f"default_model = {DEFAULT_MODEL}",
                    f"reddit_user_agent = {os.getenv('REDDIT_USER_AGENT', 'script:signal-from-the-slop:0.1')}",
                ]
            )
        )

    with guide_tab:
        st.subheader("Scoring and table glossary")
        st.dataframe(
            pd.DataFrame(METRIC_GUIDE),
            use_container_width=True,
            hide_index=True,
            column_config={
                "metric": st.column_config.TextColumn("Metric", width="medium"),
                "plain_english": st.column_config.TextColumn("Plain English", width="large"),
                "how_it_is_determined": st.column_config.TextColumn("How it is determined", width="large"),
            },
        )

    with runs_tab:
        st.subheader("Run health")
        runs = load_analysis_runs(db_path)
        if not runs.empty:
            runs = runs[runs["data_mode"] == "live"].copy()
        if runs.empty:
            st.info("No runs have been saved yet.")
        else:
            run_health = runs.copy()
            run_health["items_collected"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_collected", (value or {}).get("items_analyzed", 0)) or 0))
            run_health["items_analyzed"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_analyzed", 0)))
            run_health["ticker_mentions"] = run_health["summary_json"].map(lambda value: int((value or {}).get("ticker_mentions", 0)))
            run_health["tickers_found"] = run_health["summary_json"].map(lambda value: int((value or {}).get("tickers_found", 0)))
            run_health["fallback_count"] = run_health["summary_json"].map(lambda value: int((value or {}).get("fallback_count", 0)))
            run_health["error_message"] = run_health["summary_json"].map(lambda value: str((value or {}).get("error_message", "")))
            run_health["elapsed_minutes"] = run_health.apply(lambda row: run_elapsed_minutes(row.to_dict()) or 0, axis=1)
            run_health["elapsed"] = run_health.apply(lambda row: run_elapsed_label(row.to_dict()) if row.get("status") == "running" else "", axis=1)
            status_counts = run_health["status"].value_counts().to_dict()
            health_cols = st.columns(3)
            health_cols[0].metric("Completed", int(status_counts.get("completed", 0)))
            health_cols[1].metric("Running", int(status_counts.get("running", 0)))
            health_cols[2].metric("Failed", int(status_counts.get("failed", 0)))
            stale_running = run_health[
                (run_health["status"] == "running")
                & (run_health["elapsed_minutes"] >= 30)
            ].copy()
            if not stale_running.empty:
                st.warning(f"{len(stale_running)} run(s) have been marked running for 30+ minutes.")
                if st.button("Mark stale running runs as failed", use_container_width=True):
                    for row in stale_running.to_dict(orient="records"):
                        fail_analysis_run(
                            db_path,
                            str(row["analysis_run_id"]),
                            {
                                "items_collected": int(row.get("items_collected", 0)),
                                "items_analyzed": int(row.get("items_analyzed", 0)),
                                "ticker_mentions": int(row.get("ticker_mentions", 0)),
                                "tickers_found": int(row.get("tickers_found", 0)),
                                "bullish_items": 0,
                                "bearish_items": 0,
                                "neutral_items": 0,
                                "irrelevant_items": 0,
                                "top_mentioned_tickers": [],
                                "top_accelerating_tickers": [],
                                "high_depth_posts_found": 0,
                                "low_mentions_high_signal_count": 0,
                                "newly_detected_tickers_count": 0,
                                "fallback_count": int(row.get("fallback_count", 0)),
                                "failed_stage": "stale_running_cleanup",
                                "error_message": "Marked failed from Run health after remaining in running status for 30+ minutes.",
                            },
                        )
                    st.rerun()
            st.dataframe(
                run_health[
                    [
                        "analysis_run_id",
                        "status",
                        "started_at",
                        "completed_at",
                        "elapsed",
                        "time_window_label",
                        "items_collected",
                        "items_analyzed",
                        "ticker_mentions",
                        "tickers_found",
                        "fallback_count",
                        "error_message",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "analysis_run_id": st.column_config.TextColumn("Run"),
                    "items_collected": st.column_config.NumberColumn("Collected", format="%d"),
                    "items_analyzed": st.column_config.NumberColumn("Analysed", format="%d"),
                    "ticker_mentions": st.column_config.NumberColumn("Mentions", format="%d"),
                    "tickers_found": st.column_config.NumberColumn("Tickers", format="%d"),
                    "fallback_count": st.column_config.NumberColumn("Fallback", format="%d"),
                    "error_message": st.column_config.TextColumn("Error", width="large"),
                },
            )

    with prefs_tab:
        st.subheader("Local preferences")
        prefs = load_ui_preferences()
        render_copyable_ticker_chips(
            prefs.get("watchlist", []),
            label="Watchlist",
            help_text="Click to copy ticker.",
            max_items=80,
        )
        st.json(prefs)
        if st.button("Clear watchlist", type="secondary"):
            prefs["watchlist"] = []
            save_ui_preferences(prefs)
            st.success("Watchlist cleared.")
            st.rerun()


init_db(DEFAULT_DB_PATH, SCHEMA_PATH)
ensure_seed_sources(DEFAULT_DB_PATH, DEFAULT_SOURCES_PATH)
page, selected_run_id, _, _ = build_sidebar_state(DEFAULT_DB_PATH)

if page == "Sources":
    render_sources_page(DEFAULT_DB_PATH)
elif page == "Run Analysis":
    render_run_analysis_page(
        db_path=DEFAULT_DB_PATH,
        artifacts_dir=DEFAULT_ARTIFACTS_DIR,
        ticker_path=DEFAULT_TICKER_PATH,
    )
elif page == "Results Dashboard":
    render_results_dashboard_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Ticker Trends":
    render_ticker_trends_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Export Data":
    render_export_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Settings":
    render_settings_page(DEFAULT_DB_PATH, DEFAULT_TICKER_PATH, DEFAULT_ARTIFACTS_DIR)
