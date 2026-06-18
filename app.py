from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time as time_module
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
    init_db,
    latest_analysis_run_id,
    load_analysis_runs,
    load_completed_mentions,
    load_full_results_records,
    load_run_classification_quality,
    load_historical_ticker_summaries,
    load_run_mentions,
    load_run_summary,
    load_run_source_activity,
    load_run_source_coverage,
    load_run_ticker_summaries,
    load_run_time_buckets,
    load_signal_attention_outcomes,
    load_signal_events,
    load_signal_labels,
    load_signal_outcomes,
    load_signal_random_benchmarks,
    load_sources,
    load_ticker_consensus_cache,
    load_ticker_history,
    replace_run_classifications,
    replace_run_mentions,
    replace_run_source_coverage,
    replace_run_ticker_summaries,
    replace_run_time_buckets,
    upsert_signal_label,
    update_sources,
    upsert_ticker_consensus_cache,
    upsert_reddit_items,
)
try:
    from signal_from_the_slop.database import fail_analysis_run
except ImportError:
    def fail_analysis_run(db_path: Path, analysis_run_id: str, summary: dict[str, Any]) -> None:
        complete_analysis_run(db_path, analysis_run_id, summary)

try:
    from signal_from_the_slop.database import load_previously_analyzed_item_ids
except ImportError:
    def load_previously_analyzed_item_ids(db_path: Path, *, data_mode: str = "live") -> set[str]:
        import sqlite3

        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT c.item_id
                    FROM classifications c
                    JOIN analysis_runs ar
                        ON ar.analysis_run_id = c.analysis_run_id
                    WHERE ar.status = 'completed'
                      AND ar.data_mode = ?
                    """,
                    (data_mode,),
                ).fetchall()
        except sqlite3.Error:
            return set()
        return {str(row[0]) for row in rows}

from signal_from_the_slop.ollama_classifier import OllamaClassifier
from signal_from_the_slop.reddit_client import (
    REDDIT_SUBREDDIT_RSS_LIMIT,
    RedditClient,
    build_subreddit_source,
    parse_source_input,
)
from signal_from_the_slop.signal_validation import refresh_signal_validation, run_signal_validation
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
    {"page": "Today", "label": "Today", "step": "01", "context": "Current state"},
    {"page": "Collect", "label": "Collect", "step": "02", "context": "Sources and runs"},
    {"page": "Review", "label": "Review", "step": "03", "context": "Signals and evidence"},
    {"page": "Validate", "label": "Validate", "step": "04", "context": "Frozen outcomes"},
    {"page": "Library", "label": "Library", "step": "05", "context": "Runs and exports"},
    {"page": "Settings", "label": "Settings", "step": "06", "context": "Technical setup"},
]
PAGES = [step["page"] for step in PAGE_STEPS]
PAGE_LABELS = {step["page"]: step["label"] for step in PAGE_STEPS}
PAGE_CONTEXT = {step["page"]: step["context"] for step in PAGE_STEPS}
PAGE_ALIASES = {
    "Sources": "Collect",
    "Run Analysis": "Collect",
    "Results Dashboard": "Review",
    "Ticker Explorer": "Review",
    "Ticker Trends": "Review",
    "Signal Validation": "Validate",
    "Export Data": "Library",
}
ACTIVE_RUN_STATUSES = {"running", "pausing", "paused", "cancelling"}
CORPUS_SCOPE_ID = "__combined_completed_corpus__"
CORPUS_SCOPE_LABEL = "Combined corpus (all completed runs)"
RUN_TYPE_DISCOVERY = "discovery"
RUN_TYPE_MEASUREMENT = "measurement"
RUN_TYPE_LABELS = {
    RUN_TYPE_MEASUREMENT: "Measurement",
    RUN_TYPE_DISCOVERY: "Discovery",
}
RUN_CONTROL_POLL_SECONDS = 0.5
RUN_HEARTBEAT_STALE_SECONDS = 45
RUN_ORPHANED_CANCEL_SECONDS = 90
RUN_WORKERS: dict[str, threading.Thread] = {}
RUN_WORKERS_LOCK = threading.Lock()
TICKER_EXPLORER_WINDOWS = {
    "Last 7d": 7,
    "Last 30d": 30,
    "Last 90d": 90,
    "All": None,
}
TICKER_CONSENSUS_EVIDENCE_LIMIT = 10

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
SIGNAL_LABEL_OPTIONS = [
    "Useful lead",
    "Worth deeper research",
    "Interesting but no action",
    "Pure hype",
    "Already obvious",
    "Bad ticker extraction",
    "False positive",
    "Avoided hype trap",
    "Not enough information",
]
QUICK_SIGNAL_LABELS = [
    "Useful lead",
    "Worth deeper research",
    "Pure hype",
    "Avoided hype trap",
    "False positive",
]
METRIC_TOOLTIPS = {
    "emerging_ticker_score": "Lead-ranking score. Blends acceleration, source diversity, alpha quality, evidence quality, specificity, and penalties for hype or repetition.",
    "acceleration_score": "Coverage-adjusted acceleration. Compares mention, thread, and author rates against a matched Measurement baseline, then applies coverage and hype penalties.",
    "adjusted_acceleration_score": "Frozen signal copy of coverage-adjusted acceleration at the time the signal was saved.",
    "coverage_reliability": "0 to 1 score for how much to trust the trend sample. It reflects run type, source coverage, scrape caps, rate limits, and baseline availability.",
    "attention_spread_14d": "Post-signal Reddit attention score over 14 days. Higher means the ticker spread across more mentions, threads, authors, sources, and share of voice.",
    "attention_spread_7d": "Post-signal Reddit attention score over 7 days.",
    "attention_spread_30d": "Post-signal Reddit attention score over 30 days.",
    "share_of_voice": "Ticker mentions divided by all ticker mentions in the same scope, expressed as a percentage.",
}
METRIC_GUIDE = [
    {
        "metric": "Mentions",
        "contexts": ["results", "trends"],
        "formula": "mentions = COUNT(item_ticker_mentions WHERE ticker = T)",
        "plain_english": "How many analysed item/ticker rows matched this ticker in the selected run, bucket, or comparison view.",
        "how_it_is_determined": "Each classified Reddit post or comment can create one row per detected ticker.",
        "why_caution": "Raw count alone can still be low-quality meme noise. Read it next to hype, source diversity, and acceleration.",
        "example": "One post mentioning BE and MSTR creates one BE mention row and one MSTR mention row.",
    },
    {
        "metric": "Thread mentions",
        "contexts": ["results"],
        "formula": "thread_mentions = COUNT(rows grouped by thread_id)",
        "plain_english": "How many matching ticker mentions came from the same Reddit thread.",
        "how_it_is_determined": "The app groups rows by thread so multi-comment pile-ons do not look like separate independent theses.",
        "why_caution": "A huge thread can dominate counts without proving broad attention across many sources or authors.",
        "example": "Twenty comments in one hot thread can still be just one thread-level attention event.",
    },
    {
        "metric": "Confidence",
        "contexts": ["results"],
        "formula": "confidence ∈ [0, 1]",
        "plain_english": "How confident the classifier was about the ticker and sentiment interpretation.",
        "how_it_is_determined": "Returned by Ollama or fallback heuristics as a bounded 0 to 1 score.",
        "why_caution": "Confidence is not truth. A confident reading can still be wrong if the text is ambiguous or sarcastic.",
        "example": "0.88 means the model felt fairly sure it understood the post; it does not mean the thesis itself is correct.",
    },
    {
        "metric": "Alpha signal",
        "contexts": ["results", "trends"],
        "formula": "alpha = reddit_component + depth*2.5 + quality_weight + confidence*10 + claim_specificity*3.2 + bonuses - hype_penalties - red_flag_penalties",
        "plain_english": "A quality score for potentially useful, researchable information in the post itself.",
        "how_it_is_determined": "The scoring module adds evidence quality, depth, confidence, specificity, and some context bonuses, then subtracts hype, no-evidence, and red-flag penalties.",
        "why_caution": "Alpha measures research usefulness, not future returns. A strong alpha post can still be early, wrong, or already priced in.",
        "example": "A post with concrete revenue numbers, risks, and filing references scores better than 'this stock is going vertical.'",
    },
    {
        "metric": "Max alpha",
        "contexts": ["results"],
        "formula": "max_alpha = MAX(alpha_signal_score within grouped thread)",
        "plain_english": "The strongest single alpha signal found inside a grouped thread.",
        "how_it_is_determined": "The thread view takes the highest alpha mention inside that thread group.",
        "why_caution": "One strong comment can make the thread look better than the rest of the discussion really is.",
        "example": "If one comment is well argued but the other 30 are low-effort reactions, max alpha still reflects the best one.",
    },
    {
        "metric": "Hype",
        "contexts": ["results", "trends"],
        "formula": "hype_score ∈ [0, 10]",
        "plain_english": "How meme-like, hyperbolic, repetitive, or promotional the text looks.",
        "how_it_is_determined": "The classifier returns a 0 to 10 hype score. Downstream scoring also penalizes hyperbolic language, repeated phrases, and no-evidence language.",
        "why_caution": "High hype often inflates mention counts without improving thesis quality. It is a warning flag, not an automatic disqualifier.",
        "example": "Phrases like 'to the moon', all-caps emphasis, or repetitive slogan posting should push the hype interpretation higher.",
    },
    {
        "metric": "Hype-adjusted signal",
        "contexts": ["results", "trends"],
        "formula": "hype_adjusted = clamp(alpha_signal_score - hype_score*4.5 - repetition_score*1.5, 0, 100)",
        "plain_english": "Signal quality after explicitly penalizing hype and repetition.",
        "how_it_is_determined": "The app starts from alpha signal, then subtracts hype and repetition penalties and clamps the result to 0 to 100.",
        "why_caution": "This is stricter than alpha. A flashy post can have some useful details, but still get dragged down here for being too promotional.",
        "example": "A decent thesis with hype 8 and repetition 6 gets cut materially compared with the raw alpha score.",
    },
    {
        "metric": "Coverage-adjusted acceleration",
        "contexts": ["trends"],
        "formula": "mention_growth = bounded_rate_growth(curr_mention_rate, prev_mention_rate)\nthread_growth = bounded_rate_growth(curr_thread_rate, prev_thread_rate)\nauthor_growth = bounded_rate_growth(curr_author_rate, prev_author_rate)\ngrowth_blend = 0.45*mention_growth + 0.30*thread_growth + 0.25*author_growth\nraw = clamp(growth_blend * 35, 0, 100)\nfinal = raw * reliability * hype_penalty * repetition_penalty",
        "plain_english": "Whether attention is actually accelerating across comparable scrape coverage, not just rising in raw counts.",
        "how_it_is_determined": "Measurement runs compare current mention, thread, and author rates against the previous matched Measurement run, then penalize weak coverage, hype, repetition, and missing baselines.",
        "why_caution": "This only means something for matched Measurement runs. Discovery runs and mixed corpus views intentionally zero it out.",
        "example": "If mention rate rises but it all came from one thread and weak coverage, acceleration should stay muted.",
    },
    {
        "metric": "Trend reliability",
        "contexts": ["trends", "signals"],
        "formula": "rules-based label from run type, coverage reliability, post count, RSS caps, rate limits, and matched baseline availability",
        "plain_english": "How much trust to put in the trend or acceleration view.",
        "how_it_is_determined": "The app checks run type, sample size, RSS caps, rate limits, source coverage, and whether a matched Measurement baseline exists.",
        "why_caution": "A low label means 'be careful with this trend interpretation', not necessarily 'the ticker is bad.'",
        "example": "A Measurement run with a baseline and good source coverage can show High reliability; a Discovery run will not.",
    },
    {
        "metric": "Mention rate",
        "contexts": ["trends"],
        "formula": "mention_rate_per_100_posts = total_mentions / max(total_posts_scraped, 1) * 100",
        "plain_english": "Ticker mentions normalized by scrape depth.",
        "how_it_is_determined": "The app divides ticker mentions by total posts scraped and expresses it as mentions per 100 posts.",
        "why_caution": "A rate is more comparable than a raw count, but still depends on source quality and whether the run was Discovery or Measurement.",
        "example": "10 mentions from 50 posts = 20 mentions per 100 posts.",
    },
    {
        "metric": "Thread rate",
        "contexts": ["trends"],
        "formula": "thread_rate_per_100_posts = unique_threads / max(total_threads_scraped, 1) * 100",
        "plain_english": "How many unique threads featured the ticker relative to scrape depth.",
        "how_it_is_determined": "Uses unique thread count instead of raw mention rows so one noisy thread does not dominate as much.",
        "why_caution": "Thread rate is usually cleaner than mention count, but it still does not guarantee independent conviction.",
        "example": "5 unique threads out of 40 scraped threads = 12.5 thread mentions per 100 threads.",
    },
    {
        "metric": "Author rate",
        "contexts": ["trends"],
        "formula": "author_rate_per_100_posts = unique_authors / max(total_authors_scraped, 1) * 100",
        "plain_english": "How many unique authors mentioned the ticker relative to scrape depth.",
        "how_it_is_determined": "Counts distinct author hashes and normalizes by authors seen in the scrape.",
        "why_caution": "Useful for judging breadth, but coordinated posters or one-time throwaway accounts can still distort it.",
        "example": "7 unique authors out of 80 authors scraped = 8.75 per 100 authors.",
    },
    {
        "metric": "Emerging ticker score",
        "contexts": ["trends", "signals"],
        "formula": "emerging = accel*0.45 + source_diversity*2.8 + alpha*0.28 + evidence_quality*6 + claim_specificity*2.5 + low_volume_bonus + multi_source_bonus + flags - hype*2.2 - repetition*1.4 - mega_popularity_penalty",
        "plain_english": "The app’s main lead-ranking score for tickers that may deserve follow-up.",
        "how_it_is_determined": "It blends acceleration, source diversity, alpha quality, evidence quality, specificity, and bonuses for cleaner low-volume discovery, then subtracts hype, repetition, and over-popularity penalties.",
        "why_caution": "This is a ranking heuristic, not a valuation model. High emerging score means 'read this first', not 'buy this'.",
        "example": "A small-cap name with improving acceleration, decent evidence, and low hype can outrank a mega-cap that is merely popular.",
    },
    {
        "metric": "Source diversity",
        "contexts": ["results", "trends"],
        "formula": "source_diversity = min(source_count*2.5 + subreddit_count*1.5, 10)",
        "plain_english": "Whether discussion is coming from more than one source or subreddit.",
        "how_it_is_determined": "The app scores unique source and subreddit coverage on a capped 0 to 10 scale.",
        "why_caution": "More sources are usually better than one thread, but low-quality copycat chatter can still spread everywhere.",
        "example": "Mentions in r/stocks plus r/wallstreetbets usually score better than everything coming from a single thread URL.",
    },
    {
        "metric": "Controversy",
        "contexts": ["results", "trends"],
        "formula": "controversy = (2 * min(bullish_mentions, bearish_mentions) / (bullish_mentions + bearish_mentions)) * 10",
        "plain_english": "How split bullish and bearish mentions are.",
        "how_it_is_determined": "Balanced bullish and bearish counts push the score higher. One-sided discussion pushes it toward zero.",
        "why_caution": "High controversy means disagreement, not necessarily value. Sometimes it reflects genuine debate; sometimes it reflects confusion.",
        "example": "10 bullish and 10 bearish mentions gives high controversy; 20 bullish and 0 bearish gives 0.",
    },
    {
        "metric": "Trust",
        "contexts": ["results"],
        "formula": "rules-based label from classifier_mode, confidence, evidence_quality, and hype_score",
        "plain_english": "A quick label for how much attention to give the row before reading it closely.",
        "how_it_is_determined": "Rows become Strong, Standard, Review, or Fallback depending on classifier mode, confidence, evidence quality, and hype.",
        "why_caution": "It is a triage aid, not a truth score. A Review row can still hide a useful lead, and a Strong row can still be wrong.",
        "example": "Fallback means Ollama did not return usable JSON, so the app used deterministic heuristics instead.",
    },
    {
        "metric": "Attention spread",
        "contexts": ["signals"],
        "formula": "attention_spread = min(100, future_mentions*3.5 + future_unique_threads*5 + future_unique_authors*2.5 + future_source_count*10 + future_share_of_voice*0.35)",
        "plain_english": "How much broader Reddit attention became after the signal was frozen.",
        "how_it_is_determined": "The app only looks at Reddit data collected after signal time, then scores growth across mentions, threads, authors, sources, and future share of voice.",
        "why_caution": "Attention spread is a research-utility outcome, not a market-return outcome. A ticker can spread on Reddit and still be a bad trade.",
        "example": "If a name jumps from one thread to multiple subreddits and many new authors, attention spread should rise sharply.",
    },
    {
        "metric": "Share of voice",
        "contexts": ["signals"],
        "formula": "share_of_voice = ticker_mentions / all_scope_mentions * 100",
        "plain_english": "The ticker’s share of all ticker mention rows in the current scope or future window.",
        "how_it_is_determined": "The app divides the ticker’s mention count by the total mention count in that scope and expresses it as a percentage.",
        "why_caution": "A high share of voice can just mean one ticker dominated a very small or low-quality sample.",
        "example": "If a ticker gets 12 mention rows out of 120 total rows, share of voice is 10%.",
    },
    {
        "metric": "Excess return",
        "contexts": ["signals"],
        "formula": "excess_return_Nd = ticker_return_Nd - benchmark_return_Nd",
        "plain_english": "How much the ticker outperformed or underperformed the benchmark after the signal.",
        "how_it_is_determined": "Forward ticker returns are compared against SPY by default over the same window.",
        "why_caution": "Positive excess return can still come with ugly drawdowns, and negative excess return can still have been a useful research warning.",
        "example": "If the ticker returned 9% in 7 days and SPY returned 2%, excess return is +7%.",
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
        .global-app-bar {
            align-items: center;
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
            display: grid;
            gap: 0.75rem;
            grid-template-columns: minmax(230px, 1.1fr) minmax(320px, 2fr);
            margin-bottom: 0.75rem;
            padding-bottom: 0.8rem;
        }
        .global-app-title {
            font-size: 1.2rem;
            font-weight: 780;
            line-height: 1.18;
        }
        .global-app-subtitle {
            font-size: 0.84rem;
            margin-top: 0.15rem;
            opacity: 0.72;
        }
        .global-app-meta {
            display: grid;
            gap: 0.45rem;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
        }
        .global-app-cell {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 8px;
            padding: 0.48rem 0.58rem;
        }
        .global-app-label {
            font-size: 0.7rem;
            font-weight: 700;
            opacity: 0.68;
            text-transform: uppercase;
        }
        .global-app-value {
            font-size: 0.9rem;
            font-weight: 720;
            line-height: 1.25;
            margin-top: 0.12rem;
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
        .ticker-hero {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(128, 128, 128, 0.08), rgba(128, 128, 128, 0.03));
            margin: 0.35rem 0 0.85rem;
            padding: 0.9rem 1rem;
        }
        .ticker-hero-title {
            font-size: 1.65rem;
            font-weight: 780;
            line-height: 1.1;
            margin: 0;
        }
        .ticker-hero-meta {
            font-size: 0.9rem;
            margin-top: 0.25rem;
            opacity: 0.74;
        }
        .consensus-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            background: rgba(128, 128, 128, 0.045);
            margin: 0.4rem 0 0.9rem;
            padding: 0.95rem 1rem;
        }
        .consensus-card h3 {
            font-size: 1.05rem;
            margin: 0 0 0.45rem;
        }
        .consensus-card p {
            margin-bottom: 0.35rem;
        }
        .compact-list {
            margin: 0.15rem 0 0.6rem 1.05rem;
            padding: 0;
        }
        .compact-list li {
            margin: 0.18rem 0;
        }
        .top-signal-title {
            font-size: 0.92rem;
            font-weight: 760;
            line-height: 1.2;
            margin-bottom: 0.15rem;
        }
        .top-signal-score {
            font-size: 1.2rem;
            font-weight: 780;
            line-height: 1.15;
            margin: 0.2rem 0 0.35rem;
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
        .scope-context-shell {
            background: linear-gradient(180deg, rgba(128, 128, 128, 0.08), rgba(128, 128, 128, 0.04));
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 10px;
            margin: 0.15rem 0 0.85rem;
            padding: 0.9rem 1rem;
        }
        .scope-context-head {
            align-items: center;
            display: flex;
            gap: 0.65rem;
            justify-content: space-between;
        }
        .scope-context-title {
            font-size: 0.98rem;
            font-weight: 760;
            line-height: 1.2;
        }
        .scope-context-meta {
            font-size: 0.84rem;
            margin-top: 0.18rem;
            opacity: 0.78;
        }
        .scope-context-grid {
            display: grid;
            gap: 0.55rem;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            margin-top: 0.8rem;
        }
        .scope-context-cell {
            background: rgba(255, 255, 255, 0.42);
            border: 1px solid rgba(128, 128, 128, 0.14);
            border-radius: 8px;
            padding: 0.6rem 0.7rem;
        }
        .scope-context-label {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            opacity: 0.72;
            text-transform: uppercase;
        }
        .scope-context-value {
            font-size: 1rem;
            font-weight: 720;
            line-height: 1.25;
            margin-top: 0.18rem;
        }
        .run-banner-shell {
            align-items: center;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 8px;
            display: flex;
            gap: 0.8rem;
            margin: 0.25rem 0 0.65rem;
            padding: 0.8rem 0.95rem;
        }
        .run-banner-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .run-banner-meta {
            font-size: 0.82rem;
            margin-top: 0.15rem;
            opacity: 0.72;
        }
        .run-pulse {
            background: rgba(57, 132, 255, 0.22);
            border-radius: 999px;
            flex: 0 0 auto;
            height: 12px;
            width: 12px;
        }
        .run-pulse.active {
            animation: run-pulse 1.15s ease-in-out infinite;
            background: rgba(52, 199, 89, 0.88);
            box-shadow: 0 0 0 0 rgba(52, 199, 89, 0.26);
        }
        @keyframes run-pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(52, 199, 89, 0.30);
                transform: scale(0.95);
            }
            70% {
                box-shadow: 0 0 0 12px rgba(52, 199, 89, 0.0);
                transform: scale(1);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(52, 199, 89, 0.0);
                transform: scale(0.95);
            }
        }
        @media (max-width: 840px) {
            .global-app-bar {
                grid-template-columns: 1fr;
            }
        }
    </style>
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
            (
                '<div class="insight-card">'
                f'<div class="insight-label">{label}</div>'
                f'<div class="insight-value">{value}</div>'
                f'<div class="insight-note">{note}</div>'
                "</div>"
            )
        )
    st.markdown(f'<div class="insight-grid">{"".join(card_html)}</div>', unsafe_allow_html=True)


def status_chip_style(label: str) -> str:
    lowered = str(label or "").lower()
    if "measurement" in lowered or lowered == "high" or "high reliability" in lowered or "fresh" in lowered:
        return "background: rgba(52, 199, 89, 0.16); border-color: rgba(52, 199, 89, 0.38);"
    if "discovery" in lowered or "medium" in lowered or "review" in lowered:
        return "background: rgba(57, 132, 255, 0.14); border-color: rgba(57, 132, 255, 0.34);"
    if (
        "low" in lowered
        or "warning" in lowered
        or "fallback" in lowered
        or "needs baseline" in lowered
        or "partial" in lowered
        or "needs refresh" in lowered
    ):
        return "background: rgba(255, 159, 10, 0.16); border-color: rgba(255, 159, 10, 0.36);"
    if "error" in lowered or "failed" in lowered or "cancelled" in lowered:
        return "background: rgba(255, 69, 58, 0.14); border-color: rgba(255, 69, 58, 0.34);"
    if "watchlist" in lowered or "manual" in lowered:
        return "background: rgba(128, 128, 128, 0.12); border-color: rgba(128, 128, 128, 0.30);"
    if "low-hype" in lowered or "useful" in lowered or "worth deeper" in lowered:
        return "background: rgba(0, 163, 163, 0.14); border-color: rgba(0, 163, 163, 0.34);"
    return "background: rgba(128, 128, 128, 0.10); border-color: rgba(128, 128, 128, 0.26);"


def render_status_chips(labels: list[str]) -> None:
    chips = "".join(
        f'<span class="status-chip" style="{status_chip_style(label)}">{escape(label)}</span>'
        for label in labels
        if label
    )
    if chips:
        st.markdown(chips, unsafe_allow_html=True)


def canonical_page_name(page_name: Any) -> str:
    raw_name = str(page_name or PAGES[0])
    return PAGE_ALIASES.get(raw_name, raw_name if raw_name in PAGES else PAGES[0])


def render_page_header(title: str, subtitle: str = "") -> None:
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def latest_completed_run(runs: pd.DataFrame) -> dict[str, Any] | None:
    if runs.empty:
        return None
    completed = runs[
        (runs["status"].astype(str).str.lower() == "completed")
        & (runs.apply(lambda row: run_has_analysed_items(row.to_dict()), axis=1))
    ].copy()
    if completed.empty:
        return None
    return completed.iloc[0].to_dict()


def compact_run_status_label(runs: pd.DataFrame) -> tuple[str, str]:
    if runs.empty:
        return "Idle", "No runs yet"
    active_run = latest_active_run(runs)
    if active_run:
        status = str(active_run.get("status", "running")).replace("_", " ").title()
        runtime = (active_run.get("summary_json", {}) or {}).get("runtime", {})
        stage = str(runtime.get("stage", "") or "").replace("_", " ").title()
        detail = f"{stage} · {run_elapsed_label(active_run)}" if stage else run_elapsed_label(active_run)
        return status, detail
    latest = runs.iloc[0].to_dict()
    status = str(latest.get("status", "idle")).replace("_", " ").title()
    if status == "Completed":
        status = "Idle"
    return status, run_completed_label(latest)


def render_app_bar(
    *,
    runs: pd.DataFrame,
    run_id: str | None,
    source_lookup: dict[int, str],
    current_page: str,
) -> None:
    status_label, status_detail = compact_run_status_label(runs)
    latest_run = latest_completed_run(runs)
    latest_label = "No completed run"
    if latest_run:
        latest_summary = latest_run.get("summary_json", {}) or {}
        latest_label = (
            f"{run_completed_label(latest_run)} · "
            f"{int(latest_summary.get('items_analyzed', 0) or 0)} items"
        )
    if is_corpus_scope(run_id):
        scope_label = "Combined corpus"
    elif run_id and not runs.empty:
        match = runs[runs["analysis_run_id"].astype(str) == str(run_id)]
        scope_label = build_run_display_label(match.iloc[0].to_dict(), source_lookup) if not match.empty else str(run_id)
    else:
        scope_label = "No scope selected"

    st.markdown(
        f"""
        <div class="global-app-bar">
            <div>
                <div class="global-app-title">Signal from the Slop</div>
                <div class="global-app-subtitle">Local Reddit stock research triage. Not financial advice.</div>
            </div>
            <div class="global-app-meta">
                <div class="global-app-cell">
                    <div class="global-app-label">Run status</div>
                    <div class="global-app-value">{escape(status_label)} · {escape(status_detail)}</div>
                </div>
                <div class="global-app-cell">
                    <div class="global-app-label">Latest run</div>
                    <div class="global-app-value">{escape(latest_label)}</div>
                </div>
                <div class="global-app-cell">
                    <div class="global-app-label">Scope</div>
                    <div class="global-app-value">{escape(scope_label[:96])}</div>
                </div>
                <div class="global-app-cell">
                    <div class="global-app-label">Current page</div>
                    <div class="global-app-value">{escape(current_page)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 5])
    if action_cols[0].button("New Run", type="primary", use_container_width=True):
        st.session_state["page"] = "Collect"
        st.session_state["pending_collect_view"] = "New Run"
        st.rerun()
    if action_cols[1].button("Review", use_container_width=True):
        st.session_state["page"] = "Review"
        if run_id:
            st.session_state[f"pending_review_view_{run_id}"] = "Signals"
        st.rerun()


def render_run_preview_card(
    *,
    run_type: str,
    selected_source_count: int,
    time_window_label: str,
    include_comments: bool,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    estimated_items: int,
) -> None:
    measurement_run = run_type == RUN_TYPE_MEASUREMENT
    estimated_requests = selected_source_count
    if include_comments and max_comments_per_thread:
        estimated_requests += selected_source_count * max_posts_per_source
    chips = [
        RUN_TYPE_LABELS.get(run_type, "Discovery"),
        "Posts + comments" if include_comments else "Posts only",
        "Comparable trend run" if measurement_run and not include_comments else "Not trend-comparable",
    ]
    render_status_chips(chips)
    render_insight_cards(
        [
            {"label": "Sources", "value": str(selected_source_count), "note": "Selected active sources"},
            {"label": "Window", "value": time_window_label, "note": "Resolved before run starts"},
            {"label": "Request load", "value": str(estimated_requests), "note": "Approximate Reddit RSS requests"},
            {"label": "Items", "value": str(estimated_items), "note": "Local classification upper bound"},
        ]
    )
    if measurement_run and include_comments:
        st.warning("Measurement runs should stay posts-only. Comments make acceleration harder to compare.")
    elif measurement_run:
        st.caption("Acceleration can be trusted only against matching Measurement runs with the same source set and healthy coverage.")
    elif include_comments and max_comments_per_thread:
        st.info("Discovery runs with comments can find richer evidence, but Reddit RSS rate limits and local model time become more likely.")


def render_evidence_card_list(filtered_df: pd.DataFrame, *, limit: int = 12) -> None:
    if filtered_df.empty:
        st.info("No evidence cards matched the current filters.")
        return
    for index, row in enumerate(filtered_df.head(limit).to_dict(orient="records"), start=1):
        title = str(row.get("thread_title", "") or "Untitled Reddit item")
        ticker = str(row.get("ticker", ""))
        source = str(row.get("source_name", ""))
        created = str(row.get("created_time", ""))[:19].replace("T", " ")
        with st.container(border=True):
            st.markdown(f"**{ticker} · {title}**")
            render_status_chips(
                [
                    str(row.get("sentiment", "")).title(),
                    f"Evidence: {str(row.get('evidence_quality', 'unknown')).title()}",
                    f"Trust: {str(row.get('trust_status', 'n/a'))}",
                ]
            )
            metric_cols = st.columns(5)
            metric_cols[0].metric("Evidence score", f"{coerce_float(row.get('alpha_signal_score')):.1f}")
            metric_cols[1].metric("Quality-adjusted", f"{coerce_float(row.get('hype_adjusted_signal_score')):.1f}")
            metric_cols[2].metric("Specificity", f"{coerce_float(row.get('claim_specificity_score')):.1f}")
            metric_cols[3].metric("Hype", f"{coerce_float(row.get('hype_score')):.1f}")
            metric_cols[4].metric("Confidence", f"{coerce_float(row.get('confidence'), 0.0):.2f}")
            summary = str(row.get("summary", "") or "").strip()
            if summary:
                st.write(summary)
            claims = list_cell_values(row.get("claims_to_verify"), limit=4)
            red_flags = list_cell_values(row.get("red_flags"), limit=4)
            if claims:
                render_compact_list("Claims to verify", claims)
            if red_flags:
                render_status_chips(red_flags)
            st.caption(f"{source} · {created}")
            permalink = str(row.get("permalink", "") or "")
            if permalink:
                st.link_button("Open Reddit item", permalink)
            with st.expander("Raw text", expanded=False):
                st.code(str(row.get("body_text", "") or ""))


def metric_help(column_name: str) -> str | None:
    return METRIC_TOOLTIPS.get(column_name)


def format_timestamp_with_relative(value: Any) -> str:
    timestamp = str(value or "").strip()
    if not timestamp:
        return "Unknown time"
    try:
        dt_value = parse_created_time(timestamp)
        stamp = dt_value.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        stamp = timestamp[:19].replace("T", " ")
    relative = relative_time_label(timestamp)
    return f"{stamp} • {relative}" if relative else stamp


def ticker_status_labels(row: dict[str, Any]) -> list[str]:
    labels = [str(row.get("trend_reliability", "") or row.get("coverage_reliability", "") or "")]
    if bool(row.get("low_mentions_high_signal")):
        labels.append("Low-hype lead")
    if bool(row.get("new_ticker_detected")):
        labels.append("New ticker")
    hype = coerce_float(row.get("average_hype_score", row.get("hype_score", 0)))
    if hype >= 7:
        labels.append("High hype")
    return [label for label in labels if label]


def market_cache_status_label(review_df: pd.DataFrame) -> str:
    if review_df.empty or "outcome_last_updated" not in review_df.columns:
        return "Market cache needs refresh"
    updated = pd.to_datetime(review_df["outcome_last_updated"], errors="coerce", utc=True).dropna()
    if updated.empty:
        return "Market cache needs refresh"
    latest = updated.max()
    age_hours = max((datetime.now(UTC) - latest.to_pydatetime()).total_seconds() / 3600, 0)
    market_columns = [column for column in ("return_7d", "excess_return_7d") if column in review_df.columns]
    has_missing = bool(market_columns) and bool(review_df[market_columns].isna().to_numpy().any())
    relative = relative_time_label(latest.isoformat())
    if age_hours <= 24 and not has_missing:
        return f"Market cache fresh {relative}".strip()
    if age_hours <= 72:
        return f"Market cache partial {relative}".strip()
    return f"Market cache needs refresh {relative}".strip()


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


def metric_guide_rows(context: str = "all") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in METRIC_GUIDE:
        contexts = entry.get("contexts", [])
        if context != "all" and context not in contexts:
            continue
        rows.append(
            {
                "metric": str(entry["metric"]),
                "formula": str(entry["formula"]),
                "plain_english": str(entry["plain_english"]),
                "how_it_is_determined": str(entry["how_it_is_determined"]),
            }
        )
    return rows


def render_metric_guide(
    title: str = "How to read these metrics",
    *,
    expanded: bool = False,
    context: str = "all",
) -> None:
    with st.expander(title, expanded=expanded):
        matching = [
            entry
            for entry in METRIC_GUIDE
            if context == "all" or context in entry.get("contexts", [])
        ]
        if not matching:
            st.caption("No metric notes are available for this view yet.")
            return

        quick_tab, deep_tab = st.tabs(["Quick Reference", "Formula Deep Dive"])
        with quick_tab:
            st.dataframe(
                pd.DataFrame(metric_guide_rows(context)),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "metric": st.column_config.TextColumn("Metric", width="medium"),
                    "formula": st.column_config.TextColumn("Formula", width="large"),
                    "plain_english": st.column_config.TextColumn("Plain English", width="large"),
                    "how_it_is_determined": st.column_config.TextColumn("How it is determined", width="large"),
                },
            )
        with deep_tab:
            names = [str(entry["metric"]) for entry in matching]
            selected_metric = st.selectbox(
                "Metric",
                names,
                key=f"metric_guide_select_{title}_{context}",
            )
            entry = next(item for item in matching if str(item["metric"]) == selected_metric)
            render_status_chips([label.title() for label in entry.get("contexts", [])])
            st.markdown("**Formula**")
            st.code(str(entry["formula"]))
            st.markdown("**Plain English**")
            st.write(str(entry["plain_english"]))
            st.markdown("**How The App Computes It**")
            st.write(str(entry["how_it_is_determined"]))
            st.markdown("**Why To Be Careful**")
            st.warning(str(entry["why_caution"]))
            st.markdown("**Example**")
            st.info(str(entry["example"]))


def relative_time_label(value: Any) -> str:
    timestamp = str(value or "").strip()
    if not timestamp:
        return ""
    try:
        dt_value = parse_created_time(timestamp)
    except Exception:
        return ""
    delta_seconds = max(int((datetime.now(UTC) - dt_value).total_seconds()), 0)
    if delta_seconds < 60:
        return f"{delta_seconds}s ago"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}m ago"
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        return f"{hours}h {minutes}m ago" if minutes else f"{hours}h ago"
    days = delta_seconds // 86400
    return f"{days}d ago"


def render_scope_context_bar(
    *,
    title: str,
    meta: str,
    chips: list[str],
    metrics: list[dict[str, str]],
) -> None:
    metric_html = "".join(
        (
            '<div class="scope-context-cell">'
            f'<div class="scope-context-label">{escape(metric.get("label", ""))}</div>'
            f'<div class="scope-context-value">{escape(metric.get("value", ""))}</div>'
            "</div>"
        )
        for metric in metrics
    )
    chip_html = "".join(
        f'<span class="status-chip" style="{status_chip_style(chip)}">{escape(chip)}</span>'
        for chip in chips
        if chip
    )
    html = (
        '<div class="scope-context-shell">'
        '<div class="scope-context-head">'
        "<div>"
        f'<div class="scope-context-title">{escape(title)}</div>'
        f'<div class="scope-context-meta">{escape(meta)}</div>'
        "</div>"
        f"<div>{chip_html}</div>"
        "</div>"
        f'<div class="scope-context-grid">{metric_html}</div>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_low_reliability_warning(summary: dict[str, Any] | None) -> None:
    summary = summary or {}
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage", {}), dict) else {}
    reasons = coverage.get("reasons", []) if isinstance(coverage.get("reasons", []), list) else []
    reliability_label = str(coverage.get("reliability_label", "") or "")
    trend_note = str(coverage.get("trend_note", "") or "")
    run_type = str(summary.get("run_type", RUN_TYPE_DISCOVERY)).lower()
    rss_capped_sources = int(coverage.get("rss_capped_sources", 0) or 0)

    details: list[str] = []
    if run_type == RUN_TYPE_DISCOVERY:
        details.append(
            "This is a Discovery run, so it is best for idea finding and evidence review. "
            "Use a Measurement run on the same source set for comparable acceleration."
        )
    if reliability_label == "Low":
        details.extend(str(reason) for reason in reasons[:4] if str(reason).strip())
        if not details:
            details.append("Source coverage came back weak.")
    if rss_capped_sources > 0:
        source_word = "source" if rss_capped_sources == 1 else "sources"
        details.append(
            f"{rss_capped_sources} {source_word} hit the RSS cap, so older posts in the requested window may be missing."
        )
    if trend_note and "baseline" in trend_note.lower():
        details.append(trend_note)
    details = list(dict.fromkeys(details))
    if not details:
        return

    if reliability_label == "Low":
        st.warning("Trend trust is low for this run. Use it for evidence review, not acceleration or run-to-run comparison.")
    elif run_type == RUN_TYPE_DISCOVERY:
        st.info("Discovery run: good for finding leads, not for clean acceleration comparison.")
    else:
        st.warning("Scrape coverage has limits for this run.")

    with st.expander("Why trend trust is low", expanded=False):
        for detail in details:
            st.markdown(f"- {detail}")


def build_page_context_metrics(summary: dict[str, Any]) -> list[dict[str, str]]:
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage", {}), dict) else {}
    return [
        {"label": "Items analysed", "value": str(int(summary.get("items_analyzed", 0) or 0))},
        {"label": "Ticker mentions", "value": str(int(summary.get("ticker_mentions", 0) or 0))},
        {"label": "Posts scraped", "value": str(int(coverage.get("posts_returned", 0) or 0))},
        {"label": "Coverage", "value": str(coverage.get("reliability_label", "n/a") or "n/a")},
    ]


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
                "mention_rate_per_100_posts",
                "acceleration_score",
                "trend_reliability",
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
                    "mention_rate_per_100_posts": st.column_config.NumberColumn("Mentions / 100 posts", format="%.2f"),
                    "acceleration_score": st.column_config.NumberColumn("Adj. accel.", format="%.1f", help=metric_help("acceleration_score")),
                    "trend_reliability": st.column_config.TextColumn("Reliability"),
                    "emerging_ticker_score": st.column_config.NumberColumn("Emerging", format="%.1f", help=metric_help("emerging_ticker_score")),
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


def is_corpus_scope(scope_id: str | None) -> bool:
    return str(scope_id or "") == CORPUS_SCOPE_ID


def dedupe_corpus_mentions(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty:
        return mentions_df

    working = mentions_df.copy()
    working["run_completed_at_dt"] = pd.to_datetime(
        working.get("run_completed_at", working.get("created_time")),
        errors="coerce",
        utc=True,
    )
    working["created_time_dt"] = pd.to_datetime(working["created_time"], errors="coerce", utc=True)
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working = working.sort_values(
        ["run_completed_at_dt", "created_time_dt", "confidence"],
        ascending=[False, False, False],
    )
    return (
        working.drop_duplicates(["item_id", "ticker"], keep="first")
        .drop(columns=["run_completed_at_dt", "created_time_dt"], errors="ignore")
        .sort_values(["created_time", "alpha_signal_score"], ascending=[False, False])
    )


def canonicalize_corpus_company_names(mentions_df: pd.DataFrame) -> pd.DataFrame:
    if mentions_df.empty or "company_name" not in mentions_df.columns:
        return mentions_df

    working = mentions_df.copy()
    company_lookup: dict[str, str] = {}
    for ticker, group in working.groupby("ticker", dropna=False):
        names = group["company_name"].dropna().astype(str)
        names = names[(names.str.strip() != "") & (names.str.lower() != "unknown")]
        if names.empty:
            continue
        modes = names.mode(dropna=True)
        company_lookup[str(ticker)] = str(modes.iloc[0] if not modes.empty else names.iloc[0])
    if company_lookup:
        working["company_name"] = working.apply(
            lambda row: company_lookup.get(str(row["ticker"]), row.get("company_name") or "Unknown"),
            axis=1,
        )
    return working


def build_corpus_frames(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_mentions = load_completed_mentions(db_path, data_mode="live")
    long_df = canonicalize_corpus_company_names(dedupe_corpus_mentions(raw_mentions))
    if long_df.empty:
        completed_runs = load_analysis_runs(db_path)
        completed_count = 0
        if not completed_runs.empty:
            completed_count = int((completed_runs["status"] == "completed").sum())
        return (
            long_df,
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "items_scraped": 0,
                "items_skipped_existing": 0,
                "items_collected": 0,
                "items_analyzed": 0,
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
                "runs_included": completed_count,
                "raw_mention_rows": int(len(raw_mentions)),
                "deduplicated_mentions_removed": 0,
            },
        )

    if "classifier_mode" not in long_df.columns:
        long_df["classifier_mode"] = "unknown"
    if "model_name" not in long_df.columns:
        long_df["model_name"] = ""
    long_df["classifier_mode"] = long_df["classifier_mode"].fillna("unknown")
    long_df["model_name"] = long_df["model_name"].fillna("")
    long_df["trust_status"] = long_df.apply(classifier_trust_label, axis=1)

    summary_df = build_ticker_summaries(long_df)
    if not summary_df.empty:
        raw_acceleration = summary_df["acceleration_score"].astype(float).copy()
        summary_df["coverage_adjusted_acceleration_score"] = 0.0
        summary_df["acceleration_score"] = 0.0
        summary_df["emerging_ticker_score"] = (
            summary_df["emerging_ticker_score"].astype(float) - raw_acceleration * 0.45
        ).clip(lower=0, upper=100).round(2)
        summary_df["trend_reliability"] = "Corpus evidence"
        summary_df["trend_reliability_reason"] = (
            "Combined corpus has no single scrape denominator; use Measurement run comparisons for proper acceleration."
        )
        summary_df["coverage_reliability"] = 0.0
    long_df = apply_ticker_flags_to_mentions(long_df, summary_df) if not summary_df.empty else long_df
    bucket_df = build_time_bucket_summary(long_df) if not long_df.empty else pd.DataFrame()
    summary = build_run_summary(long_df, summary_df)
    run_count = int(long_df["analysis_run_id"].nunique()) if "analysis_run_id" in long_df.columns else 0
    summary.update(
        {
            "items_scraped": int(long_df["item_id"].nunique()),
            "items_skipped_existing": int(len(raw_mentions) - len(long_df)),
            "items_collected": int(long_df["item_id"].nunique()),
            "runs_included": run_count,
            "raw_mention_rows": int(len(raw_mentions)),
            "deduplicated_mentions_removed": int(len(raw_mentions) - len(long_df)),
            "first_seen_date": str(long_df["created_date"].min()),
            "most_recent_seen_date": str(long_df["created_date"].max()),
        }
    )
    return long_df, summary_df, bucket_df, summary


def load_scope_mentions(db_path: Path, scope_id: str | None) -> pd.DataFrame:
    if is_corpus_scope(scope_id):
        long_df, _, _, _ = build_corpus_frames(db_path)
        return long_df
    if not scope_id:
        return pd.DataFrame()
    return load_run_mentions(db_path, scope_id)


def load_scope_ticker_summaries(db_path: Path, scope_id: str | None) -> pd.DataFrame:
    if is_corpus_scope(scope_id):
        _, summary_df, _, _ = build_corpus_frames(db_path)
        return summary_df
    if not scope_id:
        return pd.DataFrame()
    return load_run_ticker_summaries(db_path, scope_id)


def load_signal_review_frame(db_path: Path, scope_id: str | None) -> pd.DataFrame:
    signal_df = load_signal_events(db_path)
    if signal_df.empty:
        return signal_df
    if scope_id and not is_corpus_scope(scope_id):
        signal_df = signal_df[signal_df["analysis_run_id"].astype(str) == str(scope_id)].copy()
    if signal_df.empty:
        return signal_df

    labels_df = load_signal_labels(db_path)
    outcomes_df = load_signal_outcomes(db_path)
    attention_df = load_signal_attention_outcomes(db_path)
    random_df = load_signal_random_benchmarks(db_path)

    review_df = signal_df.copy()
    if not labels_df.empty:
        review_df = review_df.merge(labels_df, on="signal_id", how="left")
    if not outcomes_df.empty:
        review_df = review_df.merge(outcomes_df, on=["signal_id", "ticker", "signal_time"], how="left", suffixes=("", "_market"))
    if not attention_df.empty:
        review_df = review_df.merge(attention_df, on=["signal_id", "ticker", "signal_time"], how="left", suffixes=("", "_attention"))
    if not random_df.empty:
        benchmark_summary = (
            random_df.groupby("signal_id", as_index=False)
            .agg(
                random_sample_size=("benchmark_ticker", "nunique"),
                random_mean_return_7d=("return_7d", "mean"),
                random_mean_return_30d=("return_30d", "mean"),
                random_mean_volume_spike_7d=("volume_spike_7d", "mean"),
            )
        )
        review_df = review_df.merge(benchmark_summary, on="signal_id", how="left")

    defaults: dict[str, Any] = {
        "user_label": "",
        "user_notes": "",
        "return_7d": None,
        "excess_return_7d": None,
        "attention_spread_7d": None,
        "attention_spread_14d": None,
        "attention_spread_30d": None,
        "attention_outcome_reliability": "",
    }
    for column_name, default_value in defaults.items():
        if column_name not in review_df.columns:
            review_df[column_name] = default_value
    review_df["signal_time_dt"] = pd.to_datetime(review_df["signal_time"], errors="coerce", utc=True)
    return review_df.sort_values(["signal_time_dt", "created_at"], ascending=[False, False], na_position="last")


def selected_run_source_ids(run_record: dict[str, Any]) -> list[int]:
    return [int(source_id) for source_id in run_record.get("selected_source_ids", [])]


def build_source_health_frame(
    activity_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
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
    if coverage_df.empty:
        base["posts_returned"] = 0
        base["comments_returned"] = 0
        base["coverage_reliability"] = 0.0
        base["reliability_label"] = "Legacy"
        base["hit_rss_cap"] = False
        base["hit_post_cap"] = False
        base["coverage_reason"] = "No saved scrape coverage metadata."
        base["oldest_seen_at"] = ""
        base["newest_seen_at"] = ""
    else:
        coverage = coverage_df.copy()
        coverage["source_id"] = coverage["source_id"].fillna(-1).astype(int)
        reason_column = "reliability_reason_json" if "reliability_reason_json" in coverage.columns else "reliability_reason"
        coverage["coverage_reason"] = coverage[reason_column].map(
            lambda value: "; ".join(coverage_reason_list(value)[:2])
        ) if reason_column in coverage.columns else ""
        base = base.merge(
            coverage[
                [
                    "source_id",
                    "posts_returned",
                    "comments_returned",
                    "coverage_reliability",
                    "reliability_label",
                    "hit_rss_cap",
                    "hit_post_cap",
                    "coverage_reason",
                    "oldest_seen_at",
                    "newest_seen_at",
                ]
            ],
            on="source_id",
            how="left",
        )
        for column in ("posts_returned", "comments_returned"):
            base[column] = base[column].fillna(0).astype(int)
        base["coverage_reliability"] = base["coverage_reliability"].fillna(0).astype(float)
        base["reliability_label"] = base["reliability_label"].fillna("Low")
        base["hit_rss_cap"] = base["hit_rss_cap"].fillna(False).astype(bool)
        base["hit_post_cap"] = base["hit_post_cap"].fillna(False).astype(bool)
        base["coverage_reason"] = base["coverage_reason"].fillna("")
        base["oldest_seen_at"] = base["oldest_seen_at"].fillna("")
        base["newest_seen_at"] = base["newest_seen_at"].fillna("")

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
        lambda row: "No scrape" if int(row["posts_returned"]) == 0 and int(row["items_analyzed"]) == 0 else (
            "No items" if int(row["items_analyzed"]) == 0 else
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
        load_run_source_coverage(db_path, run_id),
        run_record,
        source_lookup,
    )
    if source_health.empty:
        return

    active_count = int((source_health["posts_returned"] > 0).sum())
    ticker_count = int((source_health["ticker_mentions"] > 0).sum())
    total_count = len(source_health)
    low_count = int((source_health["reliability_label"] == "Low").sum())
    with st.expander(
        f"Source coverage: {active_count}/{total_count} returned posts, {ticker_count}/{total_count} produced ticker mentions, {low_count} low reliability",
        expanded=expanded,
    ):
        st.dataframe(
            source_health[
                [
                    "source_name",
                    "status",
                    "reliability_label",
                    "coverage_reliability",
                    "posts_returned",
                    "comments_returned",
                    "items_analyzed",
                    "ticker_mentions",
                    "hit_rss_cap",
                    "hit_post_cap",
                    "oldest_seen_at",
                    "newest_seen_at",
                    "coverage_reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_name": st.column_config.TextColumn("Source"),
                "reliability_label": st.column_config.TextColumn("Reliability"),
                "coverage_reliability": st.column_config.NumberColumn("Coverage", format="%.2f"),
                "posts_returned": st.column_config.NumberColumn("Posts", format="%d"),
                "comments_returned": st.column_config.NumberColumn("Comments", format="%d"),
                "items_analyzed": st.column_config.NumberColumn("Analysed", format="%d"),
                "ticker_mentions": st.column_config.NumberColumn("Ticker mentions", format="%d"),
                "hit_rss_cap": st.column_config.CheckboxColumn("RSS cap"),
                "hit_post_cap": st.column_config.CheckboxColumn("Post cap"),
                "oldest_seen_at": st.column_config.TextColumn("Oldest"),
                "newest_seen_at": st.column_config.TextColumn("Newest"),
                "coverage_reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )


def queue_completed_run_navigation(run_id: str) -> None:
    st.session_state["pending_selected_run_id"] = run_id
    st.session_state["completed_run_notice"] = f"Run `{run_id}` completed."
    st.session_state["pending_page"] = "Review"


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


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    return coerce_float(value, default)


def _format_percent(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:+.2%}"


def _signal_select_label(review_df: pd.DataFrame, signal_id: str) -> str:
    match = review_df[review_df["signal_id"].astype(str) == str(signal_id)]
    if match.empty:
        return str(signal_id)
    row = match.iloc[0]
    timestamp = str(row.get("signal_time", ""))[:19].replace("T", " ")
    ticker = str(row.get("ticker", ""))
    label = str(row.get("user_label", "") or "")
    suffix = f" · {label}" if label else ""
    return f"{timestamp} · {ticker}{suffix}"


def json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(inner_value) for inner_value in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def coverage_reason_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            return [value]
    return []


def summarize_source_coverage(coverage_df: pd.DataFrame) -> dict[str, Any]:
    if coverage_df.empty:
        return {
            "sources": 0,
            "posts_returned": 0,
            "comments_returned": 0,
            "threads_returned": 0,
            "authors_returned": 0,
            "coverage_reliability": 0.0,
            "reliability_label": "Legacy",
            "low_reliability_sources": 0,
            "rss_capped_sources": 0,
            "post_capped_sources": 0,
            "rate_limited_sources": 0,
            "reasons": ["No saved source coverage metadata is available for this run."],
        }

    working = coverage_df.copy()
    for column in (
        "posts_returned",
        "comments_returned",
        "threads_returned",
        "authors_returned",
        "coverage_reliability",
    ):
        if column not in working.columns:
            working[column] = 0
    weights = working["posts_returned"].fillna(0).astype(float).clip(lower=1)
    reliability = (
        float((working["coverage_reliability"].fillna(0).astype(float) * weights).sum() / weights.sum())
        if float(weights.sum()) > 0
        else 0.0
    )
    label = "High" if reliability >= 0.8 else ("Medium" if reliability >= 0.5 else "Low")
    reasons: list[str] = []
    reason_column = "reliability_reason_json" if "reliability_reason_json" in working.columns else "reliability_reason"
    if reason_column in working.columns:
        for value in working[reason_column].tolist():
            reasons.extend(coverage_reason_list(value))
    reasons = list(dict.fromkeys(reasons))[:5]
    return {
        "sources": int(len(working)),
        "posts_returned": int(working["posts_returned"].fillna(0).sum()),
        "comments_returned": int(working["comments_returned"].fillna(0).sum()),
        "threads_returned": int(working["threads_returned"].fillna(0).sum()),
        "authors_returned": int(working["authors_returned"].fillna(0).sum()),
        "coverage_reliability": round(reliability, 3),
        "reliability_label": label,
        "low_reliability_sources": int((working.get("reliability_label", "") == "Low").sum()) if "reliability_label" in working.columns else 0,
        "rss_capped_sources": int(working.get("hit_rss_cap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "post_capped_sources": int(working.get("hit_post_cap", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "rate_limited_sources": int(working.get("rate_limited", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "reasons": reasons or ["Comparable source sample."],
    }


def latest_previous_measurement_run(
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
        if str(row.get("run_type", RUN_TYPE_DISCOVERY)).lower() != RUN_TYPE_MEASUREMENT:
            continue
        if not bool(row.get("canonical_trend_run", False)):
            continue
        if int((row.get("summary_json") or {}).get("items_analyzed", 0)) <= 0:
            continue
        row_source_ids = sorted(int(source_id) for source_id in row.get("selected_source_ids", []))
        if row_source_ids == normalized_source_ids:
            return row
    return None


def bounded_rate_growth(current_rate: float, previous_rate: float) -> float:
    if previous_rate <= 0:
        if current_rate <= 0:
            return 0.0
        return min(current_rate / 0.25, 4.0)
    return max(min((current_rate - previous_rate) / previous_rate, 4.0), -1.0)


def apply_coverage_adjusted_trend_metrics(
    *,
    db_path: Path,
    run_config: dict[str, Any],
    ticker_summary_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    coverage_summary = summarize_source_coverage(coverage_df)
    if ticker_summary_df.empty:
        return ticker_summary_df, coverage_summary

    working = ticker_summary_df.copy()
    current_posts = max(int(coverage_summary["posts_returned"]), 0)
    current_threads = max(int(coverage_summary["threads_returned"]), current_posts, 1)
    current_authors = max(int(coverage_summary["authors_returned"]), 1)
    working["total_posts_scraped"] = current_posts
    working["mention_rate_per_100_posts"] = working["total_mentions"].map(
        lambda value: round(coerce_float(value) / max(current_posts, 1) * 100, 3)
    )
    working["thread_rate_per_100_posts"] = working["unique_threads"].map(
        lambda value: round(coerce_float(value) / max(current_threads, 1) * 100, 3)
    )
    working["author_rate_per_100_posts"] = working["unique_authors"].map(
        lambda value: round(coerce_float(value) / max(current_authors, 1) * 100, 3)
    )
    working["coverage_reliability"] = float(coverage_summary["coverage_reliability"])
    working["matched_source_count"] = int(coverage_summary["sources"])
    working["previous_posts_scraped"] = 0

    original_acceleration = working["acceleration_score"].astype(float).copy()
    run_type = str(run_config.get("run_type", RUN_TYPE_DISCOVERY)).lower()
    canonical_trend_run = bool(run_config.get("canonical_trend_run", False))
    if run_type != RUN_TYPE_MEASUREMENT or not canonical_trend_run:
        reason = "Discovery scrape is excluded from proper acceleration; use it for ticker discovery and evidence review."
        working["coverage_adjusted_acceleration_score"] = 0.0
        working["acceleration_score"] = 0.0
        working["trend_reliability"] = "Discovery"
        working["trend_reliability_reason"] = reason
        working["emerging_ticker_score"] = (
            working["emerging_ticker_score"].astype(float) - original_acceleration * 0.45
        ).clip(lower=0, upper=100).round(2)
        coverage_summary["trend_note"] = reason
        return working, coverage_summary

    previous_run = latest_previous_measurement_run(
        db_path,
        source_ids=[int(source_id) for source_id in run_config.get("selected_source_ids", [])],
        data_mode=str(run_config.get("data_mode", "live")),
    )
    if not previous_run:
        reason = "No previous measurement scrape exists for this exact source set yet."
        working["coverage_adjusted_acceleration_score"] = 0.0
        working["acceleration_score"] = 0.0
        working["trend_reliability"] = "Needs baseline"
        working["trend_reliability_reason"] = reason
        working["emerging_ticker_score"] = (
            working["emerging_ticker_score"].astype(float) - original_acceleration * 0.45
        ).clip(lower=0, upper=100).round(2)
        coverage_summary["trend_note"] = reason
        return working, coverage_summary

    previous_summary = load_run_ticker_summaries(db_path, str(previous_run["analysis_run_id"]))
    previous_coverage_summary = summarize_source_coverage(
        load_run_source_coverage(db_path, str(previous_run["analysis_run_id"]))
    )
    previous_posts = max(int(previous_coverage_summary["posts_returned"]), 0)
    previous_by_ticker = {
        str(row["ticker"]): row
        for row in previous_summary.to_dict(orient="records")
    } if not previous_summary.empty else {}

    adjusted_scores: list[float] = []
    reliability_labels: list[str] = []
    reliability_reasons: list[str] = []
    previous_post_counts: list[int] = []
    coverage_reliabilities: list[float] = []
    for row in working.to_dict(orient="records"):
        ticker = str(row["ticker"])
        previous_row = previous_by_ticker.get(ticker, {})
        previous_mention_rate = coerce_float(previous_row.get("mention_rate_per_100_posts"))
        previous_thread_rate = coerce_float(previous_row.get("thread_rate_per_100_posts"))
        previous_author_rate = coerce_float(previous_row.get("author_rate_per_100_posts"))
        if previous_posts and previous_mention_rate <= 0 and previous_row:
            previous_mention_rate = coerce_float(previous_row.get("total_mentions")) / previous_posts * 100

        mention_growth = bounded_rate_growth(coerce_float(row.get("mention_rate_per_100_posts")), previous_mention_rate)
        thread_growth = bounded_rate_growth(coerce_float(row.get("thread_rate_per_100_posts")), previous_thread_rate)
        author_growth = bounded_rate_growth(coerce_float(row.get("author_rate_per_100_posts")), previous_author_rate)
        growth_blend = (0.45 * mention_growth) + (0.30 * thread_growth) + (0.25 * author_growth)
        raw_score = max(min(growth_blend * 35, 100), 0)

        reliability = min(
            coerce_float(coverage_summary["coverage_reliability"]),
            coerce_float(previous_coverage_summary["coverage_reliability"]),
        )
        reasons = [*coverage_summary.get("reasons", [])[:2]]
        if previous_posts < 30 or current_posts < 30:
            reliability = min(reliability, 0.3)
            reasons.append("Current or previous measurement sample is below 30 posts.")
        if not previous_row:
            reasons.append("Ticker was not present in the previous matched measurement run.")
        if previous_coverage_summary.get("rss_capped_sources"):
            reasons.append("Previous measurement had at least one RSS-capped source.")
        if coverage_summary.get("rss_capped_sources"):
            reasons.append("Current measurement had at least one RSS-capped source.")

        hype_penalty = max(0.45, 1 - max(coerce_float(row.get("average_hype_score")) - 4, 0) * 0.07)
        repetition_penalty = max(0.55, 1 - coerce_float(row.get("repetition_score")) * 0.04)
        final_score = round(max(min(raw_score * reliability * hype_penalty * repetition_penalty, 100), 0), 2)
        label = "High" if reliability >= 0.8 else ("Medium" if reliability >= 0.5 else "Low")

        adjusted_scores.append(final_score)
        reliability_labels.append(label)
        reliability_reasons.append("; ".join(list(dict.fromkeys(reasons))[:4]))
        previous_post_counts.append(previous_posts)
        coverage_reliabilities.append(round(reliability, 3))

    working["coverage_adjusted_acceleration_score"] = adjusted_scores
    working["acceleration_score"] = working["coverage_adjusted_acceleration_score"]
    working["trend_reliability"] = reliability_labels
    working["trend_reliability_reason"] = reliability_reasons
    working["previous_posts_scraped"] = previous_post_counts
    working["coverage_reliability"] = coverage_reliabilities
    working["emerging_ticker_score"] = (
        working["emerging_ticker_score"].astype(float)
        - original_acceleration * 0.45
        + working["coverage_adjusted_acceleration_score"].astype(float) * 0.45
    ).clip(lower=0, upper=100).round(2)
    coverage_summary["trend_note"] = f"Compared against previous measurement run {previous_run['analysis_run_id']}."
    coverage_summary["previous_run_id"] = str(previous_run["analysis_run_id"])
    coverage_summary["previous_posts_returned"] = previous_posts
    return working, coverage_summary


def save_run_artifacts(
    *,
    artifacts_dir: Path,
    run_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    items: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    source_coverage_df: pd.DataFrame,
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
    source_coverage_df.to_csv(run_dir / "source_coverage.csv", index=False)
    ticker_summary_df.to_csv(run_dir / "ticker_summary.csv", index=False)
    time_bucket_df.to_csv(run_dir / "ticker_time_buckets.csv", index=False)


_ANALYSIS_RUN_COMPLETED_UNCHANGED = object()


class AnalysisRunCancelled(RuntimeError):
    pass


def analysis_run_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def decode_summary_object(raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if raw_value in (None, ""):
        return {}
    try:
        decoded = json.loads(str(raw_value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def read_analysis_run_state(db_path: Path, analysis_run_id: str) -> dict[str, Any] | None:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT analysis_run_id, started_at, completed_at, status, summary_json
                FROM analysis_runs
                WHERE analysis_run_id = ?
                """,
                (analysis_run_id,),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "analysis_run_id": str(row[0]),
        "started_at": row[1],
        "completed_at": row[2],
        "status": str(row[3]),
        "summary_json": decode_summary_object(row[4]),
    }


def update_analysis_run_state(
    db_path: Path,
    analysis_run_id: str,
    *,
    status: str | None = None,
    summary_updates: dict[str, Any] | None = None,
    completed_at: object = _ANALYSIS_RUN_COMPLETED_UNCHANGED,
) -> None:
    with sqlite3.connect(db_path) as conn:
        for _ in range(5):
            updates: list[str] = []
            values: list[Any] = []
            current_summary_raw = "{}"

            if summary_updates is not None:
                current_row = conn.execute(
                    "SELECT summary_json FROM analysis_runs WHERE analysis_run_id = ?",
                    (analysis_run_id,),
                ).fetchone()
                current_summary_raw = str(current_row[0]) if current_row and current_row[0] is not None else "{}"
                current_summary = decode_summary_object(current_summary_raw)
                current_summary.update(summary_updates)
                updates.append("summary_json = ?")
                values.append(json.dumps(current_summary, ensure_ascii=True))

            if status is not None:
                updates.append("status = ?")
                values.append(status)

            if completed_at is not _ANALYSIS_RUN_COMPLETED_UNCHANGED:
                updates.append("completed_at = ?")
                values.append(completed_at)

            if not updates:
                return

            if summary_updates is not None:
                values.extend([analysis_run_id, current_summary_raw])
                cursor = conn.execute(
                    f"UPDATE analysis_runs SET {', '.join(updates)} WHERE analysis_run_id = ? AND summary_json = ?",
                    values,
                )
                if cursor.rowcount == 0:
                    conn.rollback()
                    continue
            else:
                values.append(analysis_run_id)
                conn.execute(
                    f"UPDATE analysis_runs SET {', '.join(updates)} WHERE analysis_run_id = ?",
                    values,
                )
            conn.commit()
            return
    raise RuntimeError(f"Could not update analysis run {analysis_run_id}.")


def cancel_analysis_run_local(db_path: Path, analysis_run_id: str, summary: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE analysis_runs
            SET completed_at = ?, status = ?, summary_json = ?
            WHERE analysis_run_id = ?
            """,
            (
                analysis_run_now_iso(),
                "cancelled",
                json.dumps(summary, ensure_ascii=True),
                analysis_run_id,
            ),
        )
        conn.commit()


def build_cancelled_summary(run_state: dict[str, Any], reason: str) -> dict[str, Any]:
    summary = dict(run_state.get("summary_json", {}) or {})
    runtime = dict(summary.get("runtime", {})) if isinstance(summary.get("runtime", {}), dict) else {}
    control = dict(summary.get("control", {})) if isinstance(summary.get("control", {}), dict) else {}
    now = analysis_run_now_iso()

    runtime.update(
        {
            "stage": "cancelled",
            "message": reason,
            "progress_fraction": float(runtime.get("progress_fraction", 0.0) or 0.0),
            "progress_current": int(runtime.get("progress_current", 0) or 0),
            "progress_total": int(runtime.get("progress_total", 0) or 0),
            "progress_unit": str(runtime.get("progress_unit", "items") or "items"),
            "heartbeat_at": now,
        }
    )
    control.update(
        {
            "pause_requested": False,
            "cancel_requested": True,
            "updated_at": now,
        }
    )
    summary.update(
        {
            "items_scraped": int(summary.get("items_scraped", summary.get("items_collected", 0)) or 0),
            "items_skipped_existing": int(summary.get("items_skipped_existing", 0) or 0),
            "items_collected": int(summary.get("items_collected", summary.get("items_analyzed", 0)) or 0),
            "items_analyzed": int(summary.get("items_analyzed", 0) or 0),
            "ticker_mentions": int(summary.get("ticker_mentions", 0) or 0),
            "tickers_found": int(summary.get("tickers_found", 0) or 0),
            "bullish_items": int(summary.get("bullish_items", 0) or 0),
            "bearish_items": int(summary.get("bearish_items", 0) or 0),
            "neutral_items": int(summary.get("neutral_items", 0) or 0),
            "irrelevant_items": int(summary.get("irrelevant_items", 0) or 0),
            "top_mentioned_tickers": summary.get("top_mentioned_tickers", []),
            "top_accelerating_tickers": summary.get("top_accelerating_tickers", []),
            "high_depth_posts_found": int(summary.get("high_depth_posts_found", 0) or 0),
            "low_mentions_high_signal_count": int(summary.get("low_mentions_high_signal_count", 0) or 0),
            "newly_detected_tickers_count": int(summary.get("newly_detected_tickers_count", 0) or 0),
            "fallback_count": int(summary.get("fallback_count", 0) or 0),
            "cancelled": True,
            "failed_stage": "cancelled",
            "error_message": reason,
            "runtime": runtime,
            "control": control,
        }
    )
    return summary


def force_cancel_analysis_run(db_path: Path, analysis_run_id: str, reason: str) -> None:
    run_state = read_analysis_run_state(db_path, analysis_run_id)
    if not run_state:
        return
    current_status = str(run_state.get("status", "")).lower()
    if current_status in {"completed", "failed", "cancelled"}:
        return
    cancel_analysis_run_local(db_path, analysis_run_id, build_cancelled_summary(run_state, reason))


def build_run_config(
    *,
    run_type: str,
    model_name: str,
    ollama_url: str,
    selected_sources: list[dict[str, Any]],
    time_window_label: str,
    time_window_start: datetime,
    time_window_end: datetime,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    include_comments: bool,
    skip_previously_analyzed: bool,
    run_deeper_analysis: bool,
) -> dict[str, Any]:
    normalized_run_type = str(run_type or RUN_TYPE_DISCOVERY).lower()
    canonical_trend_run = normalized_run_type == RUN_TYPE_MEASUREMENT and not include_comments
    effective_skip_previously_analyzed = bool(skip_previously_analyzed) and not canonical_trend_run
    return {
        "data_mode": "live",
        "run_type": normalized_run_type,
        "canonical_trend_run": canonical_trend_run,
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
        "skip_previously_analyzed": effective_skip_previously_analyzed,
        "run_deeper_analysis": run_deeper_analysis,
        "ollama_model": model_name,
        "ollama_url": ollama_url,
    }


def build_active_run_mask(runs: pd.DataFrame) -> pd.Series:
    return runs["status"].astype(str).str.lower().isin(ACTIVE_RUN_STATUSES)


def latest_active_run(runs: pd.DataFrame) -> dict[str, Any] | None:
    if runs.empty:
        return None
    active_runs = runs[build_active_run_mask(runs)].copy()
    if active_runs.empty:
        return None
    return active_runs.iloc[0].to_dict()


def run_failure_detail(run_record: dict[str, Any]) -> str:
    summary = run_record.get("summary_json", {}) or {}
    runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime", {}), dict) else {}
    failed_stage = str(summary.get("failed_stage", "") or runtime.get("stage", "") or "").replace("_", " ")
    error_message = str(summary.get("error_message", "") or "").strip()
    runtime_message = str(runtime.get("message", "") or "").strip()
    parts: list[str] = []
    if failed_stage:
        parts.append(f"stage: {failed_stage}")
    if error_message:
        parts.append(error_message)
    elif runtime_message:
        parts.append(runtime_message)
    return " | ".join(parts)


def scale_progress(start: float, end: float, current: int, total: int) -> float:
    if total <= 0:
        return end
    clamped_current = min(max(current, 0), total)
    return start + ((end - start) * (clamped_current / total))


def register_run_worker(run_id: str, worker: threading.Thread) -> None:
    with RUN_WORKERS_LOCK:
        RUN_WORKERS[run_id] = worker


def unregister_run_worker(run_id: str) -> None:
    with RUN_WORKERS_LOCK:
        RUN_WORKERS.pop(run_id, None)


def has_live_run_worker(run_id: str) -> bool:
    with RUN_WORKERS_LOCK:
        worker = RUN_WORKERS.get(run_id)
        if worker is None:
            return False
        if worker.is_alive():
            return True
        RUN_WORKERS.pop(run_id, None)
        return False


def request_run_pause(db_path: Path, analysis_run_id: str, *, pause_requested: bool) -> None:
    run_state = read_analysis_run_state(db_path, analysis_run_id)
    if not run_state:
        return
    current_status = str(run_state.get("status", "")).lower()
    if current_status in {"completed", "failed", "cancelled"}:
        return
    summary = run_state.get("summary_json", {}) or {}
    control = dict(summary.get("control", {})) if isinstance(summary.get("control", {}), dict) else {}
    control["pause_requested"] = pause_requested
    control["updated_at"] = analysis_run_now_iso()
    next_status = current_status
    if pause_requested and current_status == "running":
        next_status = "pausing"
    elif pause_requested and current_status == "paused":
        next_status = "paused"
    elif not pause_requested and current_status in {"paused", "pausing"}:
        next_status = "running"
    update_analysis_run_state(
        db_path,
        analysis_run_id,
        status=next_status,
        summary_updates={"control": control},
    )


def request_run_cancel(db_path: Path, analysis_run_id: str) -> None:
    run_state = read_analysis_run_state(db_path, analysis_run_id)
    if not run_state:
        return
    current_status = str(run_state.get("status", "")).lower()
    if current_status in {"completed", "failed", "cancelled"}:
        return
    summary = run_state.get("summary_json", {}) or {}
    control = dict(summary.get("control", {})) if isinstance(summary.get("control", {}), dict) else {}
    control["cancel_requested"] = True
    control["updated_at"] = analysis_run_now_iso()
    update_analysis_run_state(
        db_path,
        analysis_run_id,
        status="cancelling",
        summary_updates={"control": control},
    )


def track_background_run(run_id: str) -> None:
    st.session_state["background_run_id"] = run_id
    st.session_state["background_run_last_status"] = "running"
    st.session_state.pop("background_run_notice", None)
    st.session_state.pop("background_run_notice_level", None)
    st.session_state.pop("background_run_notice_run_id", None)


def sync_tracked_background_run(db_path: Path) -> None:
    tracked_run_id = str(st.session_state.get("background_run_id", "") or "")
    if not tracked_run_id:
        return
    run_state = read_analysis_run_state(db_path, tracked_run_id)
    if not run_state:
        st.session_state["background_run_notice_level"] = "warning"
        st.session_state["background_run_notice"] = f"Background run `{tracked_run_id}` could not be found in the current database."
        st.session_state["background_run_notice_run_id"] = tracked_run_id
        st.session_state.pop("background_run_id", None)
        return
    status = str(run_state.get("status", "unknown")).lower()
    st.session_state["background_run_last_status"] = status
    if status not in {"completed", "failed", "cancelled"}:
        return

    if status == "completed":
        st.session_state["background_run_notice_level"] = "success"
        st.session_state["background_run_notice"] = f"Background run `{tracked_run_id}` completed."
        st.session_state["background_run_notice_run_id"] = tracked_run_id
        st.session_state["selected_run_id"] = tracked_run_id
    elif status == "cancelled":
        st.session_state["background_run_notice_level"] = "warning"
        st.session_state["background_run_notice"] = f"Background run `{tracked_run_id}` was cancelled."
        st.session_state["background_run_notice_run_id"] = tracked_run_id
    else:
        st.session_state["background_run_notice_level"] = "error"
        failure_detail = run_failure_detail(run_state)
        if failure_detail:
            st.session_state["background_run_notice"] = f"Background run `{tracked_run_id}` failed. {failure_detail}"
        else:
            st.session_state["background_run_notice"] = f"Background run `{tracked_run_id}` failed."
        st.session_state["background_run_notice_run_id"] = tracked_run_id
    st.session_state.pop("background_run_id", None)


class AnalysisRunController:
    def __init__(self, *, db_path: Path, run_id: str) -> None:
        self.db_path = db_path
        self.run_id = run_id
        self.last_status = "running"
        self.last_runtime: dict[str, Any] = {
            "stage": "queued",
            "message": "Waiting to start.",
            "progress_fraction": 0.0,
            "progress_current": 0,
            "progress_total": 0,
            "progress_unit": "items",
            "heartbeat_at": analysis_run_now_iso(),
        }

    def report(
        self,
        *,
        stage: str,
        message: str,
        progress_fraction: float,
        progress_current: int = 0,
        progress_total: int = 0,
        progress_unit: str = "items",
        status: str = "running",
    ) -> None:
        runtime = {
            "stage": stage,
            "message": message,
            "progress_fraction": min(max(float(progress_fraction), 0.0), 1.0),
            "progress_current": int(progress_current),
            "progress_total": int(progress_total),
            "progress_unit": str(progress_unit),
            "heartbeat_at": analysis_run_now_iso(),
        }
        self.last_runtime = runtime
        self.last_status = status
        try:
            update_analysis_run_state(
                self.db_path,
                self.run_id,
                status=status,
                summary_updates={"runtime": runtime},
            )
        except RuntimeError:
            logging.warning("Could not persist runtime update for analysis run %s", self.run_id)

    def checkpoint(self) -> None:
        while True:
            run_state = read_analysis_run_state(self.db_path, self.run_id)
            if not run_state:
                raise AnalysisRunCancelled("Run no longer exists.")
            status = str(run_state.get("status", "running")).lower()
            summary = run_state.get("summary_json", {}) or {}
            control = dict(summary.get("control", {})) if isinstance(summary.get("control", {}), dict) else {}
            if bool(control.get("cancel_requested")) or status == "cancelling":
                self.report(
                    stage=str(self.last_runtime.get("stage", "running")),
                    message="Cancellation requested. Stopping after the current request.",
                    progress_fraction=float(self.last_runtime.get("progress_fraction", 0.0)),
                    progress_current=int(self.last_runtime.get("progress_current", 0)),
                    progress_total=int(self.last_runtime.get("progress_total", 0)),
                    progress_unit=str(self.last_runtime.get("progress_unit", "items")),
                    status="cancelling",
                )
                raise AnalysisRunCancelled("Run cancelled by user.")
            if bool(control.get("pause_requested")) or status in {"pausing", "paused"}:
                self.report(
                    stage=str(self.last_runtime.get("stage", "running")),
                    message="Paused. Resume to continue.",
                    progress_fraction=float(self.last_runtime.get("progress_fraction", 0.0)),
                    progress_current=int(self.last_runtime.get("progress_current", 0)),
                    progress_total=int(self.last_runtime.get("progress_total", 0)),
                    progress_unit=str(self.last_runtime.get("progress_unit", "items")),
                    status="paused",
                )
                time_module.sleep(RUN_CONTROL_POLL_SECONDS)
                continue
            if self.last_status == "paused":
                self.report(
                    stage=str(self.last_runtime.get("stage", "running")),
                    message="Resuming analysis.",
                    progress_fraction=float(self.last_runtime.get("progress_fraction", 0.0)),
                    progress_current=int(self.last_runtime.get("progress_current", 0)),
                    progress_total=int(self.last_runtime.get("progress_total", 0)),
                    progress_unit=str(self.last_runtime.get("progress_unit", "items")),
                    status="running",
                )
            return
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
    run_type_label = RUN_TYPE_LABELS.get(str(run_record.get("run_type", RUN_TYPE_DISCOVERY)).lower(), "Discovery")
    return f"{status_prefix}{timestamp} | {run_type_label} | {source_label} | {items_analyzed} items"


def run_has_analysed_items(run_record: dict[str, Any]) -> bool:
    return int((run_record.get("summary_json") or {}).get("items_analyzed", 0)) > 0


def render_workflow_navigation() -> str:
    st.markdown('<div class="workflow-kicker">Workflow</div>', unsafe_allow_html=True)
    current_page = canonical_page_name(st.session_state.get("page", PAGES[0]))
    if st.session_state.get("page") != current_page:
        st.session_state["page"] = current_page

    nav_cols = st.columns(len(PAGE_STEPS))
    for column, step in zip(nav_cols, PAGE_STEPS):
        with column:
            if st.button(
                step["label"],
                key=f"workflow_nav_{step['page']}",
                use_container_width=True,
                type="primary" if current_page == step["page"] else "secondary",
            ):
                if current_page != step["page"]:
                    st.session_state["page"] = step["page"]
                    st.rerun()
    return current_page


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
    items_scraped = int(summary.get("items_scraped", items_collected) or 0)
    items_skipped_existing = int(summary.get("items_skipped_existing", 0) or 0)
    runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime", {}), dict) else {}
    run_label = f"`{latest.get('analysis_run_id')}`"
    source_label = format_source_names(latest, source_lookup)

    if status == "completed":
        st.success(f"Latest run complete: {run_label} with {items_analyzed} analysed item(s).")
    elif status == "running":
        st.warning(f"Latest run still running: {run_label} started {run_elapsed_label(latest)} ago.")
    elif status == "pausing":
        st.warning(f"Latest run pause requested: {run_label}.")
    elif status == "paused":
        st.info(f"Latest run paused: {run_label}.")
    elif status == "cancelling":
        st.warning(f"Latest run cancelling: {run_label}.")
    elif status == "cancelled":
        st.warning(f"Latest run cancelled: {run_label}.")
    elif status == "failed":
        failure_detail = run_failure_detail(latest)
        st.error(f"Latest run failed: {run_label}. {failure_detail}" if failure_detail else f"Latest run failed: {run_label}.")
    else:
        st.info(f"Latest run status: {status} for {run_label}.")
    st.caption(
        f"Sources: {source_label}. Scraped: {items_scraped}. Skipped existing: {items_skipped_existing}. "
        f"Analysed: {items_analyzed}. Completed: `{run_completed_label(latest)}`."
    )
    if runtime:
        st.caption(str(runtime.get("message", "") or ""))


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
            items_scraped = int(summary.get("items_scraped", items_collected) or 0)
            items_skipped_existing = int(summary.get("items_skipped_existing", 0) or 0)
            runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime", {}), dict) else {}
            st.markdown(f"**{status}** `{run_id}`")
            st.caption(
                f"{format_source_names(row, source_lookup)} | started `{str(row.get('started_at', ''))[:19].replace('T', ' ')}` | "
                f"scraped {items_scraped} | skipped {items_skipped_existing} | analysed {items_analyzed}"
            )
            runtime_message = str(runtime.get("message", "") or "")
            if runtime_message:
                st.caption(runtime_message)
            error_message = str(summary.get("error_message", "") or "")
            if error_message:
                st.caption(f"Error: {error_message[:180]}")


def render_background_run_notice() -> None:
    notice = str(st.session_state.get("background_run_notice", "") or "")
    if not notice:
        return
    level = str(st.session_state.get("background_run_notice_level", "info") or "info").lower()
    run_id = str(st.session_state.get("background_run_notice_run_id", "") or "")
    if level == "success":
        st.success(notice)
    elif level == "warning":
        st.warning(notice)
    elif level == "error":
        st.error(notice)
    else:
        st.info(notice)
    action_cols = st.columns([1, 1, 6])
    if run_id and level == "success" and action_cols[0].button("Open results", key=f"notice_results_{run_id}"):
        st.session_state["selected_run_id"] = run_id
        st.session_state["page"] = "Review"
        st.rerun()
    if action_cols[1].button("Dismiss", key=f"notice_dismiss_{run_id or 'latest'}"):
        st.session_state.pop("background_run_notice", None)
        st.session_state.pop("background_run_notice_level", None)
        st.session_state.pop("background_run_notice_run_id", None)
        st.rerun()


def run_heartbeat_age_seconds(run_record: dict[str, Any]) -> int | None:
    summary = run_record.get("summary_json", {}) or {}
    runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime", {}), dict) else {}
    heartbeat_at = str(runtime.get("heartbeat_at", "") or "")
    if not heartbeat_at:
        return None
    try:
        heartbeat = parse_created_time(heartbeat_at)
    except Exception:
        return None
    return max(int((datetime.now(UTC) - heartbeat).total_seconds()), 0)


def is_orphaned_active_run(run_record: dict[str, Any], *, worker_attached: bool = False) -> bool:
    status = str(run_record.get("status", "")).lower()
    if status not in ACTIVE_RUN_STATUSES:
        return False
    if worker_attached:
        return False
    heartbeat_age = run_heartbeat_age_seconds(run_record)
    if heartbeat_age is None:
        return True
    return heartbeat_age >= RUN_ORPHANED_CANCEL_SECONDS


def clear_old_active_runs_without_heartbeat(db_path: Path, runs: pd.DataFrame) -> int:
    if runs.empty:
        return 0
    cleared = 0
    active_rows = runs[build_active_run_mask(runs)].copy()
    for row in active_rows.to_dict(orient="records"):
        run_id = str(row.get("analysis_run_id", "") or "")
        if not run_id or has_live_run_worker(run_id):
            continue
        if run_heartbeat_age_seconds(row) is not None:
            continue
        elapsed_minutes = run_elapsed_minutes(row)
        if elapsed_minutes is None or elapsed_minutes < 2:
            continue
        force_cancel_analysis_run(
            db_path,
            run_id,
            "Cancelled automatically because this active run had no worker heartbeat.",
        )
        cleared += 1
    return cleared


@st.fragment(run_every=1)
def render_background_run_banner(db_path: Path, source_lookup: dict[int, str]) -> None:
    sync_tracked_background_run(db_path)
    render_background_run_notice()

    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
        cleared_runs = clear_old_active_runs_without_heartbeat(db_path, runs)
        if cleared_runs:
            runs = load_analysis_runs(db_path)
            if not runs.empty:
                runs = runs[runs["data_mode"] == "live"].copy()
    active_run = latest_active_run(runs) if not runs.empty else None
    if not active_run:
        return

    summary = active_run.get("summary_json", {}) or {}
    runtime = summary.get("runtime", {}) if isinstance(summary.get("runtime", {}), dict) else {}
    status = str(active_run.get("status", "running")).lower()
    run_id = str(active_run.get("analysis_run_id", ""))
    progress_fraction = float(runtime.get("progress_fraction", 0.0) or 0.0)
    progress_current = int(runtime.get("progress_current", 0) or 0)
    progress_total = int(runtime.get("progress_total", 0) or 0)
    progress_unit = str(runtime.get("progress_unit", "items") or "items")
    source_label = format_source_names(active_run, source_lookup)
    stage_label = str(runtime.get("stage", status)).replace("_", " ").title()
    status_message = str(runtime.get("message", "") or f"{stage_label} in progress.")
    heartbeat_age = run_heartbeat_age_seconds(active_run)
    worker_attached = has_live_run_worker(run_id)
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= RUN_HEARTBEAT_STALE_SECONDS
    live_signal = heartbeat_fresh or worker_attached
    orphaned_run = is_orphaned_active_run(active_run, worker_attached=worker_attached)

    if status == "paused":
        pulse_label = "Paused"
    elif status == "pausing":
        pulse_label = "Pause requested"
    elif status == "cancelling":
        pulse_label = "Cancelling"
    elif live_signal:
        pulse_label = "Live background run"
    else:
        pulse_label = "Checking background state"

    pulse_class = "run-pulse active" if live_signal else "run-pulse"
    st.markdown(
        f"""
        <div class="run-banner-shell">
            <div class="{pulse_class}"></div>
            <div>
                <div class="run-banner-title">{escape(pulse_label)} · {escape(stage_label)}</div>
                <div class="run-banner-meta">Run {escape(run_id)} · {escape(source_label)} · started {escape(run_elapsed_label(active_run))} ago</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress_text = status_message
    if progress_total > 0:
        progress_text = f"{status_message} ({progress_current}/{progress_total} {progress_unit})"
    st.progress(progress_fraction, text=progress_text)

    info_cols = st.columns([3, 2, 2, 2, 1, 1])
    info_cols[0].caption(status_message)
    info_cols[1].metric("Stage", stage_label)
    info_cols[2].metric("Progress", f"{progress_current}/{progress_total}" if progress_total else "Tracking")
    info_cols[3].metric("Heartbeat", f"{heartbeat_age}s ago" if heartbeat_age is not None else "Waiting")

    if status in {"running", "pausing"}:
        if info_cols[4].button("Pause", key=f"pause_run_{run_id}", use_container_width=True):
            request_run_pause(db_path, run_id, pause_requested=True)
            st.rerun()
    elif status == "paused":
        if info_cols[4].button("Resume", key=f"resume_run_{run_id}", use_container_width=True):
            request_run_pause(db_path, run_id, pause_requested=False)
            st.rerun()
    else:
        info_cols[4].empty()

    if status == "cancelling" and orphaned_run:
        cancel_label = "Clear"
    elif status == "cancelling":
        cancel_label = "Canceling"
    else:
        cancel_label = "Cancel"
    cancel_disabled = status == "cancelling" and not orphaned_run
    if info_cols[5].button(cancel_label, key=f"cancel_run_{run_id}", use_container_width=True, disabled=cancel_disabled):
        if orphaned_run:
            force_cancel_analysis_run(
                db_path,
                run_id,
                "Cancelled after the run became orphaned with no live worker heartbeat.",
            )
        else:
            request_run_cancel(db_path, run_id)
        st.rerun()

    if not live_signal and status in {"running", "pausing", "cancelling"}:
        if orphaned_run:
            st.warning(
                "This run is marked active, but no live worker heartbeat is attached. Use Clear to mark it cancelled "
                "and unblock new scrapes."
            )
        else:
            st.warning(
                "This run is marked active, but its heartbeat is stale. Wait a few seconds for the current request to finish, "
                "or open Settings -> Run health if it remains stuck."
            )


def render_watchlist_controls(db_path: Path, run_id: str | None) -> None:
    prefs = load_ui_preferences()
    watchlist = prefs.get("watchlist", [])
    ticker_options = set(watchlist)
    if run_id:
        summary_df = load_scope_ticker_summaries(db_path, run_id)
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

    summary_df = load_scope_ticker_summaries(db_path, run_id)
    mentions_df = load_scope_mentions(db_path, run_id)
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
        st.caption("No current-scope matches.")
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
                navigate_to_page("Ticker Trends", run_id=run_id, ticker=ticker)

    if not mention_matches.empty:
        st.caption("Mention matches")
        for row in mention_matches.to_dict(orient="records"):
            st.caption(f"{row['ticker']} · {str(row['thread_title'])[:72]}")


def render_body_quick_jump(db_path: Path, run_id: str | None, current_page: str) -> None:
    if not run_id:
        return
    query = st.text_input(
        "Jump to ticker",
        placeholder="Search ticker, company, thread, or claim",
        key=f"body_quick_find_{run_id}",
        help="Search the selected analysis scope from the main page body.",
    ).strip()
    if len(query) < 2:
        return

    summary_df = load_scope_ticker_summaries(db_path, run_id)
    mentions_df = load_scope_mentions(db_path, run_id)
    lowered = query.lower()
    ticker_matches = pd.DataFrame()
    if not summary_df.empty:
        ticker_matches = summary_df[
            summary_df["ticker"].astype(str).str.lower().str.contains(lowered, regex=False)
            | summary_df["company_name"].astype(str).str.lower().str.contains(lowered, regex=False)
        ].head(5)

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
        st.info("No ticker or evidence rows matched that search in the selected scope.")
        return

    if not ticker_matches.empty:
        st.caption("Ticker matches")
        cols = st.columns(min(len(ticker_matches), 5))
        for column, row in zip(cols, ticker_matches.to_dict(orient="records")):
            ticker = str(row.get("ticker", ""))
            score = format_metric_value(row.get("emerging_ticker_score"), decimals=1)
            target_page = "Signal Validation" if current_page == "Ticker Trends" else "Ticker Trends"
            if column.button(f"{ticker} · {score}", key=f"body_jump_{run_id}_{current_page}_{ticker}", use_container_width=True):
                navigate_to_page(target_page, run_id=run_id, ticker=ticker)

    if not mention_matches.empty:
        st.caption("Evidence matches")
        for index, row in enumerate(mention_matches.to_dict(orient="records")):
            ticker = str(row.get("ticker", ""))
            cols = st.columns([1, 3])
            if cols[0].button(f"Open {ticker}", key=f"body_evidence_jump_{run_id}_{ticker}_{row.get('item_id', '')}_{index}", use_container_width=True):
                navigate_to_page("Results Dashboard", run_id=run_id, ticker=ticker)
            cols[1].caption(f"{ticker} · {str(row.get('thread_title', ''))[:110]}")


def render_ticker_action_strip(
    rows: pd.DataFrame,
    *,
    run_id: str,
    current_page: str,
    key_prefix: str,
    limit: int = 5,
) -> None:
    if rows.empty:
        return
    current_canonical = canonical_page_name(current_page)
    st.caption("One-click ticker actions")
    for row in rows.head(limit).to_dict(orient="records"):
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue
        cols = st.columns([1.1, 1, 1, 2.2])
        cols[0].markdown(f"**{escape(ticker)}**")
        review_label = "Focus" if current_canonical == "Review" else "Open Review"
        if cols[1].button(review_label, key=f"{key_prefix}_review_{run_id}_{ticker}", use_container_width=True):
            navigate_to_page("Review", run_id=run_id, ticker=ticker)
        if current_canonical != "Validate" and cols[2].button("Open Validate", key=f"{key_prefix}_validate_{run_id}_{ticker}", use_container_width=True):
            navigate_to_page("Validate", run_id=run_id, ticker=ticker)
        elif current_canonical == "Validate":
            cols[2].button("Open Validate", key=f"{key_prefix}_validate_disabled_{run_id}_{ticker}", use_container_width=True, disabled=True)
        with cols[3]:
            render_status_chips(ticker_status_labels(row))


def render_top_signal_cards(summary_df: pd.DataFrame, *, run_id: str, current_page: str, limit: int = 5) -> None:
    if summary_df.empty:
        return
    top_rows = (
        summary_df.sort_values(["emerging_ticker_score", "average_alpha_signal_score"], ascending=[False, False])
        .head(limit)
        .to_dict(orient="records")
    )
    if not top_rows:
        return
    st.subheader("Top Signals")
    current_canonical = canonical_page_name(current_page)
    cols = st.columns(len(top_rows))
    for column, row in zip(cols, top_rows):
        ticker = str(row.get("ticker", ""))
        score = coerce_float(row.get("emerging_ticker_score"))
        with column.container(border=True):
            st.markdown(f'<div class="top-signal-title">{escape(ticker)}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="top-signal-score">{score:.1f} emerging</div>', unsafe_allow_html=True)
            render_status_chips(ticker_status_labels(row))
            review_label = "Focus in Review" if current_canonical == "Review" else "Open Review"
            if st.button(review_label, key=f"top_signal_review_{run_id}_{ticker}", use_container_width=True):
                navigate_to_page("Review", run_id=run_id, ticker=ticker)
            if current_canonical != "Validate" and st.button("Open Validate", key=f"top_signal_validate_{run_id}_{ticker}", use_container_width=True):
                navigate_to_page("Validate", run_id=run_id, ticker=ticker)


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
    run_id: str,
    run_config: dict[str, Any],
    model_name: str,
    ollama_url: str,
    selected_sources: list[dict[str, Any]],
    time_window_label: str,
    time_window_start: datetime,
    time_window_end: datetime,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    include_comments: bool,
    skip_previously_analyzed: bool,
    run_deeper_analysis: bool,
    controller: AnalysisRunController | None = None,
) -> tuple[str, dict[str, Any]]:
    reddit_client = RedditClient()
    extractor = TickerExtractor(ticker_path)
    classifier = OllamaClassifier(model=model_name, endpoint=ollama_url)

    item_rows: list[dict[str, Any]] = []
    item_lookup: dict[str, dict[str, Any]] = {}
    classification_rows: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    scraped_items_count = 0
    skipped_existing_count = 0
    fallback_count = 0
    run_completed = False
    source_coverage_rows: list[dict[str, Any]] = []

    def failure_summary(stage: str, error: Exception | str) -> dict[str, Any]:
        runtime = (
            dict(controller.last_runtime)
            if controller is not None and isinstance(controller.last_runtime, dict)
            else {}
        )
        runtime.update(
            {
                "stage": stage,
                "message": f"Failed during {stage.replace('_', ' ')}: {error}",
                "heartbeat_at": analysis_run_now_iso(),
            }
        )
        return {
            "items_scraped": scraped_items_count,
            "items_skipped_existing": skipped_existing_count,
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
            "coverage": summarize_source_coverage(pd.DataFrame(source_coverage_rows)),
            "run_type": run_config.get("run_type", RUN_TYPE_DISCOVERY),
            "canonical_trend_run": bool(run_config.get("canonical_trend_run", False)),
            "failed_stage": stage,
            "error_message": str(error),
            "runtime": runtime,
        }

    def empty_success_summary(message: str) -> dict[str, Any]:
        return {
            "items_scraped": scraped_items_count,
            "items_skipped_existing": skipped_existing_count,
            "items_collected": len(items),
            "items_analyzed": 0,
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
            "coverage": summarize_source_coverage(pd.DataFrame(source_coverage_rows)),
            "run_type": run_config.get("run_type", RUN_TYPE_DISCOVERY),
            "canonical_trend_run": bool(run_config.get("canonical_trend_run", False)),
            "no_new_items": True,
            "status_message": message,
            "runtime": {
                "stage": "completed",
                "message": message,
                "progress_fraction": 1.0,
                "progress_current": 0,
                "progress_total": 0,
                "progress_unit": "items",
                "heartbeat_at": analysis_run_now_iso(),
            },
        }

    def complete_empty_run(message: str) -> tuple[str, dict[str, Any]]:
        summary = empty_success_summary(message)
        empty_frame = pd.DataFrame()
        source_coverage_df = pd.DataFrame(source_coverage_rows)
        replace_run_source_coverage(db_path, run_id, source_coverage_rows)
        save_run_artifacts(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            config=run_config,
            summary=summary,
            items=[],
            classifications=[],
            source_coverage_df=source_coverage_df,
            mentions_df=empty_frame,
            ticker_summary_df=empty_frame,
            time_bucket_df=empty_frame,
        )
        complete_analysis_run(db_path, run_id, summary)
        return run_id, summary

    def report(
        *,
        stage: str,
        message: str,
        progress_fraction: float,
        progress_current: int = 0,
        progress_total: int = 0,
        progress_unit: str = "items",
        status: str = "running",
    ) -> None:
        if controller is None:
            return
        controller.report(
            stage=stage,
            message=message,
            progress_fraction=progress_fraction,
            progress_current=progress_current,
            progress_total=progress_total,
            progress_unit=progress_unit,
            status=status,
        )

    def checkpoint() -> None:
        if controller is None:
            return
        controller.checkpoint()

    try:
        report(
            stage="scraping",
            message=(
                f"Scraping {len(selected_sources)} source(s) from "
                f"{time_window_start.date().isoformat()} to {time_window_end.date().isoformat()}."
            ),
            progress_fraction=0.02,
            progress_current=0,
            progress_total=len(selected_sources),
            progress_unit="sources",
        )
        checkpoint()
        items = reddit_client.collect_live_items(
            selected_sources=selected_sources,
            window_start=time_window_start,
            window_end=time_window_end,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
            run_type=str(run_config.get("run_type", RUN_TYPE_DISCOVERY)),
            progress_callback=lambda payload: report(
                stage="scraping",
                message=str(payload.get("message", "Scraping Reddit sources...")),
                progress_fraction=scale_progress(
                    0.02,
                    0.22,
                    int(payload.get("current", 0) or 0),
                    int(payload.get("total", len(selected_sources)) or len(selected_sources) or 1),
                ),
                progress_current=int(payload.get("current", 0) or 0),
                progress_total=int(payload.get("total", len(selected_sources)) or len(selected_sources) or 0),
                progress_unit=str(payload.get("unit", "sources") or "sources"),
            ),
            coverage_callback=lambda coverage: source_coverage_rows.append(coverage),
            checkpoint=checkpoint,
        )
        if not items:
            source_names = ", ".join(str(source.get("display_name", source.get("source_id"))) for source in selected_sources)
            message = (
                "No Reddit items matched this source selection and time window. "
                f"Sources: {source_names}. Window: {time_window_start.date().isoformat()} to {time_window_end.date().isoformat()}."
            )
            report(
                stage="completed",
                message=message,
                progress_fraction=1.0,
                status="completed",
            )
            run_completed = True
            return complete_empty_run(message)
        scraped_items_count = len(items)
        effective_skip_previously_analyzed = (
            bool(run_config.get("skip_previously_analyzed", skip_previously_analyzed))
            and not bool(run_config.get("canonical_trend_run", False))
        )
        if effective_skip_previously_analyzed:
            previously_analyzed = load_previously_analyzed_item_ids(db_path, data_mode="live")
            items = [item for item in items if str(item["item_id"]) not in previously_analyzed]
            skipped_existing_count = scraped_items_count - len(items)
            if not items:
                message = (
                    "The scrape returned items, but every item has already been analysed in a completed run. "
                    "Turn off 'Skip already analysed items' if you want to re-analyse this full window."
                )
                report(
                    stage="completed",
                    message=message,
                    progress_fraction=1.0,
                    progress_current=scraped_items_count,
                    progress_total=scraped_items_count,
                    progress_unit="items",
                    status="completed",
                )
                run_completed = True
                return complete_empty_run(message)

        report(
            stage="classifying",
            message=(
                f"Scraped {scraped_items_count} item(s), skipped {skipped_existing_count} already analysed item(s), "
                f"classifying {len(items)} new item(s)."
            ),
            progress_fraction=0.24,
            progress_current=0,
            progress_total=len(items),
            progress_unit="items",
        )

        total_items = max(len(items), 1)
        for index, item in enumerate(items, start=1):
            checkpoint()
            report(
                stage="classifying",
                message=f"Classifying {index}/{len(items)}: {item['thread_title']}",
                progress_fraction=scale_progress(0.24, 0.78, index - 1, total_items),
                progress_current=index - 1,
                progress_total=total_items,
                progress_unit="items",
            )

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
            report(
                stage="classifying",
                message=f"Classified {index}/{len(items)} items.",
                progress_fraction=scale_progress(0.24, 0.78, index, total_items),
                progress_current=index,
                progress_total=total_items,
                progress_unit="items",
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

            eligible_list = sorted(eligible_item_ids)
            total_eligible = max(len(eligible_list), 1)
            for index, item_id in enumerate(eligible_list, start=1):
                checkpoint()
                report(
                    stage="deep_analysis",
                    message=f"Running deeper analysis {index}/{len(eligible_list)}: {item_lookup[item_id]['thread_title']}",
                    progress_fraction=scale_progress(0.80, 0.92, index - 1, total_eligible),
                    progress_current=index - 1,
                    progress_total=total_eligible,
                    progress_unit="items",
                )
                deeper_result = classifier.deep_analyze(item_lookup[item_id], classification_rows[item_id])
                classification_rows[item_id]["deeper_analysis_json"] = deeper_result.payload
                report(
                    stage="deep_analysis",
                    message=f"Completed deeper analysis {index}/{len(eligible_list)}.",
                    progress_fraction=scale_progress(0.80, 0.92, index, total_eligible),
                    progress_current=index,
                    progress_total=total_eligible,
                    progress_unit="items",
                )

        final_mentions = [
            mention
            for item_id, classification_record in classification_rows.items()
            for mention in build_mention_records(item_lookup[item_id], classification_record)
        ]
        long_df = pd.DataFrame(final_mentions)
        ticker_summary_df = build_ticker_summaries(long_df) if not long_df.empty else pd.DataFrame()
        source_coverage_df = pd.DataFrame(source_coverage_rows)
        ticker_summary_df, coverage_summary = apply_coverage_adjusted_trend_metrics(
            db_path=db_path,
            run_config=run_config,
            ticker_summary_df=ticker_summary_df,
            coverage_df=source_coverage_df,
        )
        long_df = apply_ticker_flags_to_mentions(long_df, ticker_summary_df) if not long_df.empty else long_df
        time_bucket_df = build_time_bucket_summary(long_df) if not long_df.empty else pd.DataFrame()
        summary = build_run_summary(long_df, ticker_summary_df)
        item_sentiments = [row["sentiment"] for row in classification_rows.values()]
        summary["items_scraped"] = scraped_items_count
        summary["items_skipped_existing"] = skipped_existing_count
        summary["items_collected"] = len(items)
        summary["items_analyzed"] = len(classification_rows)
        summary["bullish_items"] = item_sentiments.count("bullish")
        summary["bearish_items"] = item_sentiments.count("bearish")
        summary["neutral_items"] = item_sentiments.count("neutral")
        summary["irrelevant_items"] = item_sentiments.count("irrelevant")
        summary["fallback_count"] = fallback_count
        summary["coverage"] = coverage_summary
        summary["run_type"] = run_config.get("run_type", RUN_TYPE_DISCOVERY)
        summary["canonical_trend_run"] = bool(run_config.get("canonical_trend_run", False))

        checkpoint()
        report(
            stage="saving",
            message=f"Saving run {run_id} results.",
            progress_fraction=0.95,
            progress_current=1,
            progress_total=1,
            progress_unit="save",
        )

        replaceable_mentions = long_df.to_dict(orient="records") if not long_df.empty else []
        replaceable_summaries = ticker_summary_df.to_dict(orient="records") if not ticker_summary_df.empty else []
        replaceable_buckets = time_bucket_df.to_dict(orient="records") if not time_bucket_df.empty else []
        upsert_reddit_items(db_path, item_rows)
        replace_run_classifications(db_path, run_id, list(classification_rows.values()))
        replace_run_source_coverage(db_path, run_id, source_coverage_rows)
        replace_run_mentions(db_path, run_id, replaceable_mentions)
        replace_run_ticker_summaries(db_path, run_id, replaceable_summaries)
        replace_run_time_buckets(db_path, run_id, replaceable_buckets)
        report(
            stage="validation",
            message=f"Freezing signal snapshots and refreshing validation outcomes for run {run_id}.",
            progress_fraction=0.97,
            progress_current=1,
            progress_total=1,
            progress_unit="validation",
        )
        try:
            summary["signal_validation"] = run_signal_validation(
                db_path=db_path,
                ticker_path=ticker_path,
                run_id=run_id,
                run_config=run_config,
                summary_df=ticker_summary_df,
                mentions_df=long_df,
                signal_time=analysis_run_now_iso(),
            )
        except Exception as exc:
            logging.exception("Signal validation failed for run %s", run_id)
            summary["signal_validation"] = {
                "status": "error",
                "signals_created": 0,
                "market_dependency_available": False,
                "market_data_errors": [str(exc)],
            }
        save_run_artifacts(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            config=run_config,
            summary=summary,
            items=item_rows,
            classifications=list(classification_rows.values()),
            source_coverage_df=source_coverage_df,
            mentions_df=long_df,
            ticker_summary_df=ticker_summary_df,
            time_bucket_df=time_bucket_df,
        )
        complete_analysis_run(db_path, run_id, summary)
        run_completed = True
        return run_id, summary
    except AnalysisRunCancelled as exc:
        summary = failure_summary("cancelled", exc)
        summary["cancelled"] = True
        try:
            replace_run_source_coverage(db_path, run_id, source_coverage_rows)
        except Exception:
            logging.exception("Failed to save source coverage for cancelled run %s", run_id)
        cancel_analysis_run_local(db_path, run_id, summary)
        raise
    except Exception as exc:
        if not run_completed:
            try:
                replace_run_source_coverage(db_path, run_id, source_coverage_rows)
                fail_analysis_run(db_path, run_id, failure_summary("collection_or_analysis", exc))
            except Exception:
                logging.exception("Failed to mark analysis run %s as failed", run_id)
        raise


def background_analysis_worker(
    *,
    ticker_path: Path,
    db_path: Path,
    artifacts_dir: Path,
    run_id: str,
    run_config: dict[str, Any],
    model_name: str,
    ollama_url: str,
    selected_sources: list[dict[str, Any]],
    time_window_label: str,
    time_window_start: datetime,
    time_window_end: datetime,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    include_comments: bool,
    skip_previously_analyzed: bool,
    run_deeper_analysis: bool,
) -> None:
    controller = AnalysisRunController(db_path=db_path, run_id=run_id)
    try:
        run_analysis_pipeline(
            ticker_path=ticker_path,
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            run_config=run_config,
            model_name=model_name,
            ollama_url=ollama_url,
            selected_sources=selected_sources,
            time_window_label=time_window_label,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
            skip_previously_analyzed=skip_previously_analyzed,
            run_deeper_analysis=run_deeper_analysis,
            controller=controller,
        )
    except AnalysisRunCancelled:
        logging.info("Analysis run %s was cancelled by the user.", run_id)
    except Exception:
        logging.exception("Background analysis run %s crashed", run_id)
    finally:
        unregister_run_worker(run_id)


def start_background_analysis_run(
    *,
    ticker_path: Path,
    db_path: Path,
    artifacts_dir: Path,
    run_type: str,
    model_name: str,
    ollama_url: str,
    selected_sources: list[dict[str, Any]],
    time_window_label: str,
    time_window_start: datetime,
    time_window_end: datetime,
    max_posts_per_source: int,
    max_comments_per_thread: int,
    include_comments: bool,
    skip_previously_analyzed: bool,
    run_deeper_analysis: bool,
) -> str:
    existing_runs = load_analysis_runs(db_path)
    if not existing_runs.empty:
        existing_runs = existing_runs[existing_runs["data_mode"] == "live"].copy()
        active_run = latest_active_run(existing_runs)
        if active_run is not None:
            raise RuntimeError(
                f"Run `{active_run['analysis_run_id']}` is already active. Pause, resume, or cancel it before starting another scrape."
            )

    run_config = build_run_config(
        run_type=run_type,
        model_name=model_name,
        ollama_url=ollama_url,
        selected_sources=selected_sources,
        time_window_label=time_window_label,
        time_window_start=time_window_start,
        time_window_end=time_window_end,
        max_posts_per_source=max_posts_per_source,
        max_comments_per_thread=max_comments_per_thread,
        include_comments=include_comments,
        skip_previously_analyzed=skip_previously_analyzed,
        run_deeper_analysis=run_deeper_analysis,
    )
    run_id = create_analysis_run(db_path, run_config)
    update_analysis_run_state(
        db_path,
        run_id,
        summary_updates={
            "control": {
                "pause_requested": False,
                "cancel_requested": False,
                "updated_at": analysis_run_now_iso(),
            },
            "runtime": {
                "stage": "queued",
                "message": "Background worker starting.",
                "progress_fraction": 0.0,
                "progress_current": 0,
                "progress_total": 0,
                "progress_unit": "items",
                "heartbeat_at": analysis_run_now_iso(),
            },
        },
    )
    worker = threading.Thread(
        target=background_analysis_worker,
        kwargs={
            "ticker_path": ticker_path,
            "db_path": db_path,
            "artifacts_dir": artifacts_dir,
            "run_id": run_id,
            "run_config": run_config,
            "model_name": model_name,
            "ollama_url": ollama_url,
            "selected_sources": selected_sources,
            "time_window_label": time_window_label,
            "time_window_start": time_window_start,
            "time_window_end": time_window_end,
            "max_posts_per_source": max_posts_per_source,
            "max_comments_per_thread": max_comments_per_thread,
            "include_comments": include_comments,
            "skip_previously_analyzed": bool(run_config.get("skip_previously_analyzed", skip_previously_analyzed)),
            "run_deeper_analysis": run_deeper_analysis,
        },
        name=f"analysis-{run_id}",
        daemon=True,
    )
    register_run_worker(run_id, worker)
    try:
        worker.start()
    except Exception as exc:
        unregister_run_worker(run_id)
        fail_analysis_run(
            db_path,
            run_id,
            {
                "items_scraped": 0,
                "items_skipped_existing": 0,
                "items_collected": 0,
                "items_analyzed": 0,
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
                "fallback_count": 0,
                "failed_stage": "worker_startup",
                "error_message": str(exc),
            },
        )
        raise
    return run_id


def build_sidebar_state(db_path: Path) -> tuple[str, str | None, pd.DataFrame, pd.DataFrame]:
    sources = load_sources(db_path)
    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
        cleared_runs = clear_old_active_runs_without_heartbeat(db_path, runs)
        if cleared_runs:
            st.session_state["auto_cleared_orphaned_runs"] = cleared_runs
            runs = load_analysis_runs(db_path)
            if not runs.empty:
                runs = runs[runs["data_mode"] == "live"].copy()
    pending_page = st.session_state.pop("pending_page", None)
    if pending_page:
        if pending_page == "Run Analysis":
            st.session_state["pending_collect_view"] = "New Run"
        elif pending_page == "Sources":
            st.session_state["pending_collect_view"] = "Sources"
        elif pending_page == "Export Data":
            st.session_state["pending_library_view"] = "Exports"
        st.session_state["page"] = canonical_page_name(pending_page)
    pending_run_id = st.session_state.pop("pending_selected_run_id", None)
    if pending_run_id and not runs.empty and pending_run_id in runs["analysis_run_id"].tolist():
        st.session_state["selected_run_id"] = pending_run_id

    page = canonical_page_name(st.session_state.get("page", PAGES[0]))
    st.session_state["page"] = page

    completed_runs = runs[
        (runs["status"] == "completed")
        & (runs.apply(lambda row: run_has_analysed_items(row.to_dict()), axis=1))
    ].copy() if not runs.empty else runs
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}

    with st.sidebar:
        st.subheader("Analysis scope")
        auto_cleared_runs = int(st.session_state.pop("auto_cleared_orphaned_runs", 0) or 0)
        if auto_cleared_runs:
            st.warning(f"Cleared {auto_cleared_runs} old active run(s) with no heartbeat.")

        show_all_runs = False
        run_search = ""
        with st.expander("Run browser", expanded=False):
            if not runs.empty:
                overview_cols = st.columns(2)
                overview_cols[0].metric("Completed", len(completed_runs))
                overview_cols[1].metric("All runs", len(runs))
                render_latest_run_status(runs, source_lookup)
                render_recent_run_statuses(runs, source_lookup)
            show_all_runs = st.toggle(
                "Show all runs",
                value=False,
                key="sidebar_show_all_runs",
                help="Includes running, failed, and zero-item runs that are hidden from the default results picker.",
                disabled=runs.empty,
            )
            run_search = st.text_input(
                "Find run",
                placeholder="Search ID, date, or source",
                key="sidebar_run_search",
                disabled=runs.empty,
            ).strip()

        selectable_runs = runs if show_all_runs else (completed_runs if not completed_runs.empty else runs)
        run_label_lookup = {
            str(row["analysis_run_id"]): build_run_display_label(row, source_lookup)
            for row in selectable_runs.to_dict(orient="records")
        }
        if not completed_runs.empty:
            run_label_lookup[CORPUS_SCOPE_ID] = CORPUS_SCOPE_LABEL

        run_id = None
        if not selectable_runs.empty:
            filtered_runs = selectable_runs.copy()
            if run_search:
                query = run_search.lower()
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
            if not completed_runs.empty:
                corpus_search_text = "combined corpus all completed runs all data timeline aggregate"
                if not run_search or run_search.lower() in corpus_search_text:
                    filtered_run_options = [CORPUS_SCOPE_ID, *filtered_run_options]
            default_run_id = st.session_state.get("selected_run_id")
            if default_run_id not in filtered_run_options:
                default_run_id = CORPUS_SCOPE_ID if CORPUS_SCOPE_ID in filtered_run_options else str(filtered_run_options[0])
            run_index = filtered_run_options.index(default_run_id) if default_run_id in filtered_run_options else 0
            run_id = st.selectbox(
                "Analysis scope",
                filtered_run_options,
                index=run_index,
                key="selected_run_id",
                format_func=lambda value: run_label_lookup.get(str(value), str(value)),
            )
            selected_rows = selectable_runs[selectable_runs["analysis_run_id"] == run_id] if not is_corpus_scope(run_id) else pd.DataFrame()
            if is_corpus_scope(run_id):
                st.caption(f"Scope: all completed live runs with analysed ticker mentions. Overlapping item/ticker rows are counted once.")
            if not selected_rows.empty:
                selected_record = selected_rows.iloc[0].to_dict()
                st.caption(f"Sources: {format_source_names(selected_record, source_lookup)}")
                st.caption(f"Completed: `{run_completed_label(selected_record)}`")
        else:
            st.info("No analysis runs saved yet.")

        if run_id and page in {"Review", "Validate"}:
            render_sidebar_quick_search(db_path, run_id)

    return page, run_id, sources, runs


def render_sources_page(db_path: Path, *, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Sources")
    else:
        render_page_header("Sources", "Add subreddit sources or Reddit thread URLs.")
    st.write("Add subreddit sources or Reddit thread URLs.")

    if not embedded and st.button("Open New Run", type="primary"):
        st.session_state["pending_page"] = "Collect"
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
    embedded: bool = False,
) -> None:
    if embedded:
        st.subheader("New Run")
    else:
        render_page_header("New Run", "Collect and classify Reddit evidence from selected sources.")
    sources = load_sources(db_path)
    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
    active_run = latest_active_run(runs) if not runs.empty else None
    if sources.empty:
        st.warning("Add at least one source before running analysis.")
        return

    active_sources = sources[sources["active"]]
    if active_sources.empty:
        st.warning("No active sources are enabled.")
        return

    data_mode = "live"
    if active_run:
        st.caption(
            "A scrape is already running in the background. Pause, resume, or cancel it from the live run bar above."
        )

    run_type_label = st.radio(
        "Run purpose",
        [RUN_TYPE_LABELS[RUN_TYPE_MEASUREMENT], RUN_TYPE_LABELS[RUN_TYPE_DISCOVERY]],
        index=0,
        horizontal=True,
        help=(
            "Measurement runs are controlled posts-only samples used for coverage-adjusted trends. "
            "Discovery runs can include comments and flexible depth, but are down-weighted for acceleration."
        ),
    )
    run_type = RUN_TYPE_MEASUREMENT if run_type_label == RUN_TYPE_LABELS[RUN_TYPE_MEASUREMENT] else RUN_TYPE_DISCOVERY
    measurement_run = run_type == RUN_TYPE_MEASUREMENT

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
        "Start from last completed run",
        value=False,
        disabled=not previous_run,
        help="If enabled, the analysis window starts immediately after the most recent completed run for the current source selection.",
    )
    if previous_run:
        st.caption(
            f"Previous matching run: `{previous_run['analysis_run_id']}` at `{previous_run.get('completed_at') or previous_run.get('started_at')}`."
        )
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

    with st.expander("Advanced run options", expanded=False):
        limit_cols = st.columns(2)
        max_posts_per_source = int(
            limit_cols[0].number_input(
                "Max posts per source",
                min_value=1,
                max_value=REDDIT_SUBREDDIT_RSS_LIMIT,
                value=30 if measurement_run else 25,
                step=5,
                help=(
                    "Public Reddit RSS can return up to roughly the newest "
                    f"{REDDIT_SUBREDDIT_RSS_LIMIT} posts per subreddit request. "
                    "This setting trims how many posts we keep from that feed; it does not reduce the initial subreddit "
                    "request count. Use a larger number for overnight runs."
                ),
            )
        )
        max_comments_per_thread = int(
            limit_cols[1].number_input(
                "Max comments per thread",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                help=(
                    "Comments multiply both the number of items sent to Ollama and the number of Reddit RSS requests. "
                    "Set this to 0 for faster posts-only runs."
                ),
                disabled=measurement_run,
            )
        )
        item_scope = st.radio(
            "Analyse scope",
            ["Posts + comments", "Posts only"],
            index=1,
            horizontal=True,
            disabled=measurement_run,
            help="Measurement runs stay posts-only so ticker acceleration is based on comparable independent attention events.",
        )
        include_comments = item_scope == "Posts + comments"
        skip_previously_analyzed = st.toggle(
            "Skip already analysed Reddit items",
            value=not measurement_run,
            disabled=measurement_run,
            help=(
                "When enabled, items that already have a completed classification in an earlier run are not classified again. "
                "Measurement runs always re-analyse the full window so trend rates are comparable."
            ),
        )
        if measurement_run:
            skip_previously_analyzed = False
        run_deeper_analysis = st.toggle("Run deeper analysis for qualifying items", value=True)

        connection_cols = st.columns(2)
        ollama_url = connection_cols[0].text_input(
            "Ollama URL",
            DEFAULT_OLLAMA_URL,
            help="For local use, keep this on localhost. Streamlit Cloud cannot reach the Ollama server on your Mac.",
        )
        available_models, model_error = discover_ollama_models(ollama_url)
        if available_models:
            default_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
            model_name = connection_cols[1].selectbox(
                "Ollama model",
                available_models,
                index=available_models.index(default_model),
            )
            if DEFAULT_MODEL not in available_models:
                st.caption(f"Configured model `{DEFAULT_MODEL}` is not installed locally. Using `{default_model}` by default.")
        else:
            model_name = connection_cols[1].text_input("Ollama model", DEFAULT_MODEL)
            if model_error:
                st.caption(f"Could not list local Ollama models: {model_error}")

    selected_source_frame = sources[sources["source_id"].isin(selected_source_ids)].copy()
    selected_source_count = len(selected_source_ids)
    estimated_items = selected_source_count * max_posts_per_source
    if include_comments:
        estimated_items += selected_source_count * max_posts_per_source * max_comments_per_thread

    start_dt, end_dt = resolve_time_window(
        anchor=anchor,
        label=effective_time_window_label,
        custom_range=custom_range,
        previous_run=previous_run,
    )
    st.caption(f"Window resolved against current time: `{start_dt.date().isoformat()}` to `{end_dt.date().isoformat()}`.")
    if use_last_run_window and previous_run:
        st.caption(
            f"Using `{str(previous_run.get('completed_at') or previous_run['time_window_end'])[:19]}` as the start point for this source set."
        )

    st.subheader("Run Preview")
    render_run_preview_card(
        run_type=run_type,
        selected_source_count=selected_source_count,
        time_window_label=effective_time_window_label,
        include_comments=include_comments,
        max_posts_per_source=max_posts_per_source,
        max_comments_per_thread=max_comments_per_thread,
        estimated_items=estimated_items,
    )
    if max_posts_per_source >= REDDIT_SUBREDDIT_RSS_LIMIT:
        st.info(
            f"{REDDIT_SUBREDDIT_RSS_LIMIT} is the current per-subreddit ceiling for the public RSS collector. "
            "For deeper history than that, run smaller rolling windows or add Reddit API pagination later."
        )
    if measurement_run and max_posts_per_source < 30:
        st.warning("Measurement runs below 30 posts per source will be marked lower reliability for acceleration.")
    if estimated_items >= 500:
        st.warning(
            "This is a large local Ollama run. It can work, but leave the terminal and Ollama running, "
            "use the background progress bar, and expect rate limits or long classification time."
        )

    run_disabled = active_run is not None
    button_label = "Background run already active" if run_disabled else "Start background scrape"
    if st.button(button_label, type="primary", use_container_width=True, disabled=run_disabled):
        if not selected_source_ids:
            st.error("Select at least one source.")
            return
        if effective_time_window_label == "Since last completed run" and not previous_run:
            st.error("A matching completed run is required to use the 'Since last completed run' option.")
            return

        selected_sources = sources[sources["source_id"].isin(selected_source_ids)].to_dict(orient="records")
        try:
            run_id = start_background_analysis_run(
                ticker_path=ticker_path,
                db_path=db_path,
                artifacts_dir=artifacts_dir,
                run_type=run_type,
                model_name=model_name,
                ollama_url=ollama_url,
                selected_sources=selected_sources,
                time_window_label=effective_time_window_label,
                time_window_start=start_dt,
                time_window_end=end_dt,
                max_posts_per_source=max_posts_per_source,
                max_comments_per_thread=max_comments_per_thread,
                include_comments=include_comments,
                skip_previously_analyzed=skip_previously_analyzed,
                run_deeper_analysis=run_deeper_analysis,
            )
            track_background_run(run_id)
            st.session_state["pending_collect_view"] = "Run Progress"
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to the UI
            st.exception(exc)


def render_run_summary_cards(summary: dict[str, Any]) -> None:
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage", {}), dict) else {}
    render_status_chips(
        [
            RUN_TYPE_LABELS.get(str(summary.get("run_type", RUN_TYPE_DISCOVERY)).lower(), "Discovery"),
            str(coverage.get("reliability_label", "") or ""),
            "Comparable trend run" if bool(summary.get("canonical_trend_run", False)) else "",
        ]
    )
    metrics = st.columns(6)
    metrics[0].metric("Items scraped", summary.get("items_scraped", summary.get("items_collected", 0)))
    metrics[1].metric("Skipped existing", summary.get("items_skipped_existing", 0))
    metrics[2].metric("Items analysed", summary.get("items_analyzed", 0))
    metrics[3].metric("Ticker mentions", summary.get("ticker_mentions", 0))
    metrics[4].metric("Bullish items", summary.get("bullish_items", 0))
    metrics[5].metric("Bearish items", summary.get("bearish_items", 0))

    extra = st.columns(4)
    extra[0].metric("Neutral items", summary.get("neutral_items", 0))
    extra[1].metric("Irrelevant items", summary.get("irrelevant_items", 0))
    extra[2].metric("Low-mention high-signal", summary.get("low_mentions_high_signal_count", 0))
    extra[3].metric("Newly detected tickers", summary.get("newly_detected_tickers_count", 0))

    if coverage:
        coverage_cols = st.columns(4)
        coverage_cols[0].metric("Run type", RUN_TYPE_LABELS.get(str(summary.get("run_type", RUN_TYPE_DISCOVERY)).lower(), "Discovery"))
        coverage_cols[1].metric("Trend trust", coverage.get("reliability_label", "Unknown"))
        coverage_cols[2].metric("Posts scraped", coverage.get("posts_returned", 0))
        coverage_cols[3].metric("RSS-capped sources", coverage.get("rss_capped_sources", 0))
        trend_note = str(coverage.get("trend_note", "") or "")
        if trend_note:
            st.caption(trend_note)

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
            label="Top comparable acceleration",
            help_text="Click to copy ticker.",
            max_items=8,
        )

    if summary.get("fallback_count", 0):
        st.warning(
            "Some items used heuristic fallback classification because the Ollama endpoint was unavailable "
            "or returned invalid JSON."
        )


def render_corpus_summary_cards(summary: dict[str, Any]) -> None:
    render_status_chips(["Corpus evidence"])
    metrics = st.columns(6)
    metrics[0].metric("Runs included", summary.get("runs_included", 0))
    metrics[1].metric("Unique items", summary.get("items_analyzed", 0))
    metrics[2].metric("Ticker mentions", summary.get("ticker_mentions", 0))
    metrics[3].metric("Tickers found", summary.get("tickers_found", 0))
    metrics[4].metric("Bullish items", summary.get("bullish_items", 0))
    metrics[5].metric("Bearish items", summary.get("bearish_items", 0))

    extra = st.columns(4)
    extra[0].metric("Deduped overlaps", summary.get("deduplicated_mentions_removed", 0))
    extra[1].metric("Raw mention rows", summary.get("raw_mention_rows", 0))
    extra[2].metric("Low-mention high-signal", summary.get("low_mentions_high_signal_count", 0))
    extra[3].metric("Newly detected tickers", summary.get("newly_detected_tickers_count", 0))

    date_start = str(summary.get("first_seen_date", "") or "")
    date_end = str(summary.get("most_recent_seen_date", "") or "")
    if date_start or date_end:
        st.caption(f"Corpus timeline uses Reddit item dates from `{date_start or 'unknown'}` to `{date_end or 'unknown'}`.")

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
            label="Top comparable acceleration",
            help_text="Click to copy ticker.",
            max_items=8,
        )


def filter_results_dataframe(long_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    working = long_df.copy()
    st.subheader("Review controls")

    preset_names = list(FILTER_PRESETS.keys())
    preset_key = f"results_preset_{run_id}"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = preset_names[0]
    preset_name = st.selectbox(
        "Preset",
        preset_names,
        key=preset_key,
        help="Start from the view you need most often, then refine only when necessary.",
    )
    preset = FILTER_PRESETS[preset_name]
    last_preset_key = f"results_last_applied_preset_{run_id}"
    preset_defaults = {
        f"results_sentiment_filter_{run_id}": list(preset["sentiment"]),
        f"results_min_confidence_{run_id}": float(preset["min_confidence"]),
        f"results_min_depth_{run_id}": int(preset["min_depth"]),
        f"results_min_alpha_{run_id}": int(preset["min_alpha"]),
        f"results_max_hype_{run_id}": int(preset["max_hype"]),
        f"results_hide_hype_{run_id}": bool(preset["hide_hype"]),
        f"results_only_deeper_{run_id}": bool(preset["only_deeper"]),
        f"results_only_low_signal_{run_id}": bool(preset["only_low_signal"]),
        f"results_only_new_{run_id}": bool(preset["only_new"]),
    }
    if st.session_state.get(last_preset_key) != preset_name:
        for key, value in preset_defaults.items():
            st.session_state[key] = value
        st.session_state[last_preset_key] = preset_name
    else:
        for key, value in preset_defaults.items():
            st.session_state.setdefault(key, value)

    primary_cols = st.columns([1.4, 1.4, 1])
    ticker_options = sorted(working["ticker"].dropna().unique())
    source_options = sorted(working["source_name"].dropna().unique())
    ticker_key = f"results_ticker_filter_{run_id}"
    source_key = f"results_source_filter_{run_id}"
    watchlist_key = f"results_watchlist_filter_{run_id}"
    st.session_state[ticker_key] = [
        value for value in st.session_state.get(ticker_key, [])
        if value in ticker_options
    ]
    st.session_state[source_key] = [
        value for value in st.session_state.get(source_key, [])
        if value in source_options
    ]
    st.session_state.setdefault(watchlist_key, False)
    ticker_filter = primary_cols[0].multiselect(
        "Ticker",
        ticker_options,
        key=ticker_key,
    )
    source_filter = primary_cols[1].multiselect(
        "Source",
        source_options,
        key=source_key,
    )
    watchlist = get_watchlist()
    only_watchlist = primary_cols[2].checkbox(
        "Watchlist only",
        disabled=not watchlist,
        key=watchlist_key,
    )

    with st.expander("Advanced filters", expanded=False):
        filter_cols = st.columns(3)
        sentiment_filter = filter_cols[0].multiselect(
            "Sentiment",
            ["bullish", "bearish", "neutral", "irrelevant"],
            key=f"results_sentiment_filter_{run_id}",
        )
        min_confidence = filter_cols[1].slider(
            "Min confidence",
            0.0,
            1.0,
            step=0.05,
            key=f"results_min_confidence_{run_id}",
        )
        min_depth = filter_cols[2].slider(
            "Min depth score",
            0,
            10,
            key=f"results_min_depth_{run_id}",
        )

        numeric_cols = st.columns(4)
        min_alpha = numeric_cols[0].slider(
            "Min alpha signal",
            0,
            100,
            key=f"results_min_alpha_{run_id}",
        )
        max_hype = numeric_cols[1].slider(
            "Max hype",
            0,
            10,
            key=f"results_max_hype_{run_id}",
        )
        hide_hype = numeric_cols[2].checkbox(
            "Hide hype/meme",
            key=f"results_hide_hype_{run_id}",
        )
        only_deeper = numeric_cols[3].checkbox(
            "Needs verification",
            key=f"results_only_deeper_{run_id}",
        )

        flag_cols = st.columns(2)
        only_low_signal = flag_cols[0].checkbox(
            "Low-mention high-signal",
            key=f"results_only_low_signal_{run_id}",
        )
        only_new = flag_cols[1].checkbox(
            "Newly detected",
            key=f"results_only_new_{run_id}",
        )

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
        run_type = str(run_record.get("run_type", RUN_TYPE_DISCOVERY)).title()
        trend_status = "Yes" if bool(run_record.get("canonical_trend_run", False)) else "No"
        st.write(f"Run type: `{run_type}`")
        st.write(f"Comparable trend run: `{trend_status}`")
        st.write(f"Collected window: `{run_record['time_window_start']}` to `{run_record['time_window_end']}`")
        st.write(f"Started at: `{run_record['started_at']}`")
        if run_record.get("completed_at"):
            st.write(f"Completed at: `{run_record['completed_at']}`")
        st.write(f"Sources: {', '.join(source_names) if source_names else 'None'}")


def render_selected_scope_context(db_path: Path, run_id: str | None, *, summary: dict[str, Any] | None = None) -> None:
    if not run_id:
        return
    if is_corpus_scope(run_id):
        corpus_summary = summary or {}
        render_scope_context_bar(
            title=CORPUS_SCOPE_LABEL,
            meta="Deduped evidence across all completed live runs. Proper acceleration still belongs to matched Measurement runs.",
            chips=["Corpus evidence"],
            metrics=[
                {"label": "Runs included", "value": str(int(corpus_summary.get("runs_included", 0) or 0))},
                {"label": "Unique items", "value": str(int(corpus_summary.get("items_analyzed", 0) or 0))},
                {"label": "Ticker mentions", "value": str(int(corpus_summary.get("ticker_mentions", 0) or 0))},
                {"label": "Overlap rows removed", "value": str(int(corpus_summary.get("deduplicated_mentions_removed", 0) or 0))},
            ],
        )
        return

    runs = load_analysis_runs(db_path)
    run_rows = runs[runs["analysis_run_id"] == run_id]
    if run_rows.empty:
        return
    run_record = run_rows.iloc[0].to_dict()
    if summary is None:
        summary = run_record.get("summary_json", {}) or {}
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage", {}), dict) else {}
    sources = load_sources(db_path)
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}
    source_names = format_source_names(run_record, source_lookup)
    completed_at = str(run_record.get("completed_at") or run_record.get("started_at") or "")
    completed_label = format_timestamp_with_relative(completed_at)
    chips = [
        RUN_TYPE_LABELS.get(str(run_record.get("run_type", RUN_TYPE_DISCOVERY)).lower(), "Discovery"),
        str(coverage.get("reliability_label", "n/a") or "n/a"),
    ]
    if bool(run_record.get("canonical_trend_run", False)):
        chips.append("Comparable trend run")
    render_scope_context_bar(
        title=f"Run `{run_id}`",
        meta=f"{source_names}",
        chips=chips,
        metrics=[
            {"label": "Run type", "value": RUN_TYPE_LABELS.get(str(run_record.get("run_type", RUN_TYPE_DISCOVERY)).lower(), "Discovery")},
            {"label": "Source set", "value": source_names},
            {"label": "Completed", "value": completed_label},
            {"label": "Window", "value": str(run_record.get("time_window_label", "") or "Custom")},
            {"label": "Items analysed", "value": str(int(summary.get("items_analyzed", 0) or 0))},
            {"label": "Ticker mentions", "value": str(int(summary.get("ticker_mentions", 0) or 0))},
            {"label": "Coverage", "value": str(coverage.get("reliability_label", "n/a") or "n/a")},
        ],
    )


def render_persistent_run_context(db_path: Path, run_id: str | None) -> None:
    if not run_id:
        st.info("No analysis scope is selected yet. Add sources, run a scrape, then choose the run from the sidebar.")
        return
    if is_corpus_scope(run_id):
        _, _, _, summary = build_corpus_frames(db_path)
        render_selected_scope_context(db_path, run_id, summary=summary)
        return
    summary = load_run_summary(db_path, run_id)
    render_selected_scope_context(db_path, run_id, summary=summary)
    render_low_reliability_warning(summary)


def navigate_to_page(page_name: str, *, run_id: str | None = None, ticker: str | None = None) -> None:
    requested_page = str(page_name or "")
    target_page = canonical_page_name(requested_page)
    if run_id:
        st.session_state["selected_run_id"] = run_id
    if ticker:
        normalized = normalize_ticker(ticker)
        if normalized:
            st.session_state["ticker_focus"] = normalized
            if run_id:
                st.session_state[f"signal_ticker_filter_{run_id}"] = normalized
                st.session_state[f"results_ticker_filter_{run_id}"] = [normalized]
    if target_page == "Review":
        if requested_page == "Ticker Trends":
            review_view = "Trends"
        elif requested_page == "Results Dashboard":
            review_view = "Evidence"
        else:
            review_view = "Ticker Detail" if ticker else "Signals"
        if run_id:
            st.session_state[f"pending_review_view_{run_id}"] = review_view
    st.session_state["page"] = target_page
    st.rerun()


def render_ticker_navigation_buttons(*, current_page: str, run_id: str, ticker: str) -> None:
    current_canonical = canonical_page_name(current_page)
    buttons = [
        ("Review", "Open Review"),
        ("Validate", "Open Validate"),
    ]
    cols = st.columns(len(buttons))
    for column, (page_name, label) in zip(cols, buttons):
        if page_name == current_canonical:
            column.button(label, disabled=True, use_container_width=True, key=f"nav_disabled_{current_page}_{run_id}_{ticker}_{label}")
            continue
        if column.button(label, use_container_width=True, key=f"nav_{page_name}_{run_id}_{ticker}_{label}"):
            navigate_to_page(page_name, run_id=run_id, ticker=ticker)


def render_quick_ticker_jump_strip(
    summary_df: pd.DataFrame,
    *,
    run_id: str,
    target_page: str,
    title: str,
    limit: int = 6,
) -> None:
    if summary_df.empty:
        return
    top_rows = (
        summary_df.sort_values(["emerging_ticker_score", "average_alpha_signal_score"], ascending=[False, False])
        .head(limit)
        .to_dict(orient="records")
    )
    if not top_rows:
        return
    st.write(title)
    cols = st.columns(len(top_rows))
    for column, row in zip(cols, top_rows):
        ticker = str(row.get("ticker", ""))
        score = float(row.get("emerging_ticker_score", 0) or 0)
        if column.button(f"{ticker} · {score:.1f}", use_container_width=True, key=f"quick_jump_{target_page}_{run_id}_{ticker}"):
            navigate_to_page(target_page, run_id=run_id, ticker=ticker)


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
    "acceleration_score": "Coverage-adjusted acceleration",
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
    "mention_rate_per_100_posts": "Mentions per 100 posts",
    "acceleration_score": "Coverage-adjusted acceleration",
    "emerging_ticker_score": "Emerging ticker score",
    "average_alpha_signal_score": "Average alpha signal",
    "source_diversity_score": "Source diversity",
    "coverage_reliability": "Trend reliability",
}
METRIC_DOMAINS = {
    "acceleration_score": [0, 100],
    "coverage_adjusted_acceleration_score": [0, 100],
    "emerging_ticker_score": [0, 100],
    "average_alpha_signal_score": [0, 100],
    "average_hype_adjusted_signal_score": [0, 100],
    "average_hype_score": [0, 10],
    "average_depth_score": [0, 10],
    "average_claim_specificity_score": [0, 10],
    "source_diversity_score": [0, 10],
    "coverage_reliability": [0, 1],
    "controversy_score": [0, 10],
    "net_sentiment_score": [-1, 1],
}
COMPARISON_COLUMNS = [
    "total_mentions",
    "mentions_this_period",
    "mention_change",
    "mention_rate_per_100_posts",
    "acceleration_score",
    "coverage_reliability",
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
            lambda row: f"{format_delta(row['total_mentions_delta'])} mentions / {format_delta(row['acceleration_score_delta'], decimals=1)} adj. accel",
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
                "label": "Adjusted acceleration",
                "value": str(row["ticker"]),
                "note": f"{format_delta(row.get('acceleration_score_delta'), decimals=1)} coverage-adjusted acceleration",
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
    current_page: str = "Ticker Trends",
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
        metric_cols[2].metric("Comparable accel.", f"{float(ticker_row.get('acceleration_score', 0)):.1f}")
        metric_cols[3].metric("Evidence score", f"{float(ticker_row.get('average_alpha_signal_score', 0)):.1f}")
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
        render_ticker_navigation_buttons(current_page=current_page, run_id=run_id, ticker=selected_ticker)

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


def list_cell_values(value: Any, *, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
    elif str(value or "").strip():
        values = [str(value).strip()]
    else:
        values = []
    deduped = list(dict.fromkeys(values))
    return deduped[:limit] if limit is not None else deduped


def filter_ticker_window_mentions(mentions_df: pd.DataFrame, ticker: str, window_days: int | None) -> pd.DataFrame:
    if mentions_df.empty:
        return mentions_df
    working = mentions_df[mentions_df["ticker"].astype(str).str.upper() == ticker].copy()
    if working.empty:
        return working
    working["created_time_dt"] = pd.to_datetime(working["created_time"], errors="coerce", utc=True)
    working = working.dropna(subset=["created_time_dt"])
    if working.empty:
        return working
    if window_days is not None:
        anchor = working["created_time_dt"].max()
        working = working[working["created_time_dt"] >= anchor - pd.Timedelta(days=window_days)].copy()
    return working.sort_values(["created_time_dt", "alpha_signal_score"], ascending=[False, False])


def build_ticker_explorer_stats(ticker_mentions: pd.DataFrame, ticker_summary_row: dict[str, Any]) -> dict[str, Any]:
    if ticker_mentions.empty:
        return {
            "mentions": 0,
            "items": 0,
            "authors": 0,
            "threads": 0,
            "sources": 0,
            "bullish_mentions": 0,
            "bearish_mentions": 0,
            "neutral_mentions": 0,
            "average_hype": 0.0,
            "average_evidence_quality": 0.0,
            "average_alpha": 0.0,
            "fallback_mentions": 0,
            "ollama_mentions": 0,
            "first_seen": "",
            "last_seen": "",
            "latest_run_id": "",
        }

    working = ticker_mentions.copy()
    modes = working["classifier_mode"].fillna("unknown").astype(str).str.lower() if "classifier_mode" in working.columns else pd.Series([], dtype=str)
    latest_run_id = ""
    if "run_completed_at" in working.columns and "analysis_run_id" in working.columns:
        run_order = working[["analysis_run_id", "run_completed_at"]].dropna().drop_duplicates().copy()
        run_order["run_completed_at_dt"] = pd.to_datetime(run_order["run_completed_at"], errors="coerce", utc=True)
        run_order = run_order.dropna(subset=["run_completed_at_dt"]).sort_values("run_completed_at_dt")
        if not run_order.empty:
            latest_run_id = str(run_order.iloc[-1]["analysis_run_id"])
    if not latest_run_id and "analysis_run_id" in working.columns:
        latest_run_id = str(working.iloc[0].get("analysis_run_id", ""))

    return {
        "mentions": int(len(working)),
        "items": int(working["item_id"].nunique()) if "item_id" in working.columns else int(len(working)),
        "authors": int(working["author_hash"].nunique()) if "author_hash" in working.columns else 0,
        "threads": int(working["thread_id"].nunique()) if "thread_id" in working.columns else 0,
        "sources": int(working["source_name"].nunique()) if "source_name" in working.columns else 0,
        "bullish_mentions": int((working["sentiment"] == "bullish").sum()) if "sentiment" in working.columns else 0,
        "bearish_mentions": int((working["sentiment"] == "bearish").sum()) if "sentiment" in working.columns else 0,
        "neutral_mentions": int((working["sentiment"] == "neutral").sum()) if "sentiment" in working.columns else 0,
        "average_hype": round(coerce_float(ticker_summary_row.get("average_hype_score", working.get("hype_score", pd.Series([0])).mean())), 2),
        "average_evidence_quality": round(coerce_float(ticker_summary_row.get("average_evidence_quality_numeric", working.get("evidence_quality_numeric", pd.Series([0])).mean())), 2),
        "average_alpha": round(coerce_float(ticker_summary_row.get("average_alpha_signal_score", working.get("alpha_signal_score", pd.Series([0])).mean())), 2),
        "fallback_mentions": int((modes == "fallback").sum()) if not modes.empty else 0,
        "ollama_mentions": int((modes == "ollama").sum()) if not modes.empty else 0,
        "first_seen": str(working["created_time"].min()) if "created_time" in working.columns else "",
        "last_seen": str(working["created_time"].max()) if "created_time" in working.columns else "",
        "latest_run_id": latest_run_id,
    }


def select_representative_mentions(ticker_mentions: pd.DataFrame, limit: int = TICKER_CONSENSUS_EVIDENCE_LIMIT) -> pd.DataFrame:
    if ticker_mentions.empty:
        return ticker_mentions
    working = ticker_mentions.copy()
    if "created_time_dt" not in working.columns:
        working["created_time_dt"] = pd.to_datetime(working["created_time"], errors="coerce", utc=True)
    selected: dict[str, dict[str, Any]] = {}

    def add_rows(frame: pd.DataFrame, max_rows: int) -> None:
        for row in frame.head(max_rows).to_dict(orient="records"):
            key = str(row.get("item_id", "")) or f"row_{len(selected)}"
            if key not in selected:
                selected[key] = row
            if len(selected) >= limit:
                return

    add_rows(working.sort_values(["alpha_signal_score", "confidence"], ascending=False), 4)
    add_rows(working.sort_values("created_time_dt", ascending=False), 3)
    if "sentiment" in working.columns:
        add_rows(working[working["sentiment"] == "bullish"].sort_values("alpha_signal_score", ascending=False), 2)
        add_rows(working[working["sentiment"] == "bearish"].sort_values("alpha_signal_score", ascending=False), 2)
    if "red_flag_count" in working.columns:
        add_rows(working[working["red_flag_count"].fillna(0).astype(float) > 0].sort_values("hype_score", ascending=False), 2)
    if len(selected) < limit:
        add_rows(working.sort_values(["evidence_quality_numeric", "alpha_signal_score"], ascending=False), limit - len(selected))

    evidence = pd.DataFrame(selected.values())
    if evidence.empty:
        return evidence
    return evidence.sort_values(["alpha_signal_score", "created_time_dt"], ascending=[False, False]).head(limit)


def evidence_records_for_packet(evidence_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in evidence_df.to_dict(orient="records"):
        records.append(
            {
                "item_id": str(row.get("item_id", "")),
                "created_time": str(row.get("created_time", ""))[:19],
                "source": str(row.get("source_name", "")),
                "sentiment": str(row.get("sentiment", "")),
                "confidence": round(coerce_float(row.get("confidence")), 2),
                "alpha": round(coerce_float(row.get("alpha_signal_score")), 1),
                "hype": round(coerce_float(row.get("hype_score")), 1),
                "evidence_quality": str(row.get("evidence_quality", "")),
                "title": str(row.get("thread_title", ""))[:180],
                "summary": str(row.get("summary", ""))[:260],
                "claims_to_verify": list_cell_values(row.get("claims_to_verify"), limit=3),
                "red_flags": list_cell_values(row.get("red_flags"), limit=3),
            }
        )
    return records


def build_ticker_deterministic_shift(history_df: pd.DataFrame, ticker: str) -> str:
    if history_df.empty:
        return "No prior run history is available for this ticker yet."
    ticker_history = history_df[history_df["ticker"].astype(str).str.upper() == ticker].sort_values("run_completed_at").copy()
    if len(ticker_history) < 2:
        return "Only one completed run contains this ticker, so there is no clean before/after yet."
    previous = ticker_history.iloc[-2]
    current = ticker_history.iloc[-1]
    mention_delta = coerce_float(current.get("total_mentions")) - coerce_float(previous.get("total_mentions"))
    sentiment_delta = coerce_float(current.get("net_sentiment_score")) - coerce_float(previous.get("net_sentiment_score"))
    hype_delta = coerce_float(current.get("average_hype_score")) - coerce_float(previous.get("average_hype_score"))
    return (
        f"Latest run moved {format_delta(mention_delta)} mentions, "
        f"{format_delta(sentiment_delta, decimals=2)} net sentiment, "
        f"and {format_delta(hype_delta, decimals=1)} hype versus the previous completed run."
    )


def build_ticker_consensus_packet(
    *,
    ticker: str,
    company_name: str,
    window_label: str,
    stats: dict[str, Any],
    ticker_mentions: pd.DataFrame,
    evidence_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> dict[str, Any]:
    claims = extract_claims_for_ticker(ticker_mentions, ticker)
    bullish_claims = list_cell_values(
        ticker_mentions.loc[ticker_mentions["sentiment"] == "bullish", "bull_case"].dropna().head(6).tolist()
        if "sentiment" in ticker_mentions.columns and "bull_case" in ticker_mentions.columns
        else [],
        limit=5,
    )
    bearish_claims = list_cell_values(
        ticker_mentions.loc[ticker_mentions["sentiment"] == "bearish", "bear_case"].dropna().head(6).tolist()
        if "sentiment" in ticker_mentions.columns and "bear_case" in ticker_mentions.columns
        else [],
        limit=5,
    )
    return {
        "ticker": ticker,
        "company_name": company_name,
        "window": window_label,
        "stats": stats,
        "deterministic_shift": build_ticker_deterministic_shift(history_df, ticker),
        "claims_to_verify": claims,
        "bull_claims": bullish_claims,
        "bear_claims": bearish_claims,
        "evidence": evidence_records_for_packet(evidence_df),
    }


def ticker_consensus_signature(
    *,
    ticker: str,
    window_label: str,
    stats: dict[str, Any],
    evidence_df: pd.DataFrame,
) -> str:
    signature_payload = {
        "ticker": ticker,
        "window": window_label,
        "latest_run_id": stats.get("latest_run_id", ""),
        "mentions": stats.get("mentions", 0),
        "last_seen": stats.get("last_seen", ""),
        "evidence_item_ids": sorted(evidence_df["item_id"].astype(str).tolist()) if not evidence_df.empty and "item_id" in evidence_df.columns else [],
    }
    return hashlib.sha1(json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:20]


def render_compact_list(title: str, values: list[str], *, empty: str = "None in this window.") -> None:
    st.markdown(f"**{title}**")
    clean_values = [str(value).strip() for value in values if str(value).strip()]
    if not clean_values:
        st.caption(empty)
        return
    items = "".join(f"<li>{escape(value)}</li>" for value in clean_values[:5])
    st.markdown(f'<ul class="compact-list">{items}</ul>', unsafe_allow_html=True)


def render_ticker_hero(ticker: str, company_name: str, window_label: str, stats: dict[str, Any], cache_label: str) -> None:
    last_seen = format_timestamp_with_relative(stats.get("last_seen"))
    st.markdown(
        f"""
        <div class="ticker-hero">
            <div class="ticker-hero-title">{escape(ticker)} <span style="font-size: 1rem; opacity: 0.7;">{escape(company_name)}</span></div>
            <div class="ticker-hero-meta">{escape(window_label)} · {escape(str(stats.get("mentions", 0)))} mention rows · last seen {escape(last_seen)} · {escape(cache_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticker_metric_strip(stats: dict[str, Any]) -> None:
    metric_cols = st.columns(6)
    metric_cols[0].metric("Mentions", int(stats.get("mentions", 0)), help="Ticker mention rows in the selected window.")
    metric_cols[1].metric("Authors", int(stats.get("authors", 0)), help="Distinct author hashes.")
    metric_cols[2].metric("Threads", int(stats.get("threads", 0)), help="Distinct Reddit threads.")
    metric_cols[3].metric("Bull / Bear", f"{int(stats.get('bullish_mentions', 0))} / {int(stats.get('bearish_mentions', 0))}")
    metric_cols[4].metric("Avg hype", f"{coerce_float(stats.get('average_hype')):.1f}")
    metric_cols[5].metric("Fallback", int(stats.get("fallback_mentions", 0)), help="Mentions classified without a usable Ollama response.")


def render_consensus_summary(
    *,
    payload: dict[str, Any],
    classifier_mode: str,
    generated_at: str,
    cache_label: str,
) -> None:
    summary = str(payload.get("consensus_summary") or "No consensus summary was generated.").strip()
    st.markdown(
        f"""
        <div class="consensus-card">
            <h3>Current consensus</h3>
            <p>{escape(summary)}</p>
            <div class="ticker-hero-meta">{escape(cache_label)} · generated {escape(format_timestamp_with_relative(generated_at))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_status_chips(
        [
            str(payload.get("stance", "unclear")).title(),
            f"Confidence: {str(payload.get('confidence', 'low')).title()}",
            str(classifier_mode or "unknown"),
        ]
    )
    case_cols = st.columns(2)
    with case_cols[0]:
        render_compact_list("Bull case", list_cell_values(payload.get("bull_case"), limit=4))
        render_compact_list("Catalysts", list_cell_values(payload.get("catalysts"), limit=4))
    with case_cols[1]:
        render_compact_list("Bear case", list_cell_values(payload.get("bear_case"), limit=4))
        render_compact_list("Risks", list_cell_values(payload.get("risks"), limit=4))


def render_narrative_and_claims(ticker: str, payload: dict[str, Any], ticker_mentions: pd.DataFrame) -> None:
    narrative_cols = st.columns([1.15, 0.85])
    with narrative_cols[0]:
        st.subheader("Narrative shift")
        st.write(str(payload.get("narrative_shift") or "No narrative shift is available yet."))
        red_flags = list_cell_values(payload.get("red_flags"), limit=5)
        if red_flags:
            render_status_chips(red_flags)
    with narrative_cols[1]:
        st.subheader("Claims")
        claims = list_cell_values(payload.get("claims_to_verify"), limit=6) or extract_claims_for_ticker(ticker_mentions, ticker)
        render_compact_list("To verify", claims, empty="No checkable claims were extracted.")
        with st.expander("Checklist", expanded=False):
            render_evidence_checklist(ticker, claims)


def build_ticker_bucket_chart_frame(ticker_mentions: pd.DataFrame) -> pd.DataFrame:
    if ticker_mentions.empty:
        return pd.DataFrame()
    bucket_df = build_time_bucket_summary(ticker_mentions)
    if bucket_df.empty:
        return bucket_df
    sentiment = (
        ticker_mentions.groupby(["ticker", "time_bucket"], as_index=False)["sentiment_numeric"]
        .mean()
        .rename(columns={"sentiment_numeric": "net_sentiment_score"})
        if "sentiment_numeric" in ticker_mentions.columns
        else pd.DataFrame()
    )
    if not sentiment.empty:
        bucket_df = bucket_df.merge(sentiment, on=["ticker", "time_bucket"], how="left")
    bucket_df["bucket_start_dt"] = pd.to_datetime(bucket_df["bucket_start"], errors="coerce", utc=True)
    return bucket_df.dropna(subset=["bucket_start_dt"])


def render_ticker_explorer_charts(ticker: str, ticker_mentions: pd.DataFrame, history_df: pd.DataFrame) -> None:
    st.subheader("Charts")
    chart_cols = st.columns([1.05, 0.95])
    bucket_chart_df = build_ticker_bucket_chart_frame(ticker_mentions)
    with chart_cols[0]:
        st.write("Mentions over time")
        if bucket_chart_df.empty:
            st.info("No bucketed timeline is available for this ticker/window.")
        else:
            render_time_line_chart(
                bucket_chart_df,
                x_column="bucket_start_dt",
                y_column="total_mentions",
                color_column="ticker",
                y_title="Mentions",
                x_title="Week",
                tooltip_columns=["ticker", "total_mentions", "unique_authors", "unique_threads"],
                height=300,
            )
    with chart_cols[1]:
        metric_options = {
            "net_sentiment_score": "Net sentiment",
            "average_hype_score": "Hype",
            "average_evidence_quality_numeric": "Evidence quality",
            "average_alpha_signal_score": "Alpha",
        }
        selected_metric = st.selectbox(
            "Trend metric",
            [metric for metric in metric_options if metric in bucket_chart_df.columns],
            format_func=lambda value: metric_options[value],
            key=f"explorer_bucket_metric_{ticker}",
        ) if not bucket_chart_df.empty else ""
        if selected_metric:
            render_time_line_chart(
                bucket_chart_df,
                x_column="bucket_start_dt",
                y_column=selected_metric,
                color_column="ticker",
                y_title=metric_options[selected_metric],
                x_title="Week",
                tooltip_columns=["ticker", selected_metric, "total_mentions"],
                height=300,
            )

    if not history_df.empty:
        st.write("Run history")
        history_metric = st.selectbox(
            "Run metric",
            [metric for metric in HISTORY_METRICS if metric in history_df.columns],
            format_func=lambda value: HISTORY_METRICS[value],
            key=f"explorer_history_metric_{ticker}",
        )
        render_history_chart(history_df, [ticker], history_metric)


def render_ticker_evidence_drawers(ticker_mentions: pd.DataFrame, evidence_df: pd.DataFrame) -> None:
    with st.expander("Top evidence", expanded=False):
        if evidence_df.empty:
            st.caption("No representative evidence rows are available.")
        else:
            display = evidence_df.copy()
            for column in ("claims_to_verify", "red_flags"):
                if column in display.columns:
                    display[column] = display[column].map(lambda value: ", ".join(list_cell_values(value, limit=4)))
            columns = [
                "created_time",
                "source_name",
                "sentiment",
                "confidence",
                "alpha_signal_score",
                "hype_score",
                "evidence_quality",
                "thread_title",
                "summary",
                "claims_to_verify",
                "red_flags",
                "permalink",
            ]
            st.dataframe(
                display[[column for column in columns if column in display.columns]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "permalink": st.column_config.LinkColumn("permalink"),
                    "thread_title": st.column_config.TextColumn("Title", width="large"),
                    "summary": st.column_config.TextColumn("Summary", width="large"),
                },
            )

    with st.expander("Raw mentions", expanded=False):
        if ticker_mentions.empty:
            st.caption("No raw mention rows are available.")
            return
        raw = ticker_mentions.copy()
        for column in ("claims_to_verify", "red_flags"):
            if column in raw.columns:
                raw[column] = raw[column].map(lambda value: ", ".join(list_cell_values(value, limit=5)))
        raw_columns = [
            "created_time",
            "analysis_run_id",
            "source_name",
            "sentiment",
            "confidence",
            "classifier_mode",
            "alpha_signal_score",
            "hype_score",
            "evidence_quality",
            "thread_title",
            "summary",
            "body_text",
            "permalink",
        ]
        st.dataframe(
            raw[[column for column in raw_columns if column in raw.columns]].head(300),
            use_container_width=True,
            hide_index=True,
            column_config={
                "permalink": st.column_config.LinkColumn("permalink"),
                "thread_title": st.column_config.TextColumn("Title", width="large"),
                "body_text": st.column_config.TextColumn("Text", width="large"),
            },
        )


def render_ticker_explorer_page(db_path: Path, *, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Ticker Detail")
    else:
        render_page_header("Ticker Detail", "Review one ticker across the completed corpus.")
    long_df, summary_df, _, corpus_summary = build_corpus_frames(db_path)
    if summary_df.empty or long_df.empty:
        st.info("No completed ticker corpus is available yet. Run a scrape that finds ticker mentions, then return here.")
        return

    summary_df = summary_df.sort_values(["total_mentions", "emerging_ticker_score"], ascending=[False, False]).copy()
    ticker_options = summary_df["ticker"].dropna().astype(str).tolist()
    company_lookup = dict(zip(summary_df["ticker"].astype(str), summary_df["company_name"].astype(str), strict=False))
    focused = normalize_ticker(str(st.session_state.get("ticker_focus", "")))
    default_ticker = focused if focused in ticker_options else ticker_options[0]
    default_index = ticker_options.index(default_ticker)

    search_cols = st.columns([1.35, 0.95, 0.7])
    selected_ticker = search_cols[0].selectbox(
        "Ticker",
        ticker_options,
        index=default_index,
        format_func=lambda value: f"{value} · {company_lookup.get(value, 'Unknown')}",
        key="ticker_explorer_search",
        help="Search any ticker found in completed runs.",
    )
    st.session_state["ticker_focus"] = selected_ticker
    window_label = search_cols[1].radio(
        "Window",
        list(TICKER_EXPLORER_WINDOWS.keys()),
        index=1,
        horizontal=True,
        key="ticker_explorer_window",
        help="Windows are anchored to the ticker's latest collected mention.",
    )
    refresh_clicked = search_cols[2].button("Refresh", use_container_width=True, help="Regenerate the cached consensus for this ticker/window.")

    with st.expander("Consensus settings", expanded=False):
        setting_cols = st.columns(2)
        model_name = setting_cols[0].text_input("Ollama model", DEFAULT_MODEL, key="ticker_explorer_model")
        ollama_url = setting_cols[1].text_input("Ollama URL", DEFAULT_OLLAMA_URL, key="ticker_explorer_ollama_url")

    window_mentions = filter_ticker_window_mentions(long_df, selected_ticker, TICKER_EXPLORER_WINDOWS[window_label])
    if window_mentions.empty:
        st.info(f"No `{selected_ticker}` mentions matched `{window_label}`.")
        return

    window_summary_df = build_ticker_summaries(window_mentions)
    ticker_summary_row = window_summary_df.iloc[0].to_dict() if not window_summary_df.empty else {}
    company_name = str(ticker_summary_row.get("company_name") or company_lookup.get(selected_ticker) or "Unknown")
    stats = build_ticker_explorer_stats(window_mentions, ticker_summary_row)

    sources = load_sources(db_path)
    source_lookup = {
        int(row["source_id"]): str(row["display_name"])
        for row in sources.to_dict(orient="records")
    } if not sources.empty else {}
    history_raw = load_ticker_history(db_path, selected_ticker, data_mode="live")
    run_label_lookup = {
        str(row["analysis_run_id"]): build_run_display_label(row, source_lookup)
        for row in load_analysis_runs(db_path).to_dict(orient="records")
    }
    history_df = prepare_historical_summary_frame(history_raw, run_label_lookup) if not history_raw.empty else pd.DataFrame()

    evidence_df = select_representative_mentions(window_mentions)
    packet = build_ticker_consensus_packet(
        ticker=selected_ticker,
        company_name=company_name,
        window_label=window_label,
        stats=stats,
        ticker_mentions=window_mentions,
        evidence_df=evidence_df,
        history_df=history_df,
    )
    scope_key = f"ticker_explorer:{window_label.lower().replace(' ', '_')}"
    evidence_signature = ticker_consensus_signature(
        ticker=selected_ticker,
        window_label=window_label,
        stats=stats,
        evidence_df=evidence_df,
    )
    cached = load_ticker_consensus_cache(
        db_path,
        ticker=selected_ticker,
        scope_key=scope_key,
        model_name=model_name,
    )
    cache_current = bool(cached and cached.get("evidence_signature") == evidence_signature)
    generated_now = False
    if refresh_clicked or not cache_current:
        with st.spinner("Generating compact consensus..."):
            consensus_result = OllamaClassifier(model=model_name, endpoint=ollama_url, timeout=45, retries=0).summarize_ticker_consensus(packet)
        generated_now = True
        generated_at = analysis_run_now_iso()
        consensus_payload = consensus_result.payload
        classifier_mode = consensus_result.mode
        upsert_ticker_consensus_cache(
            db_path,
            {
                "ticker": selected_ticker,
                "scope_key": scope_key,
                "model_name": model_name,
                "ollama_url": ollama_url,
                "latest_run_id": stats.get("latest_run_id", ""),
                "evidence_signature": evidence_signature,
                "summary_json": consensus_payload,
                "classifier_mode": classifier_mode,
                "evidence_item_ids": evidence_df["item_id"].astype(str).tolist() if not evidence_df.empty and "item_id" in evidence_df.columns else [],
                "generated_at": generated_at,
            },
        )
    else:
        consensus_payload = cached.get("summary_json", {}) if cached else {}
        classifier_mode = str(cached.get("classifier_mode", "")) if cached else ""
        generated_at = str(cached.get("generated_at", "")) if cached else ""

    cache_label = "Generated now" if generated_now else "Fresh cache" if cache_current else "Needs refresh"
    render_ticker_hero(selected_ticker, company_name, window_label, stats, cache_label)
    watchlist = get_watchlist()
    pin_cols = st.columns([0.7, 4.3])
    if selected_ticker in watchlist:
        if pin_cols[0].button("Unpin", key=f"explorer_unpin_{selected_ticker}"):
            update_watchlist([ticker for ticker in watchlist if ticker != selected_ticker])
            st.rerun()
    elif pin_cols[0].button("Pin", key=f"explorer_pin_{selected_ticker}"):
        update_watchlist([*watchlist, selected_ticker])
        st.rerun()
    pin_cols[1].caption(
        f"Corpus scope: {int(corpus_summary.get('runs_included', 0) or 0)} completed run(s), "
        f"{int(corpus_summary.get('ticker_mentions', 0) or 0)} ticker mention rows."
    )

    render_ticker_metric_strip(stats)
    render_consensus_summary(
        payload=consensus_payload,
        classifier_mode=classifier_mode,
        generated_at=generated_at,
        cache_label=cache_label,
    )
    render_ticker_explorer_charts(selected_ticker, window_mentions, history_df)
    render_narrative_and_claims(selected_ticker, consensus_payload, window_mentions)
    render_ticker_evidence_drawers(window_mentions, evidence_df)


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
                "mention_rate_per_100_posts:Q",
                "trend_reliability:N",
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
            x=alt.X("acceleration_score:Q", title="Coverage-adjusted acceleration", scale=alt.Scale(domain=[0, 100], nice=True)),
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
                "mention_rate_per_100_posts:Q",
                "trend_reliability:N",
                "mentions_this_period:Q",
                "mentions_previous_period:Q",
                "average_hype_score:Q",
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def render_ticker_summary_table(summary_df: pd.DataFrame, *, run_id: str, current_page: str) -> None:
    if summary_df.empty:
        st.info("No ticker summary rows are available. Run a scrape that finds ticker mentions, then return to this table.")
        return
    display_source = summary_df.copy()
    watchlist = set(get_watchlist())
    display_source["watched"] = display_source["ticker"].astype(str).isin(watchlist)
    sort_presets = {
        "Best emerging": ["emerging_ticker_score", "average_alpha_signal_score"],
        "Highest acceleration": ["acceleration_score", "emerging_ticker_score"],
        "Highest alpha": ["average_alpha_signal_score", "emerging_ticker_score"],
        "Lowest hype": ["average_hype_score", "average_alpha_signal_score"],
        "Most mentioned": ["total_mentions", "emerging_ticker_score"],
    }
    sort_choice = st.selectbox(
        "Sort summary by",
        list(sort_presets.keys()),
        key=f"summary_sort_{current_page}_{run_id}",
    )
    sort_columns = [column for column in sort_presets[sort_choice] if column in display_source.columns]
    ascending = [sort_choice == "Lowest hype"] + [False] * max(len(sort_columns) - 1, 0)
    if sort_columns:
        display_source = display_source.sort_values(sort_columns, ascending=ascending[: len(sort_columns)])
    render_ticker_action_strip(
        display_source,
        run_id=run_id,
        current_page=current_page,
        key_prefix=f"summary_actions_{current_page}",
    )
    compact_columns = [
        "watched",
        "ticker",
        "company_name",
        "total_mentions",
        "acceleration_score",
        "trend_reliability",
        "emerging_ticker_score",
        "average_alpha_signal_score",
        "average_hype_score",
        "low_mentions_high_signal",
        "new_ticker_detected",
    ]
    detail_columns = [
        "watched",
        "ticker",
        "company_name",
        "total_mentions",
        "mention_rate_per_100_posts",
        "mentions_this_period",
        "mentions_previous_period",
        "mention_change",
        "acceleration_score",
        "trend_reliability",
        "coverage_reliability",
        "emerging_ticker_score",
        "average_alpha_signal_score",
        "average_hype_score",
        "source_diversity_score",
        "controversy_score",
        "low_mentions_high_signal",
        "new_ticker_detected",
    ]
    column_config = {
        "watched": st.column_config.CheckboxColumn("Watch"),
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "company_name": st.column_config.TextColumn("Company", width="medium"),
        "mention_rate_per_100_posts": st.column_config.NumberColumn("Mentions / 100 posts", format="%.2f"),
        "acceleration_score": st.column_config.NumberColumn("Comparable accel.", format="%.1f", help=metric_help("acceleration_score")),
        "trend_reliability": st.column_config.TextColumn("Trend trust"),
        "coverage_reliability": st.column_config.NumberColumn("Scrape coverage", format="%.2f", help=metric_help("coverage_reliability")),
        "emerging_ticker_score": st.column_config.NumberColumn("Emerging", format="%.1f", help=metric_help("emerging_ticker_score")),
        "average_alpha_signal_score": st.column_config.NumberColumn("Evidence score", format="%.1f"),
        "average_hype_score": st.column_config.NumberColumn("Hype", format="%.1f"),
        "source_diversity_score": st.column_config.NumberColumn("Source spread", format="%.1f"),
        "controversy_score": st.column_config.NumberColumn("Controversy", format="%.1f"),
    }
    compact_available = [column for column in compact_columns if column in display_source.columns]
    detail_available = [column for column in detail_columns if column in display_source.columns]
    st.dataframe(
        display_source[compact_available],
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        key=f"summary_compact_table_{current_page}_{run_id}",
    )
    with st.expander("Deeper ticker metrics", expanded=False):
        st.dataframe(
            display_source[detail_available],
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
            key=f"summary_detail_table_{current_page}_{run_id}",
        )


def render_results_mentions_review(long_df: pd.DataFrame, summary_df: pd.DataFrame, scope_id: str) -> None:
    render_all_tickers_panel(summary_df, long_df)
    render_metric_guide("Metric guide for Results", expanded=False, context="results")
    filtered = filter_results_dataframe(long_df, scope_id)
    st.write(f"{len(filtered)} mention rows matched the current filters.")
    if filtered.empty:
        render_filtered_empty_state(long_df)
        return

    display_columns = [
        "created_time",
        "source_name",
        "analysis_run_id",
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
    card_tab, thread_tab, mention_tab, detail_tab = st.tabs(["Evidence Cards", "Threads", "Mention Rows", "Details"])
    with card_tab:
        render_evidence_card_list(filtered)

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
        compact_mention_columns = [
            "created_time",
            "source_name",
            "ticker",
            "company_name",
            "sentiment",
            "trust_status",
            "confidence",
            "alpha_signal_score",
            "hype_adjusted_signal_score",
            "hype_score",
            "summary",
            "permalink",
        ]
        compact_mention_columns = [column for column in compact_mention_columns if column in filtered.columns]
        display_df = filtered[compact_mention_columns].rename(columns={"source_name": "source"})
        mention_column_config = {
            "permalink": st.column_config.LinkColumn("permalink"),
            "trust_status": st.column_config.TextColumn("trust"),
            "classifier_mode": st.column_config.TextColumn("mode"),
            "analysis_run_id": st.column_config.TextColumn("run", width="small"),
            "alpha_signal_score": st.column_config.NumberColumn("Alpha", format="%.1f"),
            "hype_adjusted_signal_score": st.column_config.NumberColumn("Hype-adjusted", format="%.1f"),
            "hype_score": st.column_config.NumberColumn("Hype", format="%.1f"),
            "source_diversity_score": st.column_config.NumberColumn("Source diversity", format="%.1f"),
            "controversy_score": st.column_config.NumberColumn("Controversy", format="%.1f"),
        }
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=mention_column_config,
        )
        with st.expander("Deeper mention metrics", expanded=False):
            full_display_df = filtered[display_columns].rename(columns={"source_name": "source"})
            st.dataframe(
                full_display_df,
                use_container_width=True,
                hide_index=True,
                column_config=mention_column_config,
            )

    with detail_tab:
        render_mention_details(filtered)


def render_results_dashboard_page(db_path: Path, run_id: str | None) -> None:
    st.header("Results Dashboard")
    st.caption("Use this page to review the actual Reddit evidence: which threads mentioned which tickers, how strong the signal looks, and what needs verification.")
    if not run_id:
        st.info("Run an analysis first.")
        return

    completed_notice = st.session_state.pop("completed_run_notice", None)
    if completed_notice:
        st.success(completed_notice)

    if is_corpus_scope(run_id):
        long_df, summary_df, _, summary = build_corpus_frames(db_path)
        st.subheader(CORPUS_SCOPE_LABEL)
        st.caption(
            "This combines all completed live runs into one deduped item-level dataset. "
            "If the same Reddit item/ticker appears in overlapping scrapes, the newest completed classification is kept. "
            "Coverage-adjusted acceleration is reserved for matched Measurement runs."
        )
        render_corpus_summary_cards(summary)
        render_top_signal_cards(summary_df, run_id=run_id, current_page="Results Dashboard")
        render_quick_ticker_jump_strip(
            summary_df,
            run_id=run_id,
            target_page="Ticker Trends",
            title="Quick open top leads in Trends",
        )
        if long_df.empty:
            st.info("No completed ticker mentions are available in the combined corpus yet. Run a live scrape that finds ticker mentions, then return to Results.")
            return
        render_results_mentions_review(long_df, summary_df, run_id)
        return

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
            st.warning("This run completed, but no Reddit items matched the selected sources and time window. Broaden the time window or add active sources, then run another scrape.")
        elif summary:
            st.info(
                f"This run analysed {int(summary.get('items_analyzed', 0))} Reddit items, "
                "but no catalog tickers or company-name matches were found. Add ticker aliases or scrape sources with clearer stock discussion."
            )
        else:
            st.info("No item-level mention data saved for this run. Run a scrape first, then return to Results.")
        return

    long_df = attach_quality_columns(long_df, load_run_classification_quality(db_path, run_id))
    summary_df = load_run_ticker_summaries(db_path, run_id)
    render_top_signal_cards(summary_df, run_id=run_id, current_page="Results Dashboard")
    render_quick_ticker_jump_strip(
        summary_df,
        run_id=run_id,
        target_page="Ticker Trends",
        title="Quick open top leads in Trends",
    )
    render_results_mentions_review(long_df, summary_df, run_id)


def render_ticker_trends_page(db_path: Path, run_id: str | None, *, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Trends")
    else:
        render_page_header(
            "Trends",
            "Use this page to answer what changed between scrapes, which tickers are accelerating, and which leads deserve a closer read.",
        )
    if not run_id:
        st.info("Run an analysis first.")
        return

    if is_corpus_scope(run_id):
        long_df, summary_df, bucket_df, summary = build_corpus_frames(db_path)
        st.subheader(CORPUS_SCOPE_LABEL)
        st.caption(
            "The combined timeline uses Reddit item dates across every completed live run, with overlapping item/ticker mentions counted once. "
            "Use individual Measurement runs for coverage-adjusted acceleration."
        )
        if summary_df.empty:
            st.info("No completed ticker history is available in the combined corpus yet. Run at least one live scrape with ticker mentions, then return to Trends.")
            return

        metric_cols = st.columns(5)
        metric_cols[0].metric("Runs", int(summary.get("runs_included", 0)))
        metric_cols[1].metric("Tickers", int(summary_df["ticker"].nunique()))
        metric_cols[2].metric("Mentions", int(summary_df["total_mentions"].sum()))
        metric_cols[3].metric("Best adj. accel.", f"{float(summary_df['acceleration_score'].max()):.1f}")
        metric_cols[4].metric("Overlap rows removed", int(summary.get("deduplicated_mentions_removed", 0)))
        render_all_tickers_panel(summary_df, long_df)
        render_metric_guide("Metric guide for Trends", expanded=False, context="trends")
        render_quick_ticker_jump_strip(
            summary_df,
            run_id=run_id,
            target_page="Signal Validation",
            title="Quick open top leads in Validate",
        )
        render_ticker_focus(
            run_id=run_id,
            summary_df=summary_df,
            bucket_df=bucket_df,
            long_df=long_df,
            history_df=pd.DataFrame(),
            current_page="Ticker Trends",
        )

        timeline_tab, map_tab, table_tab, run_tab = st.tabs(
            ["Corpus Timeline", "Signal Map", "Summary Table", "Run Coverage"]
        )
        with timeline_tab:
            st.subheader("All completed runs timeline")
            if bucket_df.empty:
                st.info("No time buckets are available for the combined corpus.")
            else:
                timeline_controls = st.columns([2, 1])
                selected_tickers = timeline_controls[0].multiselect(
                    "Tickers",
                    summary_df["ticker"].tolist(),
                    default=summary_df["ticker"].head(min(8, len(summary_df))).tolist(),
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

        with map_tab:
            st.subheader("Ticker signal overview")
            map_cols = st.columns([1.65, 1])
            with map_cols[0]:
                render_signal_map(summary_df)
            with map_cols[1]:
                st.write("Top comparable acceleration")
                render_acceleration_bar(summary_df)

        with table_tab:
            st.subheader("Ticker summary")
            render_ticker_summary_table(summary_df, run_id=run_id, current_page="Ticker Trends")

        with run_tab:
            runs = load_analysis_runs(db_path)
            if runs.empty:
                st.info("No runs saved yet.")
            else:
                completed = runs[
                    (runs["status"] == "completed")
                    & (runs["data_mode"] == "live")
                    & (runs.apply(lambda row: run_has_analysed_items(row.to_dict()), axis=1))
                ].copy()
                if completed.empty:
                    st.info("No completed runs with analysed items yet.")
                else:
                    completed["items_analyzed"] = completed["summary_json"].map(lambda value: int((value or {}).get("items_analyzed", 0) or 0))
                    completed["ticker_mentions"] = completed["summary_json"].map(lambda value: int((value or {}).get("ticker_mentions", 0) or 0))
                    completed["tickers_found"] = completed["summary_json"].map(lambda value: int((value or {}).get("tickers_found", 0) or 0))
                    st.dataframe(
                        completed[
                            [
                                "analysis_run_id",
                                "completed_at",
                                "time_window_label",
                                "items_analyzed",
                                "ticker_mentions",
                                "tickers_found",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
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
        st.info("No ticker summary data saved for this run. Run a scrape that finds ticker mentions, then return to Trends.")
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
    metric_cols[3].metric("Best adj. accel.", f"{float(summary_df['acceleration_score'].max()):.1f}")
    metric_cols[4].metric("Low-hype leads", int(summary_df["low_mentions_high_signal"].sum()))
    render_all_tickers_panel(summary_df, long_df)
    render_metric_guide("Metric guide for Trends", expanded=False, context="trends")
    render_quick_ticker_jump_strip(
        summary_df,
        run_id=run_id,
        target_page="Signal Validation",
        title="Quick open top leads in Validate",
    )

    historical_summary_df = load_historical_ticker_summaries(
        db_path,
        data_mode=str(run_record["data_mode"]),
        selected_source_ids_json=json.dumps(run_record.get("selected_source_ids", []), ensure_ascii=True),
    )
    if str(run_record.get("run_type", RUN_TYPE_DISCOVERY)).lower() == RUN_TYPE_MEASUREMENT and not historical_summary_df.empty:
        historical_summary_df = historical_summary_df[
            (historical_summary_df["run_type"].astype(str).str.lower() == RUN_TYPE_MEASUREMENT)
            & (historical_summary_df["canonical_trend_run"].fillna(False).astype(bool))
        ].copy()
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
        current_page="Ticker Trends",
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
                        "acceleration_score_baseline": st.column_config.NumberColumn("Base adj. accel.", format="%.1f"),
                        "acceleration_score_current": st.column_config.NumberColumn("Current adj. accel.", format="%.1f"),
                        "acceleration_score_delta": st.column_config.NumberColumn("Adj. accel. change", format="%+.1f"),
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
            st.write("Top comparable acceleration")
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
        render_ticker_summary_table(summary_df, run_id=run_id, current_page="Ticker Trends")


def render_today_page(db_path: Path, run_id: str | None, source_lookup: dict[int, str]) -> None:
    render_page_header(
        "Today",
        "Current state, strongest leads, reliability notes, and the next useful action.",
    )
    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
    sources = load_sources(db_path)
    active_run = latest_active_run(runs) if not runs.empty else None
    latest_run = latest_completed_run(runs)
    watchlist = get_watchlist()
    signal_review_df = load_signal_review_frame(db_path, CORPUS_SCOPE_ID)

    active_source_count = int(sources["active"].sum()) if not sources.empty and "active" in sources.columns else 0
    labelled_count = (
        int(signal_review_df["user_label"].fillna("").astype(str).str.len().gt(0).sum())
        if not signal_review_df.empty and "user_label" in signal_review_df.columns
        else 0
    )
    unlabelled_count = max(len(signal_review_df) - labelled_count, 0) if not signal_review_df.empty else 0
    latest_summary = latest_run.get("summary_json", {}) if latest_run else {}
    latest_coverage = (
        latest_summary.get("coverage", {})
        if latest_run and isinstance(latest_summary.get("coverage", {}), dict)
        else {}
    )
    render_insight_cards(
        [
            {
                "label": "Current run",
                "value": str(active_run.get("status", "Running")).title() if active_run else "Idle",
                "note": run_elapsed_label(active_run) if active_run else "No live scrape is active",
            },
            {
                "label": "Latest completed",
                "value": run_completed_label(latest_run) if latest_run else "None",
                "note": f"{int(latest_summary.get('items_analyzed', 0) or 0)} analysed items" if latest_run else "Start in Collect",
            },
            {
                "label": "Scrape coverage",
                "value": str(latest_coverage.get("reliability_label", "n/a") or "n/a"),
                "note": f"{int(latest_coverage.get('posts_returned', 0) or 0)} posts returned",
            },
            {
                "label": "Needs labels",
                "value": str(unlabelled_count),
                "note": f"{len(signal_review_df)} frozen signal(s)",
            },
        ]
    )

    if active_run:
        st.subheader("Current Run")
        render_latest_run_status(runs, source_lookup)
    elif latest_run:
        st.subheader("Latest Completed Run")
        render_latest_run_status(runs, source_lookup)
    else:
        st.info("No completed run exists yet. Add sources and start a Measurement or Discovery run from Collect.")

    top_scope_id = run_id
    if not top_scope_id and latest_run:
        top_scope_id = str(latest_run.get("analysis_run_id", ""))
    summary_df = load_scope_ticker_summaries(db_path, top_scope_id) if top_scope_id else pd.DataFrame()
    if not summary_df.empty and top_scope_id:
        render_top_signal_cards(summary_df, run_id=str(top_scope_id), current_page="Today", limit=5)
    else:
        st.subheader("Top Signals")
        st.caption("No ticker summary is available for the current scope yet.")

    watch_cols = st.columns([1.3, 2.7])
    with watch_cols[0]:
        st.subheader("Watchlist")
        if watchlist:
            render_copyable_ticker_chips(watchlist, label="Pinned tickers", max_items=16)
        else:
            st.caption("No pinned tickers yet.")
    with watch_cols[1]:
        st.subheader("System Notes")
        notes: list[str] = []
        if latest_run and int(latest_summary.get("fallback_count", 0) or 0):
            notes.append(f"{int(latest_summary.get('fallback_count', 0) or 0)} fallback classifications in the latest completed run.")
        trend_note = str(latest_coverage.get("trend_note", "") or "")
        if trend_note:
            notes.append(trend_note)
        if not notes:
            notes.append("No current reliability warnings from the latest completed run.")
        for note in notes:
            st.caption(note)

    if sources.empty or active_source_count == 0:
        action_label, target_page = "Add sources", "Collect"
        action_note = "Sources are required before any run can start."
    elif active_run:
        action_label, target_page = "Check progress", "Collect"
        action_note = "A run is active; watch progress before starting another."
    elif not latest_run:
        action_label, target_page = "Start first run", "Collect"
        action_note = "Use Measurement for comparable trends or Discovery for broader evidence."
    elif unlabelled_count:
        action_label, target_page = "Review labels", "Validate"
        action_note = "Frozen signals are waiting for human labels."
    else:
        action_label, target_page = "Review signals", "Review"
        action_note = "Open the current scope and inspect the strongest evidence."

    st.subheader("Recommended Next Action")
    st.caption(action_note)
    if st.button(action_label, type="primary", use_container_width=True):
        st.session_state["page"] = target_page
        if target_page == "Collect":
            st.session_state["pending_collect_view"] = "Sources" if action_label == "Add sources" else "Run Progress" if active_run else "New Run"
        st.rerun()


def render_collect_page(db_path: Path, artifacts_dir: Path, ticker_path: Path, source_lookup: dict[int, str]) -> None:
    render_page_header(
        "Collect",
        "Choose sources, set the run purpose, preview request load, then start the background scrape.",
    )
    collect_sections = ["Sources", "New Run", "Run Progress", "History"]
    pending_collect_view = st.session_state.pop("pending_collect_view", None)
    if pending_collect_view in collect_sections:
        st.session_state["collect_view"] = pending_collect_view
    collect_view = st.radio(
        "Collect section",
        collect_sections,
        horizontal=True,
        label_visibility="collapsed",
        key="collect_view",
    )
    if collect_view == "Sources":
        render_sources_page(db_path, embedded=True)
    elif collect_view == "New Run":
        render_run_analysis_page(
            db_path=db_path,
            artifacts_dir=artifacts_dir,
            ticker_path=ticker_path,
            embedded=True,
        )
    elif collect_view == "Run Progress":
        st.subheader("Run Progress")
        st.caption("Live controls appear in the run bar above while a scrape is active.")
        runs = load_analysis_runs(db_path)
        if not runs.empty:
            runs = runs[runs["data_mode"] == "live"].copy()
        if runs.empty:
            st.info("No runs have been saved yet.")
        else:
            render_latest_run_status(runs, source_lookup)
            render_recent_run_statuses(runs, source_lookup)
            latest = runs.iloc[0].to_dict()
            if str(latest.get("status", "")).lower() == "completed":
                render_source_health(
                    db_path=db_path,
                    run_id=str(latest["analysis_run_id"]),
                    run_record=latest,
                    source_lookup=source_lookup,
                )
    elif collect_view == "History":
        render_run_library_table(db_path, source_lookup)


def render_review_page(db_path: Path, run_id: str | None) -> None:
    render_page_header(
        "Review",
        "Rank signals, inspect ticker detail, read evidence, and check trend reliability.",
    )
    if not run_id:
        st.info("Select or create an analysis scope before reviewing signals.")
        return

    if is_corpus_scope(run_id):
        long_df, summary_df, _, summary = build_corpus_frames(db_path)
        run_record = None
    else:
        long_df = load_run_mentions(db_path, run_id)
        long_df = attach_quality_columns(long_df, load_run_classification_quality(db_path, run_id))
        summary_df = load_run_ticker_summaries(db_path, run_id)
        summary = load_run_summary(db_path, run_id)
        runs = load_analysis_runs(db_path)
        run_rows = runs[runs["analysis_run_id"].astype(str) == str(run_id)] if not runs.empty else pd.DataFrame()
        run_record = run_rows.iloc[0].to_dict() if not run_rows.empty else None

    review_sections = ["Signals", "Ticker Detail", "Evidence", "Trends"]
    review_view_key = f"review_view_{run_id}"
    pending_review_view = st.session_state.pop(f"pending_review_view_{run_id}", None)
    if pending_review_view in review_sections:
        st.session_state[review_view_key] = pending_review_view
    review_view = st.radio(
        "Review section",
        review_sections,
        horizontal=True,
        label_visibility="collapsed",
        key=review_view_key,
    )
    if review_view == "Signals":
        if is_corpus_scope(run_id):
            render_corpus_summary_cards(summary)
        elif summary:
            render_run_summary_cards(summary)
            render_run_metadata(db_path, run_id)
            if run_record:
                sources = load_sources(db_path)
                source_lookup = {
                    int(row["source_id"]): str(row["display_name"])
                    for row in sources.to_dict(orient="records")
                } if not sources.empty else {}
                render_source_health(
                    db_path=db_path,
                    run_id=run_id,
                    run_record=run_record,
                    source_lookup=source_lookup,
                )

        if summary_df.empty:
            st.info("No ticker summary rows are available for this scope.")
        else:
            render_top_signal_cards(summary_df, run_id=run_id, current_page="Review")
            render_ticker_summary_table(summary_df, run_id=run_id, current_page="Review")

    elif review_view == "Ticker Detail":
        st.caption("Ticker Detail uses the completed corpus so a ticker profile can include evidence across runs.")
        render_ticker_explorer_page(db_path, embedded=True)

    elif review_view == "Evidence":
        if long_df.empty:
            st.info("No evidence rows are available for this scope.")
        else:
            render_results_mentions_review(long_df, summary_df, run_id)

    elif review_view == "Trends":
        render_ticker_trends_page(db_path, run_id, embedded=True)


def render_run_library_table(db_path: Path, source_lookup: dict[int, str]) -> None:
    runs = load_analysis_runs(db_path)
    if not runs.empty:
        runs = runs[runs["data_mode"] == "live"].copy()
    if runs.empty:
        st.info("No runs have been saved yet.")
        return
    run_frame = runs.copy()
    run_frame["source_set"] = run_frame.apply(lambda row: format_source_names(row.to_dict(), source_lookup), axis=1)
    run_frame["items_scraped"] = run_frame["summary_json"].map(lambda value: int((value or {}).get("items_scraped", (value or {}).get("items_collected", 0)) or 0))
    run_frame["items_analyzed"] = run_frame["summary_json"].map(lambda value: int((value or {}).get("items_analyzed", 0) or 0))
    run_frame["ticker_mentions"] = run_frame["summary_json"].map(lambda value: int((value or {}).get("ticker_mentions", 0) or 0))
    run_frame["fallback_classifications"] = run_frame["summary_json"].map(lambda value: int((value or {}).get("fallback_count", 0) or 0))
    run_frame["scrape_coverage"] = run_frame["summary_json"].map(lambda value: str(((value or {}).get("coverage", {}) or {}).get("reliability_label", "")))
    st.dataframe(
        run_frame[
            [
                "analysis_run_id",
                "status",
                "run_type",
                "canonical_trend_run",
                "started_at",
                "completed_at",
                "time_window_label",
                "source_set",
                "items_scraped",
                "items_analyzed",
                "ticker_mentions",
                "fallback_classifications",
                "scrape_coverage",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "analysis_run_id": st.column_config.TextColumn("Run"),
            "run_type": st.column_config.TextColumn("Purpose"),
            "canonical_trend_run": st.column_config.CheckboxColumn("Comparable trend run"),
            "source_set": st.column_config.TextColumn("Sources", width="large"),
            "items_scraped": st.column_config.NumberColumn("Scraped", format="%d"),
            "items_analyzed": st.column_config.NumberColumn("Analysed", format="%d"),
            "ticker_mentions": st.column_config.NumberColumn("Mentions", format="%d"),
            "fallback_classifications": st.column_config.NumberColumn("Fallback classifications", format="%d"),
            "scrape_coverage": st.column_config.TextColumn("Scrape coverage"),
        },
    )


def render_artifact_library(artifacts_dir: Path) -> None:
    st.subheader("Artifacts")
    st.caption(f"Run artifacts are stored at `{artifacts_dir}`.")
    if not artifacts_dir.exists():
        st.info("No artifact directory exists yet.")
        return
    records: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.iterdir(), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        if not path.is_dir():
            continue
        files = [item for item in path.rglob("*") if item.is_file()]
        records.append(
            {
                "artifact": path.name,
                "path": str(path),
                "files": len(files),
                "size_kb": round(sum(item.stat().st_size for item in files) / 1024, 1),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    if not records:
        st.info("No run artifact folders are available yet.")
        return
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def render_watchlist_library(db_path: Path, run_id: str | None) -> None:
    st.subheader("Watchlist")
    watchlist = get_watchlist()
    render_copyable_ticker_chips(watchlist, label="Pinned tickers", max_items=80) if watchlist else st.caption("No pinned tickers yet.")
    render_watchlist_controls(db_path, run_id)


def render_ticker_catalog(ticker_path: Path) -> None:
    st.subheader("Ticker Catalog")
    if not ticker_path.exists():
        st.error(f"Ticker catalog was not found at `{ticker_path}`.")
        return
    try:
        catalog = pd.read_csv(ticker_path)
    except Exception as exc:
        st.error(f"Could not read ticker catalog: {exc}")
        return
    alias_counts = (
        catalog.get("aliases", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .map(lambda value: len([part for part in value.split("|") if part.strip()]))
    )
    alias_tokens: list[str] = []
    if "aliases" in catalog.columns:
        for value in catalog["aliases"].fillna("").astype(str):
            alias_tokens.extend(part.strip().lower() for part in value.split("|") if part.strip())
    duplicate_aliases = sorted({token for token in alias_tokens if alias_tokens.count(token) > 1})
    render_insight_cards(
        [
            {"label": "Tickers", "value": str(len(catalog)), "note": "Catalog-bound extraction universe"},
            {"label": "Aliases", "value": str(int(alias_counts.sum())), "note": "Names and alternate tokens"},
            {"label": "Ambiguous tokens", "value": str(len(duplicate_aliases)), "note": "Aliases used by multiple rows"},
        ]
    )
    query = st.text_input("Search catalog", placeholder="Ticker, company, alias", key="ticker_catalog_search").strip().lower()
    display = catalog.copy()
    if query:
        searchable = display.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        display = display[searchable.str.contains(query, regex=False)].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)
    with st.expander("Ambiguous alias tokens", expanded=False):
        if duplicate_aliases:
            st.write(", ".join(duplicate_aliases[:120]))
        else:
            st.caption("No duplicate alias tokens were found.")


def render_library_page(db_path: Path, ticker_path: Path, artifacts_dir: Path, run_id: str | None, source_lookup: dict[int, str]) -> None:
    render_page_header(
        "Library",
        "Runs, artifacts, exports, watchlist, and the ticker catalog.",
    )
    library_sections = ["Runs", "Artifacts", "Exports", "Watchlist", "Ticker Catalog"]
    pending_library_view = st.session_state.pop("pending_library_view", None)
    if pending_library_view in library_sections:
        st.session_state["library_view"] = pending_library_view
    library_view = st.radio(
        "Library section",
        library_sections,
        horizontal=True,
        label_visibility="collapsed",
        key="library_view",
    )
    if library_view == "Runs":
        render_run_library_table(db_path, source_lookup)
    elif library_view == "Artifacts":
        render_artifact_library(artifacts_dir)
    elif library_view == "Exports":
        render_export_page(db_path, run_id, embedded=True)
    elif library_view == "Watchlist":
        render_watchlist_library(db_path, run_id)
    elif library_view == "Ticker Catalog":
        render_ticker_catalog(ticker_path)


def render_signal_validation_page(db_path: Path, ticker_path: Path, run_id: str | None) -> None:
    render_page_header(
        "Validate",
        "Frozen signal events preserve what the app knew at the time a ticker was flagged, then track market outcomes, Reddit attention spread, and human review labels without hindsight edits.",
    )

    action_cols = st.columns([1, 1.2, 2.2])
    refresh_result: dict[str, Any] | None = None
    manual_result: dict[str, Any] | None = None
    if action_cols[0].button("Refresh outcomes", use_container_width=True):
        with st.spinner("Refreshing market caches and recomputing signal outcomes..."):
            refresh_result = refresh_signal_validation(
                db_path=db_path,
                ticker_path=ticker_path,
            )

    runs = load_analysis_runs(db_path)
    current_run_record = None
    summary_df = pd.DataFrame()
    mentions_df = pd.DataFrame()
    if run_id and not is_corpus_scope(run_id):
        run_rows = runs[runs["analysis_run_id"] == run_id]
        if not run_rows.empty:
            current_run_record = run_rows.iloc[0].to_dict()
            summary_df = load_run_ticker_summaries(db_path, run_id)
            mentions_df = load_run_mentions(db_path, run_id)
    if current_run_record and not summary_df.empty:
        with action_cols[1].popover("Freeze manual signal", use_container_width=True):
            st.caption("Save a frozen signal snapshot from the currently selected run, even if it was not auto-flagged.")
            manual_ticker = st.selectbox(
                "Ticker",
                summary_df["ticker"].tolist(),
                key=f"manual_signal_ticker_{run_id}",
            )
            if st.button("Save frozen signal", type="primary", use_container_width=True):
                with st.spinner(f"Freezing manual signal for {manual_ticker}..."):
                    manual_result = run_signal_validation(
                        db_path=db_path,
                        ticker_path=ticker_path,
                        run_id=str(current_run_record["analysis_run_id"]),
                        run_config=current_run_record,
                        summary_df=summary_df,
                        mentions_df=mentions_df,
                        signal_time=analysis_run_now_iso(),
                        manual_tickers=[manual_ticker],
                        allow_automatic_signals=False,
                    )

    if refresh_result:
        if refresh_result.get("status") == "ok":
            st.success(
                f"Refreshed {int(refresh_result.get('signals_refreshed', 0))} signal(s). "
                f"Updated {int(refresh_result.get('market_outcomes_updated', 0))} market outcome row(s)."
            )
        else:
            st.warning(
                f"Refresh completed with gaps. Signals refreshed: {int(refresh_result.get('signals_refreshed', 0))}. "
                f"Issues: {', '.join(refresh_result.get('market_data_errors', [])[:3]) or 'see cached data status'}."
            )
    if manual_result:
        created_count = int(manual_result.get("signals_created", 0) or 0)
        if created_count > 0:
            st.success(f"Saved {created_count} frozen manual signal event.")
        else:
            st.info("No new signal event was created. A recent snapshot for that run/ticker likely already exists.")

    review_df = load_signal_review_frame(db_path, run_id if run_id else CORPUS_SCOPE_ID)
    if review_df.empty:
        st.info("No frozen signal events have been created yet. Complete a run, then save a manual signal or let the auto-signal workflow freeze top leads.")
        return

    if is_corpus_scope(run_id):
        render_scope_context_bar(
            title="All frozen signal events",
            meta="Point-in-time ticker snapshots across every completed live run. Manual labels stay attached to the frozen event, not the moving corpus.",
            chips=["Signals", "Validation"],
            metrics=[
                {"label": "Frozen signals", "value": str(len(review_df))},
                {"label": "Unique tickers", "value": str(int(review_df["ticker"].nunique()))},
                {"label": "Manual signals", "value": str(int(review_df["manual_signal"].fillna(False).astype(bool).sum()))},
                {"label": "Labelled", "value": str(int(review_df["user_label"].fillna("").astype(str).str.len().gt(0).sum()))},
            ],
        )

    render_metric_guide("Metric guide for Validate", expanded=False, context="signals")

    metric_cols = st.columns(5)
    labelled_count = int(review_df["user_label"].fillna("").astype(str).str.len().gt(0).sum()) if "user_label" in review_df.columns else 0
    useful_count = int(
        review_df["user_label"].fillna("").isin(["Useful lead", "Worth deeper research", "Avoided hype trap"]).sum()
    ) if "user_label" in review_df.columns else 0
    return_7d_series = review_df["return_7d"].dropna() if "return_7d" in review_df.columns else pd.Series(dtype=float)
    excess_7d_series = review_df["excess_return_7d"].dropna() if "excess_return_7d" in review_df.columns else pd.Series(dtype=float)
    avg_return_7d = return_7d_series.mean() if not return_7d_series.empty else None
    avg_excess_7d = excess_7d_series.mean() if not excess_7d_series.empty else None
    avg_attention_14d = review_df["attention_spread_14d"].dropna().mean() if "attention_spread_14d" in review_df.columns else None
    metric_cols[0].metric("Frozen signals", len(review_df))
    metric_cols[1].metric("Labelled", labelled_count)
    metric_cols[2].metric("Useful / deeper", useful_count)
    metric_cols[3].metric("Avg ticker 7d", f"{avg_return_7d:+.2%}" if pd.notna(avg_return_7d) else "n/a")
    metric_cols[4].metric("Avg excess 7d", f"{avg_excess_7d:+.2%}" if pd.notna(avg_excess_7d) else "n/a")
    st.caption(
        f"7d market outcomes available for {len(return_7d_series)}/{len(review_df)} signal(s); "
        f"SPY-relative excess returns available for {len(excess_7d_series)}/{len(review_df)}."
    )
    if pd.notna(avg_attention_14d):
        st.caption(f"Average 14d attention spread: {float(avg_attention_14d):.1f}")
    render_status_chips([market_cache_status_label(review_df)])

    filter_cols = st.columns([1.2, 1, 1, 1])
    label_options = ["All labels", *SIGNAL_LABEL_OPTIONS]
    selected_label_filter = filter_cols[0].selectbox("Label filter", label_options, key=f"signal_label_filter_{run_id}")
    ticker_query = filter_cols[1].text_input("Ticker filter", key=f"signal_ticker_filter_{run_id}").strip().upper()
    signal_type_filter = filter_cols[2].selectbox("Signal type", ["All", "Auto", "Manual"], key=f"signal_type_filter_{run_id}")
    reliability_filter = filter_cols[3].selectbox("Attention reliability", ["All", "High", "Medium", "Low"], key=f"signal_reliability_filter_{run_id}")
    sort_choice = st.selectbox(
        "Sort inbox by",
        ["Newest signal, unlabeled first", "Best emerging", "Highest attention spread", "Best excess return", "Lowest hype"],
        key=f"signal_sort_{run_id}",
    )

    filtered_df = review_df.copy()
    filtered_df["user_label"] = filtered_df["user_label"].fillna("")
    if selected_label_filter != "All labels":
        filtered_df = filtered_df[filtered_df["user_label"] == selected_label_filter].copy()
    if ticker_query:
        filtered_df = filtered_df[filtered_df["ticker"].astype(str).str.contains(ticker_query, regex=False)].copy()
    if signal_type_filter == "Auto":
        filtered_df = filtered_df[~filtered_df["manual_signal"].fillna(False).astype(bool)].copy()
    elif signal_type_filter == "Manual":
        filtered_df = filtered_df[filtered_df["manual_signal"].fillna(False).astype(bool)].copy()
    if reliability_filter != "All" and "attention_outcome_reliability" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["attention_outcome_reliability"].fillna("").astype(str) == reliability_filter
        ].copy()
    filtered_df["unlabelled_first"] = filtered_df["user_label"].fillna("").astype(str).eq("")
    if sort_choice == "Best emerging":
        filtered_df = filtered_df.sort_values(["unlabelled_first", "emerging_ticker_score", "signal_time_dt"], ascending=[False, False, False], na_position="last")
    elif sort_choice == "Highest attention spread":
        filtered_df = filtered_df.sort_values(["unlabelled_first", "attention_spread_14d", "signal_time_dt"], ascending=[False, False, False], na_position="last")
    elif sort_choice == "Best excess return":
        filtered_df = filtered_df.sort_values(["unlabelled_first", "excess_return_7d", "signal_time_dt"], ascending=[False, False, False], na_position="last")
    elif sort_choice == "Lowest hype":
        filtered_df = filtered_df.sort_values(["unlabelled_first", "hype_score", "signal_time_dt"], ascending=[False, True, False], na_position="last")
    else:
        filtered_df = filtered_df.sort_values(
            ["unlabelled_first", "signal_time_dt"],
            ascending=[False, False],
            na_position="last",
        )

    if filtered_df.empty:
        st.info("No frozen signals matched those filters.")
        return

    frozen_tab, outcomes_tab, benchmarks_tab, labels_tab = st.tabs(
        ["Frozen Signals", "Outcomes", "Benchmarks", "Labels"]
    )

    with frozen_tab:
        st.dataframe(
            filtered_df[
                [
                    "signal_time",
                    "ticker",
                    "company_name",
                    "signal_rank",
                    "manual_signal",
                    "emerging_ticker_score",
                    "adjusted_acceleration_score",
                    "alpha_signal_score",
                    "hype_score",
                    "trend_reliability",
                    "coverage_reliability",
                    "return_7d",
                    "excess_return_7d",
                    "attention_spread_14d",
                    "user_label",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "signal_time": st.column_config.TextColumn("Signal time", width="medium"),
                "manual_signal": st.column_config.CheckboxColumn("Manual"),
                "emerging_ticker_score": st.column_config.NumberColumn("Emerging", format="%.1f", help=metric_help("emerging_ticker_score")),
                "adjusted_acceleration_score": st.column_config.NumberColumn("Adj. accel.", format="%.1f", help=metric_help("adjusted_acceleration_score")),
                "alpha_signal_score": st.column_config.NumberColumn("Alpha", format="%.1f"),
                "hype_score": st.column_config.NumberColumn("Hype", format="%.1f"),
                "coverage_reliability": st.column_config.NumberColumn("Coverage", format="%.2f", help=metric_help("coverage_reliability")),
                "return_7d": st.column_config.NumberColumn("Return 7d", format="%.2f"),
                "excess_return_7d": st.column_config.NumberColumn("Excess 7d", format="%.2f"),
                "attention_spread_14d": st.column_config.NumberColumn("Attention 14d", format="%.1f", help=metric_help("attention_spread_14d")),
            },
        )

        signal_options = filtered_df["signal_id"].astype(str).tolist()
        selected_signal_id = st.selectbox(
            "Signal event",
            signal_options,
            format_func=lambda value: _signal_select_label(filtered_df, value),
            key=f"selected_signal_event_{run_id}",
        )
        selected_signal = filtered_df[filtered_df["signal_id"].astype(str) == str(selected_signal_id)].iloc[0].to_dict()

        st.subheader(f"{selected_signal['ticker']} signal snapshot")
        render_status_chips(
            [
                "Manual" if bool(selected_signal.get("manual_signal")) else "Auto",
                str(selected_signal.get("trend_reliability", "") or ""),
                str(selected_signal.get("attention_outcome_reliability", "") or ""),
            ]
        )
        detail_cols = st.columns(5)
        detail_cols[0].metric("Emerging", f"{_coerce_float(selected_signal.get('emerging_ticker_score')):.1f}", help=metric_help("emerging_ticker_score"))
        detail_cols[1].metric("Adj. accel.", f"{_coerce_float(selected_signal.get('adjusted_acceleration_score')):.1f}", help=metric_help("adjusted_acceleration_score"))
        detail_cols[2].metric("Evidence score", f"{_coerce_float(selected_signal.get('alpha_signal_score')):.1f}")
        detail_cols[3].metric("Hype", f"{_coerce_float(selected_signal.get('hype_score')):.1f}")
        detail_cols[4].metric("Trend trust", str(selected_signal.get("trend_reliability", "")) or "n/a")
        if run_id:
            render_ticker_navigation_buttons(current_page="Validate", run_id=str(run_id), ticker=str(selected_signal["ticker"]))

        with st.expander("Frozen evidence snapshot", expanded=True):
            evidence_titles = selected_signal.get("top_evidence_titles", []) or []
            evidence_excerpts = selected_signal.get("top_evidence_excerpts", []) or []
            evidence_ids = selected_signal.get("top_evidence_item_ids", []) or []
            if not evidence_titles:
                st.caption("No evidence snapshot was stored for this signal.")
            else:
                for idx, title in enumerate(evidence_titles, start=1):
                    excerpt = evidence_excerpts[idx - 1] if idx - 1 < len(evidence_excerpts) else ""
                    item_id = evidence_ids[idx - 1] if idx - 1 < len(evidence_ids) else ""
                    st.markdown(f"**{idx}. {title}**")
                    if excerpt:
                        st.caption(excerpt)
                    if item_id:
                        st.caption(f"Item ID: `{item_id}`")

    with outcomes_tab:
        st.subheader("Market Outcomes")
        market_cols = st.columns(4)
        market_cols[0].metric("Return 7d", f"{_format_percent(selected_signal.get('return_7d'))}")
        market_cols[1].metric("Excess 7d", f"{_format_percent(selected_signal.get('excess_return_7d'))}", help=metric_help("excess_return"))
        market_cols[2].metric("Return 30d", f"{_format_percent(selected_signal.get('return_30d'))}")
        market_cols[3].metric("Excess 30d", f"{_format_percent(selected_signal.get('excess_return_30d'))}")
        st.subheader("Reddit Attention Outcomes")
        attention_cols = st.columns(5)
        attention_cols[0].metric("Attention 7d", f"{_coerce_float(selected_signal.get('attention_spread_7d')):.1f}", help=metric_help("attention_spread_7d"))
        attention_cols[1].metric("Attention 14d", f"{_coerce_float(selected_signal.get('attention_spread_14d')):.1f}")
        attention_cols[2].metric("Attention 30d", f"{_coerce_float(selected_signal.get('attention_spread_30d')):.1f}", help=metric_help("attention_spread_30d"))
        attention_cols[3].metric("Share of voice", f"{_format_percent(_coerce_float(selected_signal.get('share_of_voice')) / 100)}", help=metric_help("share_of_voice"))
        attention_cols[4].metric("Scrape coverage", f"{_coerce_float(selected_signal.get('coverage_reliability')):.2f}", help=metric_help("coverage_reliability"))
        if selected_signal.get("outcome_last_updated"):
            st.caption(
                f"Market and attention outcomes last refreshed `{format_timestamp_with_relative(selected_signal['outcome_last_updated'])}`"
            )

    with benchmarks_tab:
        random_df = load_signal_random_benchmarks(db_path, signal_id=str(selected_signal_id))
        if random_df.empty:
            st.info("No random benchmark sample is available for this signal yet.")
        else:
            sample_mean_7d = random_df["return_7d"].dropna().mean()
            sample_mean_30d = random_df["return_30d"].dropna().mean()
            sample_cols = st.columns(3)
            sample_cols[0].metric("Random sample", int(random_df["benchmark_ticker"].nunique()))
            sample_cols[1].metric("Random avg 7d", _format_percent(sample_mean_7d))
            sample_cols[2].metric("Random avg 30d", _format_percent(sample_mean_30d))
            with st.expander("Random benchmark sample", expanded=False):
                st.dataframe(
                    random_df[
                        [
                            "benchmark_ticker",
                            "sample_index",
                            "return_7d",
                            "return_30d",
                            "max_gain_7d",
                            "max_drawdown_7d",
                            "volume_spike_7d",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with labels_tab:
        current_label = str(selected_signal.get("user_label", "") or "")
        st.caption("Quick label")
        quick_cols = st.columns(len(QUICK_SIGNAL_LABELS))
        for column, label in zip(quick_cols, QUICK_SIGNAL_LABELS):
            if column.button(
                label,
                key=f"quick_label_{selected_signal_id}_{label}",
                use_container_width=True,
                disabled=current_label == label,
            ):
                upsert_signal_label(
                    db_path,
                    signal_id=str(selected_signal_id),
                    user_label=label,
                    user_notes=str(selected_signal.get("user_notes", "") or ""),
                )
                st.success("Signal label saved.")
                st.rerun()

        label_options_with_blank = ["", *SIGNAL_LABEL_OPTIONS]
        current_label_index = label_options_with_blank.index(current_label) if current_label in label_options_with_blank else 0
        with st.form(f"signal_label_form_{selected_signal_id}"):
            st.caption("Manual labels are stored against the frozen signal event, not the moving live corpus.")
            selected_label = st.selectbox(
                "Manual label",
                label_options_with_blank,
                index=current_label_index,
                format_func=lambda value: value if value else "Unlabeled",
            )
            user_notes = st.text_area(
                "Review notes",
                value=str(selected_signal.get("user_notes", "") or ""),
                height=140,
            )
            submitted = st.form_submit_button("Save label", type="primary")
            if submitted:
                upsert_signal_label(
                    db_path,
                    signal_id=str(selected_signal_id),
                    user_label=selected_label,
                    user_notes=user_notes,
                )
                st.success("Signal label saved.")
                st.rerun()


def render_export_page(db_path: Path, run_id: str | None, *, embedded: bool = False) -> None:
    if embedded:
        st.subheader("Exports")
    else:
        render_page_header("Exports", "Download current scope data without changing analysis state.")
    if not run_id:
        st.info("Run an analysis first.")
        return

    if is_corpus_scope(run_id):
        long_df, summary_df, bucket_df, summary_json = build_corpus_frames(db_path)
        export_id = "combined_corpus"
        st.caption("Exporting the deduped combined corpus across all completed live runs.")
    else:
        long_df = load_run_mentions(db_path, run_id)
        summary_df = load_run_ticker_summaries(db_path, run_id)
        bucket_df = load_run_time_buckets(db_path, run_id)
        summary_json = load_run_summary(db_path, run_id)
        export_id = run_id

    if long_df.empty:
        st.info("No data available for export in this scope. Run a scrape with ticker mentions, then return to Export.")
        return

    if not is_corpus_scope(run_id):
        run_artifact_dir = DEFAULT_ARTIFACTS_DIR / run_id
        if run_artifact_dir.exists():
            st.caption(f"Automatic run artifacts saved to `{run_artifact_dir}`.")

    st.subheader("Evidence exports")
    st.write("Long format: one row per analysed item and ticker mention. Use this when you want the raw evidence rows.")
    st.download_button(
        "Download raw item-level long format CSV",
        long_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_id}_long_format.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download ticker-level summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_id}_ticker_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download time-bucketed ticker trend CSV",
        bucket_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{export_id}_ticker_trends.csv",
        mime="text/csv",
    )

    st.subheader("Spreadsheet-friendly export")
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
        file_name=f"{export_id}_wide_format.csv",
        mime="text/csv",
    )

    st.subheader("Full JSON export")
    full_payload = {
        "analysis_scope": export_id,
        "run_summary": summary_json,
        "long_format": long_df.to_dict(orient="records") if is_corpus_scope(run_id) else load_full_results_records(db_path, run_id),
        "ticker_summaries": summary_df.to_dict(orient="records"),
        "ticker_time_buckets": bucket_df.to_dict(orient="records"),
    }
    st.download_button(
        "Download JSON export of full classified results",
        json.dumps(json_safe_value(full_payload), indent=2).encode("utf-8"),
        file_name=f"{export_id}_full_results.json",
        mime="application/json",
    )

    signal_review_df = load_signal_review_frame(db_path, run_id if run_id else CORPUS_SCOPE_ID)
    if not signal_review_df.empty:
        st.subheader("Signal validation exports")
        st.caption("These files export frozen signal snapshots, linked outcomes, and manual labels.")
        st.download_button(
            "Download frozen signal events CSV",
            signal_review_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{export_id}_signal_events.csv",
            mime="text/csv",
        )
        signal_ids = set(signal_review_df["signal_id"].astype(str).tolist())
        random_benchmarks_df = load_signal_random_benchmarks(db_path)
        if not random_benchmarks_df.empty:
            random_benchmarks_df = random_benchmarks_df[
                random_benchmarks_df["signal_id"].astype(str).isin(signal_ids)
            ].copy()
        signal_payload = {
            "analysis_scope": export_id,
            "signal_events": signal_review_df.to_dict(orient="records"),
            "random_benchmarks": random_benchmarks_df.to_dict(orient="records") if not random_benchmarks_df.empty else [],
        }
        st.download_button(
            "Download signal validation JSON",
            json.dumps(json_safe_value(signal_payload), indent=2).encode("utf-8"),
            file_name=f"{export_id}_signal_validation.json",
            mime="application/json",
        )


def render_settings_page(db_path: Path, ticker_path: Path, artifacts_dir: Path) -> None:
    render_page_header(
        "Settings",
        "Technical configuration, diagnostics, maintenance, and metric definitions.",
    )
    overview = db_overview(db_path)
    overview_cols = st.columns(6)
    overview_cols[0].metric("Sources", overview["sources"])
    overview_cols[1].metric("Analysis runs", overview["analysis_runs"])
    overview_cols[2].metric("Reddit items", overview["reddit_items"])
    overview_cols[3].metric("Mention rows", overview["item_ticker_mentions"])
    overview_cols[4].metric("Signal events", overview.get("ticker_signal_events", 0))
    overview_cols[5].metric("Market rows", overview.get("ticker_market_prices", 0))

    storage_tab, model_tab, reddit_tab, diagnostics_tab, maintenance_tab, guide_tab = st.tabs(
        ["Storage", "Model", "Reddit Collector", "Diagnostics", "Maintenance", "Metric Guide"]
    )

    with storage_tab:
        st.subheader("Storage")
        storage_parts = DEFAULT_STORAGE_ROOT.parts
        using_external_storage = len(storage_parts) >= 3 and storage_parts[1] == "Volumes"
        if using_external_storage:
            st.success(f"App data writes directly to the external SSD at `{DEFAULT_STORAGE_ROOT}`.")
        else:
            st.warning(f"App data is currently writing to `{DEFAULT_STORAGE_ROOT}` on the local machine.")
        st.code(
            "\n".join(
                [
                    f"db_path = {db_path}",
                    f"artifacts_dir = {artifacts_dir}",
                    f"ui_preferences = {DEFAULT_UI_PREFS_PATH}",
                    f"storage_root = {DEFAULT_STORAGE_ROOT}",
                    f"ticker_catalog_path = {ticker_path}",
                ]
            )
        )

    with model_tab:
        st.subheader("Model")
        ollama_models_dir = Path(os.getenv("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models"))).expanduser()
        st.code(
            "\n".join(
                [
                    f"ollama_url = {DEFAULT_OLLAMA_URL}",
                    f"default_model = {DEFAULT_MODEL}",
                    f"ollama_models = {ollama_models_dir}",
                ]
            )
        )
        if not str(ollama_models_dir).startswith(str(DEFAULT_STORAGE_ROOT)):
            st.warning(f"Ollama model files are separate from app data and currently live at `{ollama_models_dir}`.")

    with reddit_tab:
        st.subheader("Reddit Collector")
        st.code(
            "\n".join(
                [
                    f"reddit_user_agent = {os.getenv('REDDIT_USER_AGENT', 'script:signal-from-the-slop:0.1')}",
                    f"subreddit_rss_limit = {REDDIT_SUBREDDIT_RSS_LIMIT}",
                ]
            )
        )
        st.caption("Normal collection actions live in Collect. This tab only shows collector configuration.")

    with diagnostics_tab:
        st.subheader("Package Versions")
        env_cols = st.columns(4)
        env_cols[0].metric("Streamlit", st.__version__)
        env_cols[1].metric("Altair", alt.__version__)
        env_cols[2].metric("Pandas", pd.__version__)
        env_cols[3].metric("Requests", requests.__version__)

        st.subheader("Diagnostic Paths")
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

    with maintenance_tab:
        st.subheader("Run health")
        runs = load_analysis_runs(db_path)
        if not runs.empty:
            runs = runs[runs["data_mode"] == "live"].copy()
        if runs.empty:
            st.info("No runs have been saved yet.")
        else:
            run_health = runs.copy()
            run_health["items_scraped"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_scraped", (value or {}).get("items_collected", (value or {}).get("items_analyzed", 0))) or 0))
            run_health["items_skipped_existing"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_skipped_existing", 0) or 0))
            run_health["items_collected"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_collected", (value or {}).get("items_analyzed", 0)) or 0))
            run_health["items_analyzed"] = run_health["summary_json"].map(lambda value: int((value or {}).get("items_analyzed", 0)))
            run_health["ticker_mentions"] = run_health["summary_json"].map(lambda value: int((value or {}).get("ticker_mentions", 0)))
            run_health["tickers_found"] = run_health["summary_json"].map(lambda value: int((value or {}).get("tickers_found", 0)))
            run_health["fallback_count"] = run_health["summary_json"].map(lambda value: int((value or {}).get("fallback_count", 0)))
            run_health["error_message"] = run_health["summary_json"].map(lambda value: str((value or {}).get("error_message", "")))
            run_health["coverage_label"] = run_health["summary_json"].map(lambda value: str(((value or {}).get("coverage", {}) or {}).get("reliability_label", "")))
            run_health["posts_returned"] = run_health["summary_json"].map(lambda value: int(((value or {}).get("coverage", {}) or {}).get("posts_returned", 0) or 0))
            run_health["elapsed_minutes"] = run_health.apply(lambda row: run_elapsed_minutes(row.to_dict()) or 0, axis=1)
            run_health["heartbeat_age_seconds"] = run_health.apply(lambda row: run_heartbeat_age_seconds(row.to_dict()), axis=1)
            run_health["elapsed"] = run_health.apply(
                lambda row: run_elapsed_label(row.to_dict()) if str(row.get("status", "")).lower() in ACTIVE_RUN_STATUSES else "",
                axis=1,
            )
            status_counts = run_health["status"].value_counts().to_dict()
            active_count = int(sum(int(status_counts.get(status, 0)) for status in ACTIVE_RUN_STATUSES))
            health_cols = st.columns(4)
            health_cols[0].metric("Completed", int(status_counts.get("completed", 0)))
            health_cols[1].metric("Active", active_count)
            health_cols[2].metric("Failed", int(status_counts.get("failed", 0)))
            health_cols[3].metric("Cancelled", int(status_counts.get("cancelled", 0)))
            orphaned_active = run_health[
                run_health.apply(lambda row: is_orphaned_active_run(row.to_dict()), axis=1)
            ].copy()
            if not orphaned_active.empty:
                st.warning(f"{len(orphaned_active)} active run(s) have no live worker heartbeat.")
                if st.button("Mark orphaned active runs as cancelled", use_container_width=True):
                    for row in orphaned_active.to_dict(orient="records"):
                        force_cancel_analysis_run(
                            db_path,
                            str(row["analysis_run_id"]),
                            "Cancelled from Run health because no live worker heartbeat was available.",
                        )
                    st.rerun()
            st.dataframe(
                run_health[
                    [
                        "analysis_run_id",
                        "status",
                        "run_type",
                        "canonical_trend_run",
                        "started_at",
                        "completed_at",
                        "elapsed",
                        "time_window_label",
                        "items_scraped",
                        "posts_returned",
                        "coverage_label",
                        "items_skipped_existing",
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
                    "items_scraped": st.column_config.NumberColumn("Scraped", format="%d"),
                    "posts_returned": st.column_config.NumberColumn("Posts", format="%d"),
                    "coverage_label": st.column_config.TextColumn("Coverage"),
                    "items_skipped_existing": st.column_config.NumberColumn("Skipped existing", format="%d"),
                    "items_collected": st.column_config.NumberColumn("Collected", format="%d"),
                    "items_analyzed": st.column_config.NumberColumn("Analysed", format="%d"),
                    "ticker_mentions": st.column_config.NumberColumn("Mentions", format="%d"),
                    "tickers_found": st.column_config.NumberColumn("Tickers", format="%d"),
                    "fallback_count": st.column_config.NumberColumn("Fallback", format="%d"),
                    "error_message": st.column_config.TextColumn("Error", width="large"),
                },
            )

    with guide_tab:
        st.subheader("Scoring and table glossary")
        render_metric_guide("Full metric reference", expanded=True, context="all")


init_db(DEFAULT_DB_PATH, SCHEMA_PATH)
ensure_seed_sources(DEFAULT_DB_PATH, DEFAULT_SOURCES_PATH)
page, selected_run_id, sources_df, runs_df = build_sidebar_state(DEFAULT_DB_PATH)
source_lookup = {
    int(row["source_id"]): str(row["display_name"])
    for row in sources_df.to_dict(orient="records")
} if not sources_df.empty else {}
render_app_bar(
    runs=runs_df,
    run_id=selected_run_id,
    source_lookup=source_lookup,
    current_page=page,
)
page = render_workflow_navigation()
render_background_run_banner(DEFAULT_DB_PATH, source_lookup)
if page in {"Review", "Validate"}:
    render_persistent_run_context(DEFAULT_DB_PATH, selected_run_id)
    render_body_quick_jump(DEFAULT_DB_PATH, selected_run_id, page)

if page == "Today":
    render_today_page(DEFAULT_DB_PATH, selected_run_id, source_lookup)
elif page == "Collect":
    render_collect_page(
        db_path=DEFAULT_DB_PATH,
        artifacts_dir=DEFAULT_ARTIFACTS_DIR,
        ticker_path=DEFAULT_TICKER_PATH,
        source_lookup=source_lookup,
    )
elif page == "Review":
    render_review_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Validate":
    render_signal_validation_page(DEFAULT_DB_PATH, DEFAULT_TICKER_PATH, selected_run_id)
elif page == "Library":
    render_library_page(DEFAULT_DB_PATH, DEFAULT_TICKER_PATH, DEFAULT_ARTIFACTS_DIR, selected_run_id, source_lookup)
elif page == "Settings":
    render_settings_page(DEFAULT_DB_PATH, DEFAULT_TICKER_PATH, DEFAULT_ARTIFACTS_DIR)
