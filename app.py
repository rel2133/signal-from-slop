from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
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
    load_full_results_records,
    load_run_mentions,
    load_run_summary,
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
DEFAULT_DATA_PATH = ROOT / "data" / "fake_reddit_data.json"
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
PAGES = [
    "Sources",
    "Run Analysis",
    "Results Dashboard",
    "Ticker Trends",
    "Export Data",
    "Settings",
]

LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
DATA_MODE_LABELS = {
    "live": "Live Reddit scrape",
    "fake": "Bundled fake dataset",
}


def resolve_config_path(env_name: str, default_name: str, *, base_dir: Path) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return base_dir / default_name
    configured = Path(raw_value).expanduser()
    return configured if configured.is_absolute() else base_dir / configured


DEFAULT_DB_PATH = resolve_config_path("SQLITE_PATH", "signal_from_the_slop.db", base_dir=DEFAULT_STORAGE_ROOT)
DEFAULT_ARTIFACTS_DIR = resolve_config_path("ARTIFACTS_DIR", "artifacts", base_dir=DEFAULT_STORAGE_ROOT)


st.set_page_config(page_title="Signal from the Slop", layout="wide")
st.title("Signal from the Slop")
st.caption("Research dashboard for Reddit stock discussion triage. Not financial advice.")


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


def queue_completed_run_navigation(run_id: str) -> None:
    st.session_state["pending_selected_run_id"] = run_id
    st.session_state["completed_run_notice"] = f"Run `{run_id}` completed."
    st.session_state["page"] = "Results Dashboard"


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


def seed_fake_subreddit_sources(db_path: Path, data_path: Path) -> int:
    items = RedditClient(data_path).load_fake_data()
    subreddits = sorted({item["subreddit"] for item in items if item.get("subreddit")})
    for subreddit in subreddits:
        source = build_subreddit_source(subreddit)
        add_source(
            db_path,
            source_key=source.source_key,
            source_type=source.source_type,
            display_name=source.display_name,
            normalized_value=source.normalized_value,
            url=source.url,
        )
    return len(subreddits)


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


def ensure_seed_sources(db_path: Path, data_path: Path, sources_path: Path) -> None:
    if sources_path.exists():
        seed_sources_from_file(db_path, sources_path)
        return

    sources = load_sources(db_path)
    if sources.empty:
        seed_fake_subreddit_sources(db_path, data_path)


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
        previous_end = parse_created_time(str(previous_run["time_window_end"]))
        if previous_end >= anchor:
            start_dt = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = (previous_end + timedelta(seconds=1)).replace(microsecond=0)
        end_dt = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
        return start_dt, end_dt
    if label == "Since last completed run":
        label = "Last 7 days"

    days = TIME_WINDOW_OPTIONS[label]
    end_dt = anchor.replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = (end_dt - timedelta(days=(days or 1) - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start_dt, end_dt


def run_analysis_pipeline(
    *,
    data_path: Path,
    ticker_path: Path,
    db_path: Path,
    artifacts_dir: Path,
    data_mode: str,
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
    reddit_client = RedditClient(data_path)
    extractor = TickerExtractor(ticker_path)
    classifier = OllamaClassifier(model=model_name, endpoint=ollama_url)

    if data_mode == "live":
        items = reddit_client.collect_live_items(
            selected_sources=selected_sources,
            window_start=time_window_start,
            window_end=time_window_end,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
        )
    else:
        items = reddit_client.collect_fake_items(
            selected_sources=selected_sources,
            window_start=time_window_start,
            window_end=time_window_end,
            max_posts_per_source=max_posts_per_source,
            max_comments_per_thread=max_comments_per_thread,
            include_comments=include_comments,
        )

    run_config = {
        "data_mode": data_mode,
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

    run_id = create_analysis_run(
        db_path,
        run_config,
    )

    item_rows: list[dict[str, Any]] = []
    item_lookup: dict[str, dict[str, Any]] = {}
    classification_rows: dict[str, dict[str, Any]] = {}
    temp_mentions: list[dict[str, Any]] = []
    fallback_count = 0

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
    summary["fallback_count"] = fallback_count

    replaceable_mentions = long_df.to_dict(orient="records") if not long_df.empty else []
    replaceable_summaries = ticker_summary_df.to_dict(orient="records") if not ticker_summary_df.empty else []
    replaceable_buckets = time_bucket_df.to_dict(orient="records") if not time_bucket_df.empty else []
    upsert_reddit_items(db_path, item_rows)
    replace_run_classifications(db_path, run_id, list(classification_rows.values()))
    replace_run_mentions(db_path, run_id, replaceable_mentions)
    replace_run_ticker_summaries(db_path, run_id, replaceable_summaries)
    replace_run_time_buckets(db_path, run_id, replaceable_buckets)
    complete_analysis_run(db_path, run_id, summary)
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

    progress_bar.empty()
    status_box.success("Analysis complete.")
    return run_id, summary


def build_sidebar_state(db_path: Path, data_path: Path) -> tuple[str, str | None, pd.DataFrame, pd.DataFrame]:
    sources = load_sources(db_path)
    runs = load_analysis_runs(db_path)
    pending_run_id = st.session_state.pop("pending_selected_run_id", None)
    if pending_run_id and not runs.empty and pending_run_id in runs["analysis_run_id"].tolist():
        st.session_state["selected_run_id"] = pending_run_id

    with st.sidebar:
        page = st.radio("Navigate", PAGES, key="page")
        run_id = None
        if not runs.empty:
            run_options = runs["analysis_run_id"].tolist()
            default_run_id = st.session_state.get("selected_run_id") or latest_analysis_run_id(db_path)
            run_index = run_options.index(default_run_id) if default_run_id in run_options else 0
            run_id = st.selectbox("Analysis run", run_options, index=run_index, key="selected_run_id")
        else:
            st.info("No analysis runs saved yet.")

        st.markdown("---")
        st.caption(f"Fake data: `{data_path}`")
        st.caption(f"SQLite: `{db_path}`")

    return page, run_id, sources, runs


def render_sources_page(db_path: Path, data_path: Path) -> None:
    st.header("Sources")
    st.write("Add subreddit sources or Reddit thread URLs. Inputs are normalized before saving.")

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

    seed_col, _ = st.columns([1, 3])
    with seed_col:
        if st.button("Seed fake subreddit sources"):
            added = seed_fake_subreddit_sources(db_path, data_path)
            st.success(f"Seeded {added} subreddit sources from the bundled fake dataset.")
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
    data_path: Path,
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

    client = RedditClient(data_path)
    fake_subreddits = sorted(
        {
            build_subreddit_source(str(item["subreddit"])).display_name
            for item in client.load_fake_data()
            if item.get("subreddit")
        }
    )

    default_data_mode_index = 0
    data_mode_label = st.radio(
        "Collection mode",
        [DATA_MODE_LABELS["live"], DATA_MODE_LABELS["fake"]],
        index=default_data_mode_index,
        horizontal=True,
    )
    data_mode = "live" if data_mode_label == DATA_MODE_LABELS["live"] else "fake"

    if data_mode == "live":
        st.info(
            "Live Reddit scrape uses public Reddit RSS feeds and stores the collected run locally. "
            "No Reddit API keys are required, but large runs may be rate-limited."
        )
    else:
        st.info(
            "Bundled fake dataset mode is active. "
            f"The packaged dataset currently has rows for {', '.join(fake_subreddits)}."
        )

    selected_source_ids = st.multiselect(
        "Active sources to include",
        active_sources["source_id"].tolist(),
        default=active_sources["source_id"].tolist(),
        format_func=lambda source_id: active_sources.loc[active_sources["source_id"] == source_id, "display_name"].iloc[0],
    )
    anchor = client.latest_available_timestamp() if data_mode == "fake" else datetime.now(UTC)
    previous_run = latest_matching_run(
        db_path,
        source_ids=[int(source_id) for source_id in selected_source_ids],
        data_mode=data_mode,
    ) if selected_source_ids else None
    use_last_run_window = st.toggle(
        "Use last completed run -> now",
        value=bool(previous_run),
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
    anchor_label = "latest bundled fake item" if data_mode == "fake" else "current time"
    st.caption(f"Window resolved against {anchor_label}: `{start_dt.date().isoformat()}` to `{end_dt.date().isoformat()}`.")
    if use_last_run_window and previous_run:
        st.caption(
            f"Most recent matching completed run ended at `{str(previous_run['time_window_end'])[:19]}` "
            f"for `{DATA_MODE_LABELS.get(str(previous_run['data_mode']), str(previous_run['data_mode']))}` mode."
        )

    run_disabled = False
    if st.button("Run Analysis", type="primary", use_container_width=True, disabled=run_disabled):
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
                data_path=data_path,
                ticker_path=ticker_path,
                db_path=db_path,
                artifacts_dir=artifacts_dir,
                data_mode=data_mode,
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
        st.write("Top mentioned tickers")
        st.dataframe(pd.DataFrame(summary.get("top_mentioned_tickers", [])), use_container_width=True, hide_index=True)
    with top_cols[1]:
        st.write("Top accelerating tickers")
        st.dataframe(pd.DataFrame(summary.get("top_accelerating_tickers", [])), use_container_width=True, hide_index=True)

    if summary.get("fallback_count", 0):
        st.warning(
            "Some items used heuristic fallback classification because the Ollama endpoint was unavailable "
            "or returned invalid JSON."
        )


def filter_results_dataframe(long_df: pd.DataFrame) -> pd.DataFrame:
    working = long_df.copy()
    st.subheader("Filters")
    filter_cols = st.columns(5)
    ticker_filter = filter_cols[0].multiselect("ticker", sorted(working["ticker"].dropna().unique()))
    source_filter = filter_cols[1].multiselect("source", sorted(working["source_name"].dropna().unique()))
    sentiment_filter = filter_cols[2].multiselect(
        "sentiment",
        ["bullish", "bearish", "neutral", "irrelevant"],
        default=["bullish", "bearish", "neutral", "irrelevant"],
    )
    min_confidence = filter_cols[3].slider("min confidence", 0.0, 1.0, 0.0, 0.05)
    min_depth = filter_cols[4].slider("min depth_score", 0, 10, 0)

    numeric_cols = st.columns(4)
    min_alpha = numeric_cols[0].slider("min alpha_signal_score", 0, 100, 0)
    max_hype = numeric_cols[1].slider("max hype_score", 0, 10, 10)
    hide_hype = numeric_cols[2].checkbox("hide hype/meme posts", value=False)
    only_deeper = numeric_cols[3].checkbox("show only needs_deeper_analysis", value=False)

    flag_cols = st.columns(3)
    only_low_signal = flag_cols[0].checkbox("show only low_mentions_high_signal", value=False)
    only_new = flag_cols[1].checkbox("show only new_ticker_detected", value=False)

    if ticker_filter:
        working = working[working["ticker"].isin(ticker_filter)]
    if source_filter:
        working = working[working["source_name"].isin(source_filter)]
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
        st.write(f"Collection mode: `{DATA_MODE_LABELS.get(str(run_record['data_mode']), str(run_record['data_mode']))}`")
        st.write(f"Collected window: `{run_record['time_window_start']}` to `{run_record['time_window_end']}`")
        st.write(f"Started at: `{run_record['started_at']}`")
        if run_record.get("completed_at"):
            st.write(f"Completed at: `{run_record['completed_at']}`")
        st.write(f"Sources: {', '.join(source_names) if source_names else 'None'}")


def render_results_dashboard_page(db_path: Path, run_id: str | None) -> None:
    st.header("Results Dashboard")
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

    long_df = load_run_mentions(db_path, run_id)
    if long_df.empty:
        st.info("No item-level mention data saved for this run.")
        return

    filtered = filter_results_dataframe(long_df)
    st.write(f"{len(filtered)} mention rows matched the current filters.")
    display_columns = [
        "created_time",
        "source_name",
        "ticker",
        "company_name",
        "sentiment",
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
    display_df = filtered[display_columns].rename(columns={"source_name": "source"})
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={"permalink": st.column_config.LinkColumn("permalink")},
    )

    st.subheader("Details")
    for row in filtered.head(12).to_dict(orient="records"):
        expander_title = f"{row['created_time'][:10]} · {row['ticker']} · {row['source_name']} · {row['sentiment']}"
        with st.expander(expander_title):
            detail_cols = st.columns(4)
            detail_cols[0].metric("confidence", f"{row['confidence']:.2f}")
            detail_cols[1].metric("alpha", f"{row['alpha_signal_score']:.1f}")
            detail_cols[2].metric("hype-adjusted", f"{row['hype_adjusted_signal_score']:.1f}")
            detail_cols[3].metric("controversy", f"{row['controversy_score']:.1f}")
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


def render_ticker_trends_page(db_path: Path, run_id: str | None) -> None:
    st.header("Ticker Trends")
    if not run_id:
        st.info("Run an analysis first.")
        return

    summary_df = load_run_ticker_summaries(db_path, run_id)
    bucket_df = load_run_time_buckets(db_path, run_id)
    long_df = load_run_mentions(db_path, run_id)
    if summary_df.empty:
        st.info("No ticker summary data saved for this run.")
        return

    st.subheader("Top accelerating and emerging tickers")
    top_cols = st.columns(2)
    with top_cols[0]:
        st.dataframe(
            summary_df[["ticker", "mentions_this_period", "mentions_previous_period", "acceleration_score"]].head(10),
            use_container_width=True,
            hide_index=True,
        )
    with top_cols[1]:
        st.dataframe(
            summary_df[["ticker", "emerging_ticker_score", "low_mentions_high_signal", "new_ticker_detected"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

    ticker_selection = st.multiselect(
        "Tickers for time-series charts",
        summary_df["ticker"].tolist(),
        default=summary_df["ticker"].head(4).tolist(),
    )

    if ticker_selection and not bucket_df.empty:
        chart_df = bucket_df[bucket_df["ticker"].isin(ticker_selection)]
        mention_chart = chart_df.pivot(index="bucket_start", columns="ticker", values="total_mentions").fillna(0)
        sentiment_chart = long_df[long_df["ticker"].isin(ticker_selection)].pivot_table(
            index="time_bucket",
            columns="ticker",
            values="sentiment_numeric",
            aggfunc="mean",
        ).fillna(0)
        st.write("Ticker mention counts over time")
        st.line_chart(mention_chart)
        st.write("Average sentiment over time")
        st.line_chart(sentiment_chart)

    viz_cols = st.columns(3)
    with viz_cols[0]:
        st.write("High-signal low-hype")
        st.dataframe(
            summary_df[summary_df["low_mentions_high_signal"]][
                ["ticker", "average_alpha_signal_score", "average_hype_score", "emerging_ticker_score"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with viz_cols[1]:
        st.write("Controversy by ticker")
        controversy_chart = summary_df.set_index("ticker")["controversy_score"]
        st.bar_chart(controversy_chart)
    with viz_cols[2]:
        st.write("Source diversity by ticker")
        diversity_chart = summary_df.set_index("ticker")["source_diversity_score"]
        st.bar_chart(diversity_chart)

    st.subheader("Ticker summary table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


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


def render_settings_page(db_path: Path, data_path: Path, ticker_path: Path, artifacts_dir: Path) -> None:
    st.header("Settings")
    overview = db_overview(db_path)
    overview_cols = st.columns(4)
    overview_cols[0].metric("Sources", overview["sources"])
    overview_cols[1].metric("Analysis runs", overview["analysis_runs"])
    overview_cols[2].metric("Reddit items", overview["reddit_items"])
    overview_cols[3].metric("Mention rows", overview["item_ticker_mentions"])

    st.write("Paths and environment")
    st.code(
        "\n".join(
            [
                f"db_path = {db_path}",
                f"artifacts_dir = {artifacts_dir}",
                f"storage_root = {DEFAULT_STORAGE_ROOT}",
                f"fake_data_path = {data_path}",
                f"ticker_catalog_path = {ticker_path}",
                f"ollama_url = {DEFAULT_OLLAMA_URL}",
                f"default_model = {DEFAULT_MODEL}",
                f"reddit_user_agent = {os.getenv('REDDIT_USER_AGENT', 'script:signal-from-the-slop:0.1')}",
            ]
        )
    )

    latest_fake_item = RedditClient(data_path).latest_available_timestamp()
    st.write(f"Latest bundled fake item: `{latest_fake_item.date().isoformat()}`")
    st.write("Live collection uses the no-key Reddit RSS scraper. Bundled fake data is still available for offline testing.")


init_db(DEFAULT_DB_PATH, SCHEMA_PATH)
ensure_seed_sources(DEFAULT_DB_PATH, DEFAULT_DATA_PATH, DEFAULT_SOURCES_PATH)
page, selected_run_id, _, _ = build_sidebar_state(DEFAULT_DB_PATH, DEFAULT_DATA_PATH)

if page == "Sources":
    render_sources_page(DEFAULT_DB_PATH, DEFAULT_DATA_PATH)
elif page == "Run Analysis":
    render_run_analysis_page(
        db_path=DEFAULT_DB_PATH,
        artifacts_dir=DEFAULT_ARTIFACTS_DIR,
        data_path=DEFAULT_DATA_PATH,
        ticker_path=DEFAULT_TICKER_PATH,
    )
elif page == "Results Dashboard":
    render_results_dashboard_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Ticker Trends":
    render_ticker_trends_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Export Data":
    render_export_page(DEFAULT_DB_PATH, selected_run_id)
elif page == "Settings":
    render_settings_page(DEFAULT_DB_PATH, DEFAULT_DATA_PATH, DEFAULT_TICKER_PATH, DEFAULT_ARTIFACTS_DIR)
