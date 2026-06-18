from __future__ import annotations

import csv
import hashlib
import logging
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from signal_from_the_slop.analytics import parse_created_time
from signal_from_the_slop.database import (
    create_signal_event,
    load_analysis_runs,
    load_completed_mentions,
    load_market_prices,
    load_signal_events,
    load_signal_random_benchmarks,
    upsert_market_prices,
    upsert_signal_attention_outcomes,
    upsert_signal_outcomes,
    upsert_signal_random_benchmarks,
)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency at runtime
    yf = None


LOGGER = logging.getLogger(__name__)

SIGNAL_SCORING_VERSION = "signal_validation_v2"
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_TICKER_UNIVERSE_NAME = "local_ticker_catalog"
AUTO_SIGNAL_TOP_N = 5
AUTO_SIGNAL_MIN_EMERGING = 60.0
AUTO_SIGNAL_MIN_ACCELERATION = 20.0
AUTO_SIGNAL_MIN_COVERAGE_RELIABILITY = 0.5
AUTO_SIGNAL_MIN_MENTIONS = 3
AUTO_SIGNAL_MIN_UNIQUE_AUTHORS = 2
AUTO_SIGNAL_MIN_UNIQUE_THREADS = 2
AUTO_SIGNAL_ALLOWED_TREND_RELIABILITY = {"Medium", "High"}
SIGNAL_COOLDOWN_HOURS = 24
MATERIAL_SIGNAL_DELTA = 12.0
RANDOM_BENCHMARK_SAMPLE_SIZE = 12
MARKET_LOOKBACK_CALENDAR_DAYS = 60
MARKET_FORWARD_CALENDAR_DAYS = 45
MARKET_TICKER_ALIASES = {
    "SQ": "XYZ",
}


def run_signal_validation(
    *,
    db_path: Path,
    ticker_path: Path,
    run_id: str,
    run_config: dict[str, Any],
    summary_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    signal_time: str | None = None,
    manual_tickers: list[str] | None = None,
    allow_automatic_signals: bool = True,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> dict[str, Any]:
    signal_time = signal_time or _now_iso()
    created_signal_ids = freeze_signal_events_for_run(
        db_path=db_path,
        run_id=run_id,
        run_config=run_config,
        summary_df=summary_df,
        mentions_df=mentions_df,
        signal_time=signal_time,
        manual_tickers=manual_tickers or [],
        allow_automatic_signals=allow_automatic_signals,
    )
    refresh_summary = refresh_signal_validation(
        db_path=db_path,
        ticker_path=ticker_path,
        benchmark_ticker=benchmark_ticker,
    )
    refresh_summary["signals_created"] = len(created_signal_ids)
    refresh_summary["created_signal_ids"] = created_signal_ids
    return refresh_summary


def refresh_signal_validation(
    *,
    db_path: Path,
    ticker_path: Path,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    signal_ids: list[str] | None = None,
) -> dict[str, Any]:
    events = load_signal_events(db_path)
    if events.empty:
        return {
            "status": "idle",
            "signals_total": 0,
            "signals_refreshed": 0,
            "market_dependency_available": yf is not None,
            "benchmark_ticker": benchmark_ticker,
            "universe_name": DEFAULT_TICKER_UNIVERSE_NAME,
            "market_data_errors": [],
        }

    if signal_ids:
        allowed = {str(signal_id) for signal_id in signal_ids}
        events = events[events["signal_id"].astype(str).isin(allowed)].copy()
    if events.empty:
        return {
            "status": "idle",
            "signals_total": 0,
            "signals_refreshed": 0,
            "market_dependency_available": yf is not None,
            "benchmark_ticker": benchmark_ticker,
            "universe_name": DEFAULT_TICKER_UNIVERSE_NAME,
            "market_data_errors": [],
        }

    events["signal_time_dt"] = pd.to_datetime(events["signal_time"], errors="coerce", utc=True)
    events = events.dropna(subset=["signal_time_dt"]).copy()
    if events.empty:
        return {
            "status": "idle",
            "signals_total": 0,
            "signals_refreshed": 0,
            "market_dependency_available": yf is not None,
            "benchmark_ticker": benchmark_ticker,
            "universe_name": DEFAULT_TICKER_UNIVERSE_NAME,
            "market_data_errors": [],
        }

    benchmark_ticker = benchmark_ticker.upper()
    universe = load_ticker_universe(ticker_path)
    signal_records = events.to_dict(orient="records")
    _ensure_random_benchmark_samples(
        db_path=db_path,
        signal_records=signal_records,
        universe=universe,
        universe_name=DEFAULT_TICKER_UNIVERSE_NAME,
    )
    random_benchmarks = load_signal_random_benchmarks(db_path)
    if signal_ids and not random_benchmarks.empty:
        allowed = {str(signal_id) for signal_id in signal_ids}
        random_benchmarks = random_benchmarks[random_benchmarks["signal_id"].astype(str).isin(allowed)].copy()

    market_data_errors: list[str] = []
    market_refresh_count = 0
    if yf is not None:
        required_market_tickers = _required_market_tickers(signal_records, random_benchmarks, benchmark_ticker)
        ticker_windows = _ticker_market_windows(signal_records, random_benchmarks, benchmark_ticker)
        for ticker in required_market_tickers:
            start_date, end_date = ticker_windows[ticker]
            fetch_result = ensure_market_data(
                db_path=db_path,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if fetch_result.get("fetched"):
                market_refresh_count += 1
            error = str(fetch_result.get("error", "") or "").strip()
            if error:
                market_data_errors.append(f"{ticker}: {error}")
    else:
        market_data_errors.append("yfinance is not installed, so market outcomes were refreshed from cache only.")

    market_outcome_rows = build_market_outcomes(db_path=db_path, signal_records=signal_records, benchmark_ticker=benchmark_ticker)
    upsert_signal_outcomes(db_path, market_outcome_rows)

    random_outcome_rows = build_random_benchmark_outcomes(
        db_path=db_path,
        random_benchmarks=random_benchmarks,
    )
    upsert_signal_random_benchmarks(db_path, random_outcome_rows)

    attention_rows = build_attention_outcomes(db_path=db_path, signal_records=signal_records)
    upsert_signal_attention_outcomes(db_path, attention_rows)

    status = "ok"
    if market_data_errors:
        status = "partial"
    return {
        "status": status,
        "signals_total": len(signal_records),
        "signals_refreshed": len(signal_records),
        "market_dependency_available": yf is not None,
        "market_tickers_refreshed": market_refresh_count,
        "market_outcomes_updated": len(market_outcome_rows),
        "attention_outcomes_updated": len(attention_rows),
        "random_benchmark_rows_updated": len(random_outcome_rows),
        "benchmark_ticker": benchmark_ticker,
        "universe_name": DEFAULT_TICKER_UNIVERSE_NAME,
        "market_data_errors": market_data_errors[:12],
    }


def freeze_signal_events_for_run(
    *,
    db_path: Path,
    run_id: str,
    run_config: dict[str, Any],
    summary_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    signal_time: str,
    manual_tickers: list[str],
    allow_automatic_signals: bool,
) -> list[str]:
    if summary_df.empty:
        return []

    summary_working = summary_df.copy()
    summary_working["ticker"] = summary_working["ticker"].astype(str).str.upper()
    mentions_working = mentions_df.copy() if not mentions_df.empty else pd.DataFrame()
    if not mentions_working.empty:
        mentions_working["ticker"] = mentions_working["ticker"].astype(str).str.upper()
        mentions_working["created_time_dt"] = pd.to_datetime(mentions_working["created_time"], errors="coerce", utc=True)
    source_set_hash = compute_source_set_hash(run_config.get("selected_source_ids", []))
    manual_set = {str(ticker).upper() for ticker in manual_tickers if str(ticker).strip()}
    summary_working = summary_working.sort_values(
        ["emerging_ticker_score", "acceleration_score", "average_alpha_signal_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    total_scope_mentions = max(int(summary_working["total_mentions"].sum()), 1)
    existing_run_events = load_signal_events(db_path, analysis_run_id=run_id)
    existing_pairs = {
        (str(row["analysis_run_id"]), str(row["ticker"]).upper())
        for row in existing_run_events.to_dict(orient="records")
    } if not existing_run_events.empty else set()

    created_signal_ids: list[str] = []
    for index, row in enumerate(summary_working.to_dict(orient="records"), start=1):
        ticker = str(row.get("ticker", "")).upper()
        if not ticker or ticker == "UNKNOWN":
            continue
        manual_signal = ticker in manual_set
        automatic_score_qualifies = (
            index <= AUTO_SIGNAL_TOP_N
            or _coerce_float(row.get("emerging_ticker_score")) >= AUTO_SIGNAL_MIN_EMERGING
            or _coerce_float(row.get("acceleration_score")) >= AUTO_SIGNAL_MIN_ACCELERATION
        )
        evidence_stats = _ticker_evidence_stats(mentions_working, row, ticker)
        qualifies_automatic = (
            allow_automatic_signals
            and automatic_score_qualifies
            and _automatic_signal_quality_passes(row, evidence_stats)
        )
        if not manual_signal and not qualifies_automatic:
            continue
        if (run_id, ticker) in existing_pairs:
            continue

        top_evidence = _build_top_evidence(mentions_working, ticker)
        candidate = {
            "ticker": ticker,
            "company_name": str(row.get("company_name", "Unknown") or "Unknown"),
            "signal_time": signal_time,
            "analysis_run_id": run_id,
            "run_type": str(run_config.get("run_type", "discovery")),
            "source_set_hash": source_set_hash,
            "signal_rank": index,
            "emerging_ticker_score": round(_coerce_float(row.get("emerging_ticker_score")), 2),
            "adjusted_acceleration_score": round(_coerce_float(row.get("acceleration_score")), 2),
            "alpha_signal_score": round(_coerce_float(row.get("average_alpha_signal_score")), 2),
            "hype_score": round(_coerce_float(row.get("average_hype_score")), 2),
            "hype_adjusted_signal_score": round(_coerce_float(row.get("average_hype_adjusted_signal_score")), 2),
            "evidence_quality": round(_coerce_float(row.get("average_evidence_quality_numeric")), 2),
            "claim_specificity_score": round(_coerce_float(row.get("average_claim_specificity_score")), 2),
            "source_diversity": round(_coerce_float(row.get("source_diversity_score")), 2),
            "controversy_score": round(_coerce_float(row.get("controversy_score")), 2),
            "mention_rate": round(_coerce_float(row.get("mention_rate_per_100_posts")), 3),
            "thread_rate": round(_coerce_float(row.get("thread_rate_per_100_posts")), 3),
            "author_rate": round(_coerce_float(row.get("author_rate_per_100_posts")), 3),
            "share_of_voice": round((_coerce_float(row.get("total_mentions")) / total_scope_mentions) * 100, 3),
            "trend_reliability": str(row.get("trend_reliability", "")),
            "coverage_reliability": round(_coerce_float(row.get("coverage_reliability")), 3),
            "top_evidence_item_ids": top_evidence["item_ids"],
            "top_evidence_titles": top_evidence["titles"],
            "top_evidence_excerpts": top_evidence["excerpts"],
            "scoring_version": SIGNAL_SCORING_VERSION,
            "manual_signal": manual_signal,
            "created_at": signal_time,
        }
        recent_events = load_signal_events(db_path, ticker=ticker, limit=8)
        if not _should_create_signal_event(candidate, recent_events, manual_signal=manual_signal):
            continue
        signal_id = create_signal_event(db_path, candidate)
        created_signal_ids.append(signal_id)
        existing_pairs.add((run_id, ticker))
    return created_signal_ids


def compute_source_set_hash(source_ids: list[Any]) -> str:
    normalized = ",".join(str(int(source_id)) for source_id in sorted(int(source_id) for source_id in source_ids))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def load_ticker_universe(ticker_path: Path) -> list[str]:
    tickers: list[str] = []
    try:
        with ticker_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ticker = market_symbol_for_ticker(str(row.get("ticker", "")).strip().upper())
                if ticker:
                    tickers.append(ticker)
    except OSError:
        LOGGER.exception("Could not load ticker universe from %s", ticker_path)
    return sorted(set(tickers))


def ensure_market_data(
    *,
    db_path: Path,
    ticker: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    requested_ticker = ticker.upper()
    ticker = market_symbol_for_ticker(requested_ticker)
    cached = load_market_prices(
        db_path,
        ticker=ticker,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    if _market_data_is_sufficient(cached, start_date, end_date):
        return {"ticker": ticker, "fetched": False, "cached_rows": len(cached), "error": ""}
    if yf is None:
        return {"ticker": ticker, "fetched": False, "cached_rows": len(cached), "error": "missing yfinance dependency"}

    try:
        frame = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            interval="1d",
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network/provider failures
        LOGGER.warning("Market data fetch failed for %s: %s", ticker, exc)
        return {
            "ticker": requested_ticker,
            "market_ticker": ticker,
            "fetched": False,
            "cached_rows": len(cached),
            "error": str(exc),
        }

    normalized_rows = _normalize_market_rows(frame, ticker=ticker)
    if not normalized_rows:
        return {
            "ticker": requested_ticker,
            "market_ticker": ticker,
            "fetched": False,
            "cached_rows": len(cached),
            "error": "no market data returned",
        }
    upsert_market_prices(db_path, normalized_rows)
    return {
        "ticker": requested_ticker,
        "market_ticker": ticker,
        "fetched": True,
        "cached_rows": len(cached),
        "error": "",
    }


def build_market_outcomes(
    *,
    db_path: Path,
    signal_records: list[dict[str, Any]],
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_cache: dict[str, pd.DataFrame] = {}
    price_cache: dict[str, pd.DataFrame] = {}
    for signal in signal_records:
        ticker = str(signal.get("ticker", "")).upper()
        signal_time = _coerce_datetime(signal.get("signal_time"))
        if not ticker or signal_time is None:
            continue
        market_ticker = market_symbol_for_ticker(ticker)
        prices = price_cache.setdefault(
            market_ticker,
            _prepare_market_prices(load_market_prices(db_path, ticker=market_ticker)),
        )
        benchmark_prices = benchmark_cache.setdefault(
            market_symbol_for_ticker(benchmark_ticker),
            _prepare_market_prices(load_market_prices(db_path, ticker=market_symbol_for_ticker(benchmark_ticker))),
        )
        outcome = _build_single_market_outcome(
            ticker=ticker,
            signal_time=signal_time,
            signal_id=str(signal["signal_id"]),
            prices=prices,
            benchmark_prices=benchmark_prices,
        )
        if outcome:
            rows.append(outcome)
    return rows


def build_random_benchmark_outcomes(
    *,
    db_path: Path,
    random_benchmarks: pd.DataFrame,
) -> list[dict[str, Any]]:
    if random_benchmarks.empty:
        return []
    working = random_benchmarks.copy()
    working["signal_date_dt"] = pd.to_datetime(working["signal_date"], errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for record in working.to_dict(orient="records"):
        ticker = str(record.get("benchmark_ticker", "")).upper()
        signal_time = _coerce_datetime(record.get("signal_date"))
        if not ticker or signal_time is None:
            continue
        market_ticker = market_symbol_for_ticker(ticker)
        prices = price_cache.setdefault(
            market_ticker,
            _prepare_market_prices(load_market_prices(db_path, ticker=market_ticker)),
        )
        outcome = _build_single_random_benchmark_outcome(
            signal_id=str(record["signal_id"]),
            benchmark_ticker=ticker,
            signal_time=signal_time,
            sample_index=int(record.get("sample_index", 0) or 0),
            universe_name=str(record.get("universe_name", "")),
            sample_seed=str(record.get("sample_seed", "")),
            prices=prices,
        )
        if outcome:
            rows.append(outcome)
    return rows


def build_attention_outcomes(
    *,
    db_path: Path,
    signal_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions = load_completed_mentions(db_path, data_mode="live")
    runs = load_analysis_runs(db_path)
    coverage_lookup = _run_coverage_lookup(runs)
    completed_measurement_runs = _completed_measurement_runs(runs)
    if mentions.empty:
        return [
            {
                "signal_id": str(signal["signal_id"]),
                "ticker": str(signal.get("ticker", "")).upper(),
                "signal_time": str(signal.get("signal_time", "")),
                "future_mentions_7d": 0,
                "future_mentions_14d": 0,
                "future_mentions_30d": 0,
                "future_unique_threads_7d": 0,
                "future_unique_threads_14d": 0,
                "future_unique_threads_30d": 0,
                "future_unique_authors_7d": 0,
                "future_unique_authors_14d": 0,
                "future_unique_authors_30d": 0,
                "future_source_count_7d": 0,
                "future_source_count_14d": 0,
                "future_source_count_30d": 0,
                "future_share_of_voice_7d": None,
                "future_share_of_voice_14d": None,
                "future_share_of_voice_30d": None,
                "attention_spread_7d": 0.0,
                "attention_spread_14d": 0.0,
                "attention_spread_30d": 0.0,
                "attention_outcome_reliability": "Low",
                "outcome_last_updated": _now_iso(),
            }
            for signal in signal_records
        ]

    working = mentions.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper()
    working["created_time_dt"] = pd.to_datetime(working["created_time"], errors="coerce", utc=True)
    working = working.dropna(subset=["created_time_dt"]).copy()

    rows: list[dict[str, Any]] = []
    for signal in signal_records:
        ticker = str(signal.get("ticker", "")).upper()
        signal_time = _coerce_datetime(signal.get("signal_time"))
        if not ticker or signal_time is None:
            continue
        future_mentions = working[
            (working["ticker"] == ticker)
            & (working["created_time_dt"] > signal_time)
        ].copy()
        future_measurement_run_ids = _future_measurement_run_ids(
            completed_measurement_runs,
            signal_time=signal_time,
            max_window_days=30,
        )
        preferred_mode = "measurement" if future_measurement_run_ids else "corpus"
        if future_measurement_run_ids:
            selected_mentions = future_mentions[future_mentions["analysis_run_id"].astype(str).isin(future_measurement_run_ids)].copy()
            universe_mentions = working[working["analysis_run_id"].astype(str).isin(future_measurement_run_ids)].copy()
        else:
            selected_mentions = future_mentions
            universe_mentions = working[working["created_time_dt"] > signal_time].copy()

        row = {
            "signal_id": str(signal["signal_id"]),
            "ticker": ticker,
            "signal_time": str(signal.get("signal_time", "")),
            "attention_outcome_reliability": _attention_reliability_label(
                selected_mentions=selected_mentions,
                future_measurement_run_ids=future_measurement_run_ids,
                coverage_lookup=coverage_lookup,
                fallback_mode=preferred_mode,
            ),
            "outcome_last_updated": _now_iso(),
        }
        for window in (7, 14, 30):
            window_end = signal_time + timedelta(days=window)
            ticker_window = selected_mentions[selected_mentions["created_time_dt"] <= window_end].copy()
            universe_window = universe_mentions[universe_mentions["created_time_dt"] <= window_end].copy()
            share_of_voice = None
            if not universe_window.empty:
                share_of_voice = round((len(ticker_window) / len(universe_window)) * 100, 3)
            row[f"future_mentions_{window}d"] = int(len(ticker_window))
            row[f"future_unique_threads_{window}d"] = int(ticker_window["thread_id"].nunique()) if not ticker_window.empty else 0
            row[f"future_unique_authors_{window}d"] = int(ticker_window["author_hash"].nunique()) if not ticker_window.empty else 0
            row[f"future_source_count_{window}d"] = int(ticker_window["source_name"].nunique()) if not ticker_window.empty else 0
            row[f"future_share_of_voice_{window}d"] = share_of_voice
            row[f"attention_spread_{window}d"] = round(
                min(
                    100.0,
                    row[f"future_mentions_{window}d"] * 3.5
                    + row[f"future_unique_threads_{window}d"] * 5.0
                    + row[f"future_unique_authors_{window}d"] * 2.5
                    + row[f"future_source_count_{window}d"] * 10.0
                    + (_coerce_float(share_of_voice) * 0.35),
                ),
                2,
            )
        rows.append(row)
    return rows


def _build_top_evidence(mentions_df: pd.DataFrame, ticker: str) -> dict[str, list[str]]:
    if mentions_df.empty:
        return {"item_ids": [], "titles": [], "excerpts": []}
    ranked = mentions_df[mentions_df["ticker"] == ticker].copy()
    if ranked.empty:
        return {"item_ids": [], "titles": [], "excerpts": []}
    ranked = ranked.sort_values(
        ["alpha_signal_score", "confidence", "created_time_dt"],
        ascending=[False, False, False],
        na_position="last",
    ).head(3)
    titles = ranked["thread_title"].fillna("").astype(str).map(lambda value: value[:160]).tolist()
    excerpts = ranked.apply(
        lambda row: _short_excerpt(str(row.get("summary", "") or row.get("body_text", ""))),
        axis=1,
    ).tolist()
    return {
        "item_ids": ranked["item_id"].fillna("").astype(str).tolist(),
        "titles": titles,
        "excerpts": excerpts,
    }


def _ticker_evidence_stats(mentions_df: pd.DataFrame, row: dict[str, Any], ticker: str) -> dict[str, Any]:
    stats = {
        "total_mentions": int(_coerce_float(row.get("total_mentions"))),
        "unique_authors": int(_coerce_float(row.get("unique_authors"))),
        "unique_threads": int(_coerce_float(row.get("unique_threads"))),
        "has_non_fallback_mention": False,
    }
    if mentions_df.empty:
        return stats

    ticker_mentions = mentions_df[mentions_df["ticker"] == ticker].copy()
    if ticker_mentions.empty:
        return stats

    stats["total_mentions"] = max(stats["total_mentions"], len(ticker_mentions))
    if "author_hash" in ticker_mentions.columns:
        stats["unique_authors"] = max(stats["unique_authors"], int(ticker_mentions["author_hash"].nunique()))
    if "thread_id" in ticker_mentions.columns:
        stats["unique_threads"] = max(stats["unique_threads"], int(ticker_mentions["thread_id"].nunique()))
    if "classifier_mode" in ticker_mentions.columns:
        modes = ticker_mentions["classifier_mode"].fillna("").astype(str).str.lower().str.strip()
        stats["has_non_fallback_mention"] = bool(modes.eq("ollama").any())
    return stats


def _automatic_signal_quality_passes(row: dict[str, Any], evidence_stats: dict[str, Any]) -> bool:
    if _coerce_float(row.get("coverage_reliability")) < AUTO_SIGNAL_MIN_COVERAGE_RELIABILITY:
        return False
    if str(row.get("trend_reliability", "")).strip().title() not in AUTO_SIGNAL_ALLOWED_TREND_RELIABILITY:
        return False
    if int(evidence_stats.get("total_mentions", 0)) < AUTO_SIGNAL_MIN_MENTIONS:
        return False
    if int(evidence_stats.get("unique_authors", 0)) < AUTO_SIGNAL_MIN_UNIQUE_AUTHORS:
        return False
    if int(evidence_stats.get("unique_threads", 0)) < AUTO_SIGNAL_MIN_UNIQUE_THREADS:
        return False
    return bool(evidence_stats.get("has_non_fallback_mention", False))


def _should_create_signal_event(candidate: dict[str, Any], recent_events: pd.DataFrame, *, manual_signal: bool) -> bool:
    if recent_events.empty:
        return True
    recent = recent_events.copy()
    recent["signal_time_dt"] = pd.to_datetime(recent["signal_time"], errors="coerce", utc=True)
    recent = recent.dropna(subset=["signal_time_dt"]).sort_values("signal_time_dt", ascending=False)
    if recent.empty:
        return True
    latest = recent.iloc[0].to_dict()
    if str(latest.get("analysis_run_id", "")) == str(candidate.get("analysis_run_id", "")):
        return False
    if manual_signal:
        return True
    signal_time = _coerce_datetime(candidate.get("signal_time"))
    latest_time = _coerce_datetime(latest.get("signal_time"))
    if signal_time is None or latest_time is None:
        return True
    if signal_time - latest_time > timedelta(hours=SIGNAL_COOLDOWN_HOURS):
        return True
    candidate_strength = _signal_strength(candidate)
    latest_strength = _signal_strength(latest)
    return candidate_strength >= latest_strength + MATERIAL_SIGNAL_DELTA


def _signal_strength(row: dict[str, Any]) -> float:
    return (
        _coerce_float(row.get("emerging_ticker_score")) * 0.55
        + _coerce_float(row.get("adjusted_acceleration_score")) * 0.25
        + _coerce_float(row.get("alpha_signal_score")) * 0.20
        - _coerce_float(row.get("hype_score")) * 0.10
    )


def _ensure_random_benchmark_samples(
    *,
    db_path: Path,
    signal_records: list[dict[str, Any]],
    universe: list[str],
    universe_name: str,
) -> None:
    if not universe:
        return
    existing = load_signal_random_benchmarks(db_path)
    existing_signal_ids = set(existing["signal_id"].astype(str).unique().tolist()) if not existing.empty else set()
    rows: list[dict[str, Any]] = []
    for signal in signal_records:
        signal_id = str(signal["signal_id"])
        if signal_id in existing_signal_ids:
            continue
        ticker = str(signal.get("ticker", "")).upper()
        eligible = [candidate for candidate in universe if candidate and candidate != ticker]
        if not eligible:
            continue
        sample_size = min(RANDOM_BENCHMARK_SAMPLE_SIZE, len(eligible))
        sample_seed = hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:16]
        sampler = random.Random(sample_seed)
        sampled = sampler.sample(eligible, sample_size)
        signal_date = str(signal.get("signal_time", ""))[:10]
        for index, benchmark_ticker in enumerate(sampled, start=1):
            rows.append(
                {
                    "signal_id": signal_id,
                    "benchmark_ticker": benchmark_ticker,
                    "sample_index": index,
                    "universe_name": universe_name,
                    "sample_seed": sample_seed,
                    "signal_date": signal_date,
                    "outcome_last_updated": _now_iso(),
                }
            )
    if rows:
        upsert_signal_random_benchmarks(db_path, rows)


def _required_market_tickers(
    signal_records: list[dict[str, Any]],
    random_benchmarks: pd.DataFrame,
    benchmark_ticker: str,
) -> list[str]:
    tickers = {market_symbol_for_ticker(benchmark_ticker.upper())}
    tickers.update(
        market_symbol_for_ticker(str(signal.get("ticker", "")).upper())
        for signal in signal_records
        if str(signal.get("ticker", "")).strip()
    )
    if not random_benchmarks.empty:
        tickers.update(
            market_symbol_for_ticker(value)
            for value in random_benchmarks["benchmark_ticker"].dropna().astype(str).str.upper().tolist()
        )
    return sorted(ticker for ticker in tickers if ticker)


def _ticker_market_windows(
    signal_records: list[dict[str, Any]],
    random_benchmarks: pd.DataFrame,
    benchmark_ticker: str,
) -> dict[str, tuple[date, date]]:
    windows: dict[str, tuple[date, date]] = {}
    today = datetime.now(UTC).date()
    def update_window(ticker: str, signal_day: date) -> None:
        start = signal_day - timedelta(days=MARKET_LOOKBACK_CALENDAR_DAYS)
        end = min(today, signal_day + timedelta(days=MARKET_FORWARD_CALENDAR_DAYS))
        if ticker not in windows:
            windows[ticker] = (start, end)
            return
        existing_start, existing_end = windows[ticker]
        windows[ticker] = (min(existing_start, start), max(existing_end, end))

    for signal in signal_records:
        signal_time = _coerce_datetime(signal.get("signal_time"))
        ticker = market_symbol_for_ticker(str(signal.get("ticker", "")).upper())
        if signal_time is None or not ticker:
            continue
        update_window(ticker, signal_time.date())
        update_window(market_symbol_for_ticker(benchmark_ticker.upper()), signal_time.date())
    if not random_benchmarks.empty:
        for record in random_benchmarks.to_dict(orient="records"):
            ticker = market_symbol_for_ticker(str(record.get("benchmark_ticker", "")).upper())
            signal_time = _coerce_datetime(record.get("signal_date"))
            if signal_time is None or not ticker:
                continue
            update_window(ticker, signal_time.date())
    return windows


def _market_data_is_sufficient(frame: pd.DataFrame, start_date: date, end_date: date) -> bool:
    if frame.empty:
        return False
    working = frame.copy()
    working["date_dt"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working = working.dropna(subset=["date_dt"])
    if working.empty:
        return False
    min_date = working["date_dt"].min().date()
    max_date = working["date_dt"].max().date()
    if min_date > start_date:
        return False
    if max_date >= end_date:
        return True
    today = datetime.now(UTC).date()
    required_end = min(end_date, today)
    return max_date >= required_end


def _normalize_market_rows(frame: pd.DataFrame, *, ticker: str) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    if isinstance(working.columns, pd.MultiIndex):
        working.columns = [str(column[0]) if isinstance(column, tuple) else str(column) for column in working.columns]
    working = working.reset_index()
    if "Date" not in working.columns:
        working = working.rename(columns={working.columns[0]: "Date"})
    rows: list[dict[str, Any]] = []
    fetched_at = _now_iso()
    for row in working.to_dict(orient="records"):
        row_date = pd.to_datetime(row.get("Date"), errors="coerce", utc=True)
        if pd.isna(row_date):
            continue
        rows.append(
            {
                "ticker": ticker,
                "date": row_date.date().isoformat(),
                "open": _coerce_float_or_none(row.get("Open")),
                "high": _coerce_float_or_none(row.get("High")),
                "low": _coerce_float_or_none(row.get("Low")),
                "close": _coerce_float_or_none(row.get("Close")),
                "adjusted_close": _coerce_float_or_none(row.get("Adj Close", row.get("Close"))),
                "volume": _coerce_float_or_none(row.get("Volume")),
                "data_source": "yfinance",
                "fetched_at": fetched_at,
            }
        )
    return rows


def _prepare_market_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working["date_dt"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working = working.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)
    working["reference_close"] = working["adjusted_close"].where(
        working["adjusted_close"].notna(),
        working["close"],
    )
    return working


def _build_single_market_outcome(
    *,
    ticker: str,
    signal_time: datetime,
    signal_id: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
) -> dict[str, Any] | None:
    start_row = _first_row_on_or_after(prices, signal_time.date())
    if start_row is None:
        return {
            "signal_id": signal_id,
            "ticker": ticker,
            "signal_time": signal_time.isoformat(),
            "outcome_last_updated": _now_iso(),
        }
    start_price = _coerce_float_or_none(start_row.get("reference_close"))
    baseline_volume = _baseline_volume(prices, start_row_index=int(start_row.name))
    benchmark_start_row = _first_row_on_or_after(benchmark_prices, signal_time.date())
    row: dict[str, Any] = {
        "signal_id": signal_id,
        "ticker": ticker,
        "signal_time": signal_time.isoformat(),
        "price_at_signal": start_price,
        "outcome_last_updated": _now_iso(),
    }
    for window in (1, 3, 7, 14, 30):
        row[f"return_{window}d"] = _forward_return(prices, start_row, signal_time.date(), window)
    for window in (7, 14, 30):
        max_gain, max_drawdown = _path_metrics(prices, start_row, signal_time.date(), window)
        row[f"max_gain_{window}d"] = max_gain
        row[f"max_drawdown_{window}d"] = max_drawdown
    for window in (3, 7, 14, 30):
        volume_spike, abnormal_volume = _volume_metrics(prices, start_row, signal_time.date(), window, baseline_volume)
        row[f"volume_spike_{window}d"] = volume_spike
        if window in {7, 30}:
            row[f"abnormal_volume_{window}d"] = abnormal_volume
    row["benchmark_return_7d"] = _forward_return(benchmark_prices, benchmark_start_row, signal_time.date(), 7) if benchmark_start_row is not None else None
    row["benchmark_return_30d"] = _forward_return(benchmark_prices, benchmark_start_row, signal_time.date(), 30) if benchmark_start_row is not None else None
    row["excess_return_7d"] = _difference_or_none(row.get("return_7d"), row.get("benchmark_return_7d"))
    row["excess_return_30d"] = _difference_or_none(row.get("return_30d"), row.get("benchmark_return_30d"))
    return row


def _build_single_random_benchmark_outcome(
    *,
    signal_id: str,
    benchmark_ticker: str,
    signal_time: datetime,
    sample_index: int,
    universe_name: str,
    sample_seed: str,
    prices: pd.DataFrame,
) -> dict[str, Any] | None:
    start_row = _first_row_on_or_after(prices, signal_time.date())
    if start_row is None:
        return {
            "signal_id": signal_id,
            "benchmark_ticker": benchmark_ticker,
            "sample_index": sample_index,
            "universe_name": universe_name,
            "sample_seed": sample_seed,
            "signal_date": signal_time.date().isoformat(),
            "outcome_last_updated": _now_iso(),
        }
    baseline_volume = _baseline_volume(prices, start_row_index=int(start_row.name))
    max_gain_7d, max_drawdown_7d = _path_metrics(prices, start_row, signal_time.date(), 7)
    volume_spike_7d, _ = _volume_metrics(prices, start_row, signal_time.date(), 7, baseline_volume)
    volume_spike_30d, _ = _volume_metrics(prices, start_row, signal_time.date(), 30, baseline_volume)
    return {
        "signal_id": signal_id,
        "benchmark_ticker": benchmark_ticker,
        "sample_index": sample_index,
        "universe_name": universe_name,
        "sample_seed": sample_seed,
        "signal_date": signal_time.date().isoformat(),
        "price_at_signal": _coerce_float_or_none(start_row.get("reference_close")),
        "return_7d": _forward_return(prices, start_row, signal_time.date(), 7),
        "return_30d": _forward_return(prices, start_row, signal_time.date(), 30),
        "max_gain_7d": max_gain_7d,
        "max_drawdown_7d": max_drawdown_7d,
        "volume_spike_7d": volume_spike_7d,
        "volume_spike_30d": volume_spike_30d,
        "outcome_last_updated": _now_iso(),
    }


def _baseline_volume(prices: pd.DataFrame, *, start_row_index: int) -> float | None:
    if prices.empty or start_row_index <= 0:
        return None
    history = prices.iloc[max(start_row_index - 20, 0):start_row_index].copy()
    if history.empty or "volume" not in history.columns:
        return None
    mean_volume = history["volume"].dropna().astype(float).mean()
    if pd.isna(mean_volume):
        return None
    return float(mean_volume)


def _first_row_on_or_after(prices: pd.DataFrame, target_date: date) -> pd.Series | None:
    if prices is None or prices.empty:
        return None
    filtered = prices[prices["date_dt"].dt.date >= target_date]
    if filtered.empty:
        return None
    return filtered.iloc[0]


def _forward_return(prices: pd.DataFrame, start_row: pd.Series | None, signal_date: date, window_days: int) -> float | None:
    if start_row is None:
        return None
    start_price = _coerce_float_or_none(start_row.get("reference_close"))
    if start_price in (None, 0):
        return None
    target_date = signal_date + timedelta(days=window_days)
    future_row = _first_row_on_or_after(prices, target_date)
    if future_row is None:
        return None
    future_price = _coerce_float_or_none(future_row.get("reference_close"))
    if future_price is None:
        return None
    return round((future_price / start_price) - 1, 4)


def _path_metrics(prices: pd.DataFrame, start_row: pd.Series, signal_date: date, window_days: int) -> tuple[float | None, float | None]:
    start_price = _coerce_float_or_none(start_row.get("reference_close"))
    if start_price in (None, 0):
        return None, None
    window_end = signal_date + timedelta(days=window_days)
    window = prices[
        (prices["date_dt"].dt.date >= signal_date)
        & (prices["date_dt"].dt.date <= window_end)
    ].copy()
    if window.empty:
        return None, None
    closes = window["reference_close"].dropna().astype(float)
    if closes.empty:
        return None, None
    max_gain = round((closes.max() / start_price) - 1, 4)
    max_drawdown = round((closes.min() / start_price) - 1, 4)
    return max_gain, max_drawdown


def _volume_metrics(
    prices: pd.DataFrame,
    start_row: pd.Series,
    signal_date: date,
    window_days: int,
    baseline_volume: float | None,
) -> tuple[float | None, float | None]:
    if baseline_volume in (None, 0):
        return None, None
    window_end = signal_date + timedelta(days=window_days)
    window = prices[
        (prices["date_dt"].dt.date > signal_date)
        & (prices["date_dt"].dt.date <= window_end)
    ].copy()
    if window.empty:
        return None, None
    volumes = window["volume"].dropna().astype(float)
    if volumes.empty:
        return None, None
    return (
        round(float(volumes.max() / baseline_volume), 4),
        round(float(volumes.mean() / baseline_volume), 4),
    )


def _difference_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 4)


def _run_coverage_lookup(runs: pd.DataFrame) -> dict[str, float]:
    if runs.empty:
        return {}
    lookup: dict[str, float] = {}
    for row in runs.to_dict(orient="records"):
        summary = row.get("summary_json", {}) or {}
        coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage", {}), dict) else {}
        lookup[str(row["analysis_run_id"])] = _coerce_float(coverage.get("coverage_reliability"))
    return lookup


def _completed_measurement_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    working = runs[
        (runs["status"] == "completed")
        & (runs["data_mode"] == "live")
        & (runs["run_type"].astype(str).str.lower() == "measurement")
        & (runs["canonical_trend_run"].fillna(False).astype(bool))
    ].copy()
    if working.empty:
        return working
    working["completed_at_dt"] = pd.to_datetime(working["completed_at"], errors="coerce", utc=True)
    return working.dropna(subset=["completed_at_dt"]).copy()


def _future_measurement_run_ids(
    completed_measurement_runs: pd.DataFrame,
    *,
    signal_time: datetime,
    max_window_days: int,
) -> list[str]:
    if completed_measurement_runs.empty:
        return []
    window_end = signal_time + timedelta(days=max_window_days)
    window_runs = completed_measurement_runs[
        (completed_measurement_runs["completed_at_dt"] > signal_time)
        & (completed_measurement_runs["completed_at_dt"] <= window_end)
    ]
    return window_runs["analysis_run_id"].astype(str).tolist()


def _attention_reliability_label(
    *,
    selected_mentions: pd.DataFrame,
    future_measurement_run_ids: list[str],
    coverage_lookup: dict[str, float],
    fallback_mode: str,
) -> str:
    if future_measurement_run_ids:
        coverages = [coverage_lookup.get(run_id, 0.0) for run_id in future_measurement_run_ids]
        average_coverage = sum(coverages) / max(len(coverages), 1)
        if average_coverage >= 0.8:
            return "High"
        if average_coverage >= 0.5:
            return "Medium"
        return "Low"
    if fallback_mode == "corpus" and not selected_mentions.empty:
        if int(selected_mentions["source_name"].nunique()) >= 2 and int(selected_mentions["thread_id"].nunique()) >= 2:
            return "Medium"
    return "Low"


def _short_excerpt(text: str, *, limit: int = 180) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3].rstrip()}..."


def _coerce_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    raw_value = str(value).strip()
    if len(raw_value) == 10 and raw_value.count("-") == 2:
        try:
            return datetime.fromisoformat(raw_value).replace(tzinfo=UTC)
        except ValueError:
            return None
    try:
        return parse_created_time(raw_value)
    except Exception:
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "") or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def market_symbol_for_ticker(ticker: str) -> str:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return ""
    return MARKET_TICKER_ALIASES.get(normalized, normalized)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
