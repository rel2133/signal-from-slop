from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from signal_from_the_slop.database import (
    load_completed_mentions,
    load_market_prices,
    load_run_time_buckets,
    upsert_market_prices,
)
from signal_from_the_slop.signal_validation import ensure_market_data, market_symbol_for_ticker

try:  # scipy is optional; the app falls back to asymptotic approximations.
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover - depends on local optional packages
    scipy_stats = None


DEFAULT_CORRELATION_METHOD = "Spearman"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_RETURN_WINDOW = "+5 trading days"
PRICE_LOOKBACK_CALENDAR_DAYS = 20
PRICE_FORWARD_CALENDAR_DAYS = 55

RETURN_WINDOWS: list[dict[str, Any]] = [
    {"label": "Same day", "suffix": "same_day", "trading_days": 0},
    {"label": "+1 trading day", "suffix": "1d", "trading_days": 1},
    {"label": "+3 trading days", "suffix": "3d", "trading_days": 3},
    {"label": "+5 trading days", "suffix": "5d", "trading_days": 5},
    {"label": "+10 trading days", "suffix": "10d", "trading_days": 10},
    {"label": "+20 trading days", "suffix": "20d", "trading_days": 20},
    {"label": "+1 month", "suffix": "1m", "trading_days": 21},
]

SIGNAL_VARIABLES: dict[str, str] = {
    "Mentions": "mentions",
    "Mention share of voice": "mention_share_of_voice",
    "Mention acceleration": "mention_acceleration",
    "Percentage mention acceleration": "percentage_mention_acceleration",
    "Post count": "post_count",
    "Comment count": "comment_count",
    "Unique subreddit count": "unique_subreddit_count",
    "Unique author count": "unique_author_count",
    "Average hype score": "average_hype_score",
    "Average evidence score": "average_evidence_score",
    "Average specificity score": "average_specificity_score",
    "Average sentiment score": "average_sentiment_score",
    "Bullish mention count": "bullish_mention_count",
    "Bearish mention count": "bearish_mention_count",
    "Bull/bear ratio": "bull_bear_ratio",
    "Low mentions, high signal flag": "low_mentions_high_signal_flag",
    "Narrative shift score": "narrative_shift_score",
}

OUTCOME_GROUPS: dict[str, str] = {
    "Raw return": "raw_return",
    "Market-adjusted return": "abnormal_return",
    "Volume change": "volume_change",
    "Volatility": "volatility",
    "Absolute return": "absolute_return",
    "Drawdown": "drawdown",
    "Gap up/down": "gap_return",
    "Price direction": "price_direction",
}


def return_window_suffix(label: str) -> str:
    for window in RETURN_WINDOWS:
        if window["label"] == label:
            return str(window["suffix"])
    return "5d"


def outcome_column(outcome_group: str, return_window_label: str) -> str:
    prefix = OUTCOME_GROUPS.get(outcome_group, "abnormal_return")
    suffix = return_window_suffix(return_window_label)
    if prefix == "gap_return":
        suffix = "1d"
    return f"{prefix}_{suffix}"


def outcome_label(column_name: str) -> str:
    replacements = {
        "raw_return": "Raw return",
        "abnormal_return": "Abnormal return",
        "market_return": "Market return",
        "volume_change": "Volume change",
        "volatility": "Volatility",
        "absolute_return": "Absolute return",
        "drawdown": "Drawdown",
        "gap_return": "Gap up/down",
        "price_direction": "Price direction",
        "prior_return": "Prior return",
        "prior_volatility": "Prior volatility",
    }
    suffix_labels = {str(item["suffix"]): str(item["label"]) for item in RETURN_WINDOWS}
    for prefix, label in replacements.items():
        if column_name.startswith(f"{prefix}_"):
            suffix = column_name.removeprefix(f"{prefix}_")
            return f"{label} {suffix_labels.get(suffix, suffix)}"
    return column_name.replace("_", " ").title()


def build_correlation_dataset(
    *,
    db_path: Path,
    bucket_type: str,
    date_start: date | None = None,
    date_end: date | None = None,
    ticker_filter: Iterable[str] | None = None,
    min_mentions: int = 1,
    benchmark: str = DEFAULT_BENCHMARK,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    buckets = build_reddit_signal_buckets(
        db_path=db_path,
        bucket_type=bucket_type,
        date_start=date_start,
        date_end=date_end,
        ticker_filter=ticker_filter,
        min_mentions=min_mentions,
    )
    if buckets.empty:
        return buckets, {
            "bucket_rows": 0,
            "ticker_count": 0,
            "price_available_rows": 0,
            "missing_price_rows": 0,
            "benchmark": benchmark,
        }
    enriched = attach_price_outcomes(db_path=db_path, buckets=buckets, benchmark=benchmark)
    return enriched, dataset_diagnostics(enriched, benchmark=benchmark)


def build_reddit_signal_buckets(
    *,
    db_path: Path,
    bucket_type: str,
    date_start: date | None = None,
    date_end: date | None = None,
    ticker_filter: Iterable[str] | None = None,
    min_mentions: int = 1,
) -> pd.DataFrame:
    mentions = load_completed_mentions(db_path, data_mode="live")
    if mentions.empty:
        return pd.DataFrame()

    working = mentions.copy()
    working["ticker"] = working["ticker"].fillna("").astype(str).str.upper()
    working = working[(working["ticker"] != "") & (working["ticker"] != "UNKNOWN")].copy()
    if working.empty:
        return pd.DataFrame()

    allowed_tickers = {str(ticker).upper().strip() for ticker in ticker_filter or [] if str(ticker).strip()}
    if allowed_tickers:
        working = working[working["ticker"].isin(allowed_tickers)].copy()
        if working.empty:
            return pd.DataFrame()

    working["created_time_dt"] = pd.to_datetime(working["created_time"], errors="coerce", utc=True)
    working = working.dropna(subset=["created_time_dt"]).copy()
    if working.empty:
        return pd.DataFrame()

    working["bucket_start"] = working.apply(lambda row: _bucket_start(row, bucket_type), axis=1)
    working = working.dropna(subset=["bucket_start"]).copy()
    working["bucket_date"] = pd.to_datetime(working["bucket_start"], errors="coerce").dt.date
    if date_start is not None:
        working = working[working["bucket_date"] >= date_start].copy()
    if date_end is not None:
        working = working[working["bucket_date"] <= date_end].copy()
    if working.empty:
        return pd.DataFrame()

    total_mentions_by_bucket = working.groupby("bucket_start", dropna=False).size().to_dict()
    records: list[dict[str, Any]] = []
    group_columns = ["ticker", "bucket_start"]
    if bucket_type == "Per analysis run":
        group_columns.append("analysis_run_id")

    for group_key, group in working.groupby(group_columns, dropna=False):
        if len(group_columns) == 3:
            ticker, bucket_start, analysis_run_id = group_key
        else:
            ticker, bucket_start = group_key
            analysis_run_id = ""
        total_bucket_mentions = max(int(total_mentions_by_bucket.get(bucket_start, len(group)) or len(group)), 1)
        bullish = int((group["sentiment"].astype(str).str.lower() == "bullish").sum()) if "sentiment" in group else 0
        bearish = int((group["sentiment"].astype(str).str.lower() == "bearish").sum()) if "sentiment" in group else 0
        row = {
            "ticker": str(ticker).upper(),
            "analysis_run_id": str(analysis_run_id or ""),
            "bucket_type": bucket_type,
            "bucket_start": str(bucket_start),
            "bucket_date": pd.to_datetime(bucket_start).date().isoformat(),
            "company_name": _first_non_empty(group.get("company_name", pd.Series(dtype=str)), "Unknown"),
            "mentions": int(len(group)),
            "mention_share_of_voice": round((len(group) / total_bucket_mentions) * 100, 4),
            "post_count": _count_item_type(group, "post"),
            "comment_count": _count_item_type(group, "comment"),
            "unique_subreddit_count": _nunique(group, "subreddit"),
            "unique_author_count": _nunique(group, "author_hash"),
            "unique_thread_count": _nunique(group, "thread_id"),
            "average_hype_score": _mean(group, "hype_score"),
            "average_evidence_score": _mean(group, "evidence_quality_numeric"),
            "average_specificity_score": _mean(group, "claim_specificity_score"),
            "average_sentiment_score": _mean(group, "sentiment_numeric"),
            "average_alpha_signal_score": _mean(group, "alpha_signal_score"),
            "average_hype_adjusted_signal_score": _mean(group, "hype_adjusted_signal_score"),
            "bullish_mention_count": bullish,
            "bearish_mention_count": bearish,
            "bull_bear_ratio": round((bullish + 0.5) / (bearish + 0.5), 4),
            "low_mentions_high_signal_flag": int(bool(group.get("low_mentions_high_signal", pd.Series([False])).fillna(False).astype(bool).max())),
        }
        records.append(row)

    buckets = pd.DataFrame(records)
    if buckets.empty:
        return buckets
    buckets["bucket_start_dt"] = pd.to_datetime(buckets["bucket_start"], errors="coerce", utc=True)
    buckets = buckets.dropna(subset=["bucket_start_dt"]).sort_values(["ticker", "bucket_start_dt"]).reset_index(drop=True)
    buckets["mention_acceleration"] = buckets.groupby("ticker")["mentions"].diff().fillna(0)
    prior_mentions = buckets.groupby("ticker")["mentions"].shift(1)
    buckets["percentage_mention_acceleration"] = np.where(
        prior_mentions > 0,
        (buckets["mention_acceleration"] / prior_mentions) * 100,
        np.nan,
    )
    prior_sentiment = buckets.groupby("ticker")["average_sentiment_score"].shift(1)
    buckets["narrative_shift_score"] = (buckets["average_sentiment_score"] - prior_sentiment).abs()
    buckets["narrative_shift_score"] = buckets["narrative_shift_score"].fillna(0)
    buckets = buckets[buckets["mentions"] >= max(int(min_mentions), 1)].copy()
    return buckets.reset_index(drop=True)


def attach_price_outcomes(*, db_path: Path, buckets: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if buckets.empty:
        return buckets
    working = buckets.copy()
    benchmark = str(benchmark or "").strip().upper()
    use_benchmark = benchmark not in {"", "NO BENCHMARK", "NONE"}
    price_cache: dict[str, pd.DataFrame] = {}

    if use_benchmark:
        benchmark_market_ticker = market_symbol_for_ticker(benchmark)
        benchmark_prices = _prepare_prices(load_market_prices(db_path, ticker=benchmark_market_ticker))
    else:
        benchmark_market_ticker = ""
        benchmark_prices = pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in working.to_dict(orient="records"):
        ticker = str(row.get("ticker", "")).upper()
        market_ticker = market_symbol_for_ticker(ticker)
        prices = price_cache.setdefault(market_ticker, _prepare_prices(load_market_prices(db_path, ticker=market_ticker)))
        bucket_day = _coerce_date(row.get("bucket_date") or row.get("bucket_start"))
        enriched = dict(row)
        enriched["market_ticker"] = market_ticker
        enriched["benchmark"] = benchmark if use_benchmark else "No benchmark"
        enriched["benchmark_ticker"] = benchmark_market_ticker
        enriched["price_data_source"] = ""
        enriched["price_fetch_timestamp"] = ""
        enriched["price_available"] = False
        enriched["missing_price_reason"] = "missing bucket date"
        if bucket_day is not None:
            enriched.update(_price_outcome_row(
                ticker=ticker,
                prices=prices,
                benchmark_prices=benchmark_prices,
                bucket_day=bucket_day,
                use_benchmark=use_benchmark,
            ))
        records.append(enriched)
    return pd.DataFrame(records)


def refresh_correlation_price_cache(
    *,
    db_path: Path,
    buckets: pd.DataFrame,
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    if buckets.empty:
        return {"status": "idle", "tickers_checked": 0, "tickers_fetched": 0, "errors": []}
    working = buckets.copy()
    working["bucket_day"] = pd.to_datetime(
        working.get("bucket_date", working.get("bucket_start")),
        errors="coerce",
        utc=True,
    )
    working = working.dropna(subset=["bucket_day"]).copy()
    if working.empty:
        return {"status": "idle", "tickers_checked": 0, "tickers_fetched": 0, "errors": ["No valid bucket dates found."]}

    start_date = working["bucket_day"].min().date() - timedelta(days=PRICE_LOOKBACK_CALENDAR_DAYS)
    end_date = min(datetime.now(UTC).date(), working["bucket_day"].max().date() + timedelta(days=PRICE_FORWARD_CALENDAR_DAYS))
    tickers = sorted({str(ticker).upper() for ticker in working["ticker"].dropna().astype(str) if str(ticker).strip()})
    benchmark = str(benchmark or "").strip().upper()
    if benchmark and benchmark not in {"NO BENCHMARK", "NONE"}:
        tickers.append(benchmark)
    tickers = sorted({ticker for ticker in tickers if ticker})

    errors: list[str] = []
    fetched = 0
    for ticker in tickers:
        result = ensure_market_data(
            db_path=db_path,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if result.get("fetched"):
            fetched += 1
        error = str(result.get("error", "") or "").strip()
        if error:
            errors.append(f"{ticker}: {error}")
            _record_missing_bucket_prices(
                db_path=db_path,
                ticker=ticker,
                bucket_dates=working["bucket_day"].dt.date.unique().tolist(),
                reason=error,
            )
    return {
        "status": "partial" if errors else "ok",
        "tickers_checked": len(tickers),
        "tickers_fetched": fetched,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "errors": errors[:12],
    }


def refresh_price_cache_for_run(
    *,
    db_path: Path,
    analysis_run_id: str,
    benchmark: str = DEFAULT_BENCHMARK,
) -> dict[str, Any]:
    buckets = load_run_time_buckets(db_path, analysis_run_id)
    if buckets.empty:
        return {"status": "idle", "tickers_checked": 0, "tickers_fetched": 0, "errors": ["No ticker buckets found for run."]}
    prepared = buckets.rename(columns={"total_mentions": "mentions"}).copy()
    prepared["bucket_date"] = pd.to_datetime(prepared["bucket_start"], errors="coerce").dt.date.astype(str)
    return refresh_correlation_price_cache(db_path=db_path, buckets=prepared, benchmark=benchmark)


def compute_correlation_grid(
    frame: pd.DataFrame,
    *,
    signal_columns: list[str],
    outcome_columns: list[str],
    method: str,
    lag_window: str = "Selected",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in signal_columns:
        if signal not in frame.columns:
            continue
        for outcome in outcome_columns:
            if outcome not in frame.columns:
                continue
            stats = correlation_pair_stats(frame, signal, outcome, method=method)
            rows.append(
                {
                    "signal_variable": signal,
                    "signal_label": variable_label(signal),
                    "outcome_variable": outcome,
                    "outcome_label": outcome_label(outcome),
                    "lag_window": lag_window,
                    "correlation_method": method,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def compute_lead_lag_grid(
    frame: pd.DataFrame,
    *,
    signal_columns: list[str],
    outcome_group: str,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prefix = OUTCOME_GROUPS.get(outcome_group, "abnormal_return")
    for signal in signal_columns:
        if signal not in frame.columns:
            continue
        for window in RETURN_WINDOWS:
            suffix = str(window["suffix"])
            if prefix == "gap_return" and suffix != "1d":
                continue
            outcome = f"{prefix}_{suffix}"
            if outcome not in frame.columns:
                continue
            stats = correlation_pair_stats(frame, signal, outcome, method=method)
            rows.append(
                {
                    "signal_variable": signal,
                    "signal_label": variable_label(signal),
                    "outcome_variable": outcome,
                    "outcome_label": outcome_label(outcome),
                    "lag_window": str(window["label"]),
                    "correlation_method": method,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def compute_rolling_correlation(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    outcome_column_name: str,
    method: str,
    window_size: int,
) -> pd.DataFrame:
    if frame.empty or signal_column not in frame.columns or outcome_column_name not in frame.columns:
        return pd.DataFrame()
    working = frame[["bucket_start", "ticker", signal_column, outcome_column_name]].copy()
    working["bucket_dt"] = pd.to_datetime(working["bucket_start"], errors="coerce", utc=True)
    working[signal_column] = pd.to_numeric(working[signal_column], errors="coerce")
    working[outcome_column_name] = pd.to_numeric(working[outcome_column_name], errors="coerce")
    working = working.dropna(subset=["bucket_dt", signal_column, outcome_column_name]).sort_values("bucket_dt").reset_index(drop=True)
    if working.empty:
        return pd.DataFrame()
    min_periods = min(max(8, int(window_size * 0.5)), int(window_size))
    records: list[dict[str, Any]] = []
    for index in range(len(working)):
        start = max(0, index - int(window_size) + 1)
        window = working.iloc[start:index + 1]
        if len(window) < min_periods:
            continue
        stats = correlation_pair_stats(window, signal_column, outcome_column_name, method=method)
        records.append(
            {
                "date": working.iloc[index]["bucket_dt"].date().isoformat(),
                "correlation": stats["correlation_value"],
                "sample_size": stats["sample_size"],
                "reliability_label": stats["reliability_label"],
            }
        )
    return pd.DataFrame(records)


def run_regression(
    frame: pd.DataFrame,
    *,
    dependent_column: str,
    predictor_columns: list[str],
) -> dict[str, Any]:
    available_predictors = [column for column in predictor_columns if column in frame.columns]
    missing_observations = len(frame)
    if dependent_column not in frame.columns or not available_predictors:
        return {"status": "unavailable", "reason": "Select a dependent variable and at least one predictor.", "rows": pd.DataFrame()}

    working = frame[[dependent_column, *available_predictors, "bucket_start", "ticker"]].copy()
    for column in [dependent_column, *available_predictors]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=[dependent_column, *available_predictors]).copy()
    missing_observations -= len(working)
    n = len(working)
    predictor_count = len(available_predictors)
    if n <= predictor_count + 2:
        return {
            "status": "unavailable",
            "reason": "Not enough complete observations for the selected model.",
            "observations": n,
            "missing_observations": missing_observations,
            "rows": pd.DataFrame(),
        }

    y = working[dependent_column].to_numpy(dtype=float)
    x_values = working[available_predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(n), x_values])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predicted = x @ coefficients
    residuals = y - predicted
    sse = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    df_resid = max(n - predictor_count - 1, 1)
    mse = sse / df_resid
    covariance = mse * np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))

    rows: list[dict[str, Any]] = []
    names = ["Intercept", *available_predictors]
    for name, estimate, standard_error in zip(names, coefficients, standard_errors):
        t_value = float(estimate / standard_error) if standard_error not in (0, None) else np.nan
        p_value = _two_sided_normal_p_value(t_value) if not pd.isna(t_value) else np.nan
        rows.append(
            {
                "predictor": name,
                "predictor_label": "Intercept" if name == "Intercept" else variable_label(name),
                "estimate": float(estimate),
                "standard_error": float(standard_error),
                "t": t_value,
                "p": p_value,
                "confidence_interval_low": float(estimate - 1.96 * standard_error),
                "confidence_interval_high": float(estimate + 1.96 * standard_error),
            }
        )

    r_squared = 1 - (sse / tss) if tss else np.nan
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / df_resid) if not pd.isna(r_squared) else np.nan
    explained = max(tss - sse, 0.0)
    f_statistic = (explained / max(predictor_count, 1)) / mse if mse else np.nan
    model_p = _two_sided_normal_p_value(math.sqrt(f_statistic)) if not pd.isna(f_statistic) else np.nan
    prediction_frame = working[["bucket_start", "ticker", dependent_column]].copy()
    prediction_frame["predicted"] = predicted
    prediction_frame["residual"] = residuals

    return {
        "status": "ok",
        "observations": n,
        "missing_observations": missing_observations,
        "r_squared": float(r_squared) if not pd.isna(r_squared) else np.nan,
        "adjusted_r_squared": float(adjusted_r_squared) if not pd.isna(adjusted_r_squared) else np.nan,
        "f_statistic": float(f_statistic) if not pd.isna(f_statistic) else np.nan,
        "model_p_value": float(model_p) if not pd.isna(model_p) else np.nan,
        "residual_standard_error": float(math.sqrt(mse)) if mse >= 0 else np.nan,
        "rows": pd.DataFrame(rows),
        "predictions": prediction_frame,
    }


def correlation_pair_stats(frame: pd.DataFrame, x_column: str, y_column: str, *, method: str) -> dict[str, Any]:
    pair = frame[[x_column, y_column]].copy()
    pair[x_column] = pd.to_numeric(pair[x_column], errors="coerce")
    pair[y_column] = pd.to_numeric(pair[y_column], errors="coerce")
    missing_count = int(pair.isna().any(axis=1).sum())
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(pair))
    if n < 2:
        return {
            "correlation_value": np.nan,
            "p_value": np.nan,
            "sample_size": n,
            "confidence_interval_low": np.nan,
            "confidence_interval_high": np.nan,
            "missing_count": missing_count,
            "reliability_label": reliability_label(n),
            "warning_flags": ["not enough complete observations"],
        }

    method_key = method.strip().lower()
    x = pair[x_column]
    y = pair[y_column]
    correlation_value = _correlation_value(x, y, method_key)
    p_value = _correlation_p_value(x, y, method_key, correlation_value)
    ci_low, ci_high = _correlation_confidence_interval(correlation_value, n)
    warnings = warning_flags(pair, x_column, y_column, correlation_value, missing_count)
    return {
        "correlation_value": correlation_value,
        "p_value": p_value,
        "sample_size": n,
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high,
        "missing_count": missing_count,
        "reliability_label": reliability_label(n),
        "warning_flags": warnings,
    }


def reliability_label(sample_size: int) -> str:
    if sample_size < 20:
        return "Unreliable"
    if sample_size < 50:
        return "Weak evidence"
    if sample_size < 100:
        return "Moderate evidence"
    return "Stronger evidence"


def warning_flags(
    pair: pd.DataFrame,
    x_column: str,
    y_column: str,
    correlation_value: float,
    missing_count: int,
) -> list[str]:
    warnings: list[str] = []
    n = len(pair)
    if n < 20:
        warnings.append("sample size too small")
    if n and missing_count / max(n + missing_count, 1) > 0.25:
        warnings.append("many missing price values")
    if n >= 8:
        for column in (x_column, y_column):
            unique_ratio = pair[column].nunique(dropna=True) / max(n, 1)
            if unique_ratio <= 0.12:
                warnings.append(f"{variable_label(column)} has many tied values")
                break
    if n >= 12 and not pd.isna(correlation_value):
        if _outlier_sensitive(pair, x_column, y_column, correlation_value):
            warnings.append("correlation may be driven by one extreme outlier")
    if "same_day" in y_column:
        warnings.append("same-period relationship may be reactive rather than predictive")
    if "abnormal_return" in y_column and abs(float(correlation_value or 0)) < 0.05:
        warnings.append("market-adjusted relationship is very weak")
    warnings.append("multiple comparisons can create false positives")
    return warnings


def dataset_diagnostics(frame: pd.DataFrame, *, benchmark: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "bucket_rows": 0,
            "ticker_count": 0,
            "price_available_rows": 0,
            "missing_price_rows": 0,
            "data_completeness": 0.0,
            "benchmark": benchmark,
        }
    available = frame.get("price_available", pd.Series([False] * len(frame))).fillna(False).astype(bool)
    return {
        "bucket_rows": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame else 0,
        "price_available_rows": int(available.sum()),
        "missing_price_rows": int((~available).sum()),
        "data_completeness": round(float(available.mean()) if len(available) else 0.0, 4),
        "benchmark": benchmark,
        "date_start": str(frame["bucket_date"].min()) if "bucket_date" in frame and not frame.empty else "",
        "date_end": str(frame["bucket_date"].max()) if "bucket_date" in frame and not frame.empty else "",
    }


def strongest_relationships(results: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    if results.empty or "correlation_value" not in results:
        return {"positive": None, "negative": None, "absolute": None}
    working = results.dropna(subset=["correlation_value"]).copy()
    if working.empty:
        return {"positive": None, "negative": None, "absolute": None}
    positive = working.sort_values("correlation_value", ascending=False).iloc[0].to_dict()
    negative = working.sort_values("correlation_value", ascending=True).iloc[0].to_dict()
    absolute = working.assign(abs_corr=working["correlation_value"].abs()).sort_values("abs_corr", ascending=False).iloc[0].to_dict()
    return {"positive": positive, "negative": negative, "absolute": absolute}


def variable_label(column_name: str) -> str:
    for label, column in SIGNAL_VARIABLES.items():
        if column == column_name:
            return label
    return outcome_label(column_name)


def relationship_sentence(row: dict[str, Any] | None) -> str:
    if not row:
        return "No complete correlation result is available yet."
    value = _coerce_float(row.get("correlation_value"), np.nan)
    if pd.isna(value):
        return "No complete correlation result is available yet."
    direction = "positive" if value > 0 else "negative" if value < 0 else "flat"
    return (
        f"{variable_label(str(row.get('signal_variable', '')))} vs "
        f"{outcome_label(str(row.get('outcome_variable', '')))} is {direction} "
        f"(r={value:.2f}, n={int(row.get('sample_size', 0) or 0)})."
    )


def abnormal_return_interpretation(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "No raw versus abnormal comparison is available for the selected filters."
    working = comparison.copy()
    if {"raw_correlation", "abnormal_correlation"}.issubset(working.columns):
        working["difference"] = working["raw_correlation"].abs() - working["abnormal_correlation"].abs()
        strongest = working.reindex(working["difference"].abs().sort_values(ascending=False).index).iloc[0]
        raw = _coerce_float(strongest.get("raw_correlation"), np.nan)
        abnormal = _coerce_float(strongest.get("abnormal_correlation"), np.nan)
        if pd.isna(raw) or pd.isna(abnormal):
            return "Some comparison cells still lack enough complete price data."
        if abs(raw) > abs(abnormal) + 0.08:
            return "The raw return relationship is stronger than the abnormal return relationship, suggesting broader market movement may explain part of the result."
        if abs(abnormal) >= abs(raw) - 0.03 and abs(abnormal) >= 0.1:
            return "The abnormal return relationship remains visible after market adjustment, so the pattern may be more ticker-specific."
    return "No meaningful abnormal return relationship was found under the current filters."


def rolling_interpretation(rolling: pd.DataFrame) -> str:
    if rolling.empty or "correlation" not in rolling:
        return "Rolling correlation needs more complete observations."
    complete = rolling.dropna(subset=["correlation"]).copy()
    if len(complete) < 3:
        return "Rolling correlation is based on a small sample and may not be reliable."
    first = float(complete.iloc[0]["correlation"])
    latest = float(complete.iloc[-1]["correlation"])
    std = float(complete["correlation"].std()) if len(complete) > 1 else 0.0
    if std > 0.35:
        return "Correlation is unstable over time, so this signal should be treated cautiously."
    if latest > first + 0.12:
        return "The relationship has strengthened over the selected rolling window."
    if latest < first - 0.12:
        return "The relationship was stronger earlier and has recently weakened."
    return "The rolling relationship is broadly stable under the selected window."


def _price_outcome_row(
    *,
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    bucket_day: date,
    use_benchmark: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "price_available": False,
        "missing_price_reason": "no cached price data",
    }
    _set_all_outcomes_missing(result)
    start_index = _first_index_on_or_after(prices, bucket_day)
    if start_index is None:
        return result
    start = prices.iloc[start_index]
    start_close = _coerce_float(start.get("reference_close"), np.nan)
    start_open = _coerce_float(start.get("open"), np.nan)
    if pd.isna(start_close) or start_close == 0:
        result["missing_price_reason"] = "cached row has no close price"
        return result

    benchmark_start_index = _first_index_on_or_after(benchmark_prices, bucket_day) if use_benchmark else None
    benchmark_start = benchmark_prices.iloc[benchmark_start_index] if benchmark_start_index is not None else None
    benchmark_start_close = _coerce_float(benchmark_start.get("reference_close"), np.nan) if benchmark_start is not None else np.nan

    result.update(
        {
            "price_available": True,
            "missing_price_reason": "",
            "price_date": str(start.get("date", "")),
            "open_price": _none_if_nan(start_open),
            "high_price": _none_if_nan(_coerce_float(start.get("high"), np.nan)),
            "low_price": _none_if_nan(_coerce_float(start.get("low"), np.nan)),
            "close_price": _none_if_nan(start_close),
            "adjusted_close_price": _none_if_nan(_coerce_float(start.get("adjusted_close"), np.nan)),
            "volume": _none_if_nan(_coerce_float(start.get("volume"), np.nan)),
            "price_data_source": str(start.get("data_source", "")),
            "price_fetch_timestamp": str(start.get("fetched_at", "")),
            "benchmark_price": _none_if_nan(benchmark_start_close),
        }
    )

    same_day_return = None
    if not pd.isna(start_open) and start_open:
        same_day_return = (start_close / start_open) - 1
    benchmark_same_day_return = None
    if benchmark_start is not None:
        benchmark_open = _coerce_float(benchmark_start.get("open"), np.nan)
        if not pd.isna(benchmark_open) and benchmark_open:
            benchmark_same_day_return = (benchmark_start_close / benchmark_open) - 1

    result["raw_return_same_day"] = _rounded(same_day_return)
    result["market_return_same_day"] = _rounded(benchmark_same_day_return)
    result["abnormal_return_same_day"] = _difference(result["raw_return_same_day"], result["market_return_same_day"]) if use_benchmark else None
    result["absolute_return_same_day"] = abs(result["raw_return_same_day"]) if result["raw_return_same_day"] is not None else None
    result["price_direction_same_day"] = _direction(result["raw_return_same_day"])
    result["volume_change_same_day"] = 0.0
    result["volatility_same_day"] = 0.0
    result["drawdown_same_day"] = _rounded(min(0.0, same_day_return if same_day_return is not None else 0.0))

    prior_index = max(start_index - 5, 0)
    prior_close = _coerce_float(prices.iloc[prior_index].get("reference_close"), np.nan) if not prices.empty else np.nan
    result["prior_return_5d"] = _rounded((start_close / prior_close) - 1) if not pd.isna(prior_close) and prior_close else None
    result["prior_volatility_10d"] = _volatility(prices, max(start_index - 10, 0), start_index)

    for window in RETURN_WINDOWS:
        suffix = str(window["suffix"])
        trading_days = int(window["trading_days"])
        if suffix == "same_day":
            continue
        future_index = start_index + trading_days
        if future_index >= len(prices):
            _set_window_missing(result, suffix)
            continue
        future = prices.iloc[future_index]
        future_close = _coerce_float(future.get("reference_close"), np.nan)
        if pd.isna(future_close):
            _set_window_missing(result, suffix)
            continue
        raw_return = (future_close / start_close) - 1
        result[f"raw_return_{suffix}"] = _rounded(raw_return)
        result[f"absolute_return_{suffix}"] = _rounded(abs(raw_return))
        result[f"price_direction_{suffix}"] = _direction(raw_return)
        result[f"volume_change_{suffix}"] = _volume_change(prices, start_index, future_index)
        result[f"volatility_{suffix}"] = _volatility(prices, start_index, future_index)
        result[f"drawdown_{suffix}"] = _drawdown(prices, start_index, future_index, start_close)
        benchmark_return = None
        if use_benchmark and benchmark_start_index is not None and benchmark_start_index + trading_days < len(benchmark_prices):
            benchmark_future = benchmark_prices.iloc[benchmark_start_index + trading_days]
            benchmark_future_close = _coerce_float(benchmark_future.get("reference_close"), np.nan)
            if not pd.isna(benchmark_future_close) and not pd.isna(benchmark_start_close) and benchmark_start_close:
                benchmark_return = (benchmark_future_close / benchmark_start_close) - 1
        result[f"market_return_{suffix}"] = _rounded(benchmark_return)
        result[f"abnormal_return_{suffix}"] = _difference(result[f"raw_return_{suffix}"], result[f"market_return_{suffix}"]) if use_benchmark else None

    next_index = start_index + 1
    if next_index < len(prices):
        next_open = _coerce_float(prices.iloc[next_index].get("open"), np.nan)
        result["gap_return_1d"] = _rounded((next_open / start_close) - 1) if not pd.isna(next_open) and start_close else None
    else:
        result["gap_return_1d"] = None
    return result


def _set_window_missing(result: dict[str, Any], suffix: str) -> None:
    for prefix in ("raw_return", "absolute_return", "price_direction", "volume_change", "volatility", "drawdown", "market_return", "abnormal_return"):
        result[f"{prefix}_{suffix}"] = None


def _set_all_outcomes_missing(result: dict[str, Any]) -> None:
    for window in RETURN_WINDOWS:
        _set_window_missing(result, str(window["suffix"]))
    result["gap_return_1d"] = None
    result["prior_return_5d"] = None
    result["prior_volatility_10d"] = None


def _bucket_start(row: pd.Series, bucket_type: str) -> str | None:
    created = row.get("created_time_dt")
    if pd.isna(created):
        return None
    created_dt = created.to_pydatetime()
    if bucket_type == "Weekly":
        start = created_dt.date() - timedelta(days=created_dt.weekday())
        return start.isoformat()
    if bucket_type == "Monthly":
        return created_dt.date().replace(day=1).isoformat()
    if bucket_type == "Per analysis run":
        run_completed = pd.to_datetime(row.get("run_completed_at") or row.get("run_started_at"), errors="coerce", utc=True)
        if not pd.isna(run_completed):
            return run_completed.date().isoformat()
    return created_dt.date().isoformat()


def _prepare_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    if "price_available" in working.columns:
        working = working[working["price_available"].fillna(True).astype(bool)].copy()
    working["date_dt"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working = working.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)
    working["reference_close"] = pd.to_numeric(working.get("adjusted_close"), errors="coerce")
    close = pd.to_numeric(working.get("close"), errors="coerce")
    working["reference_close"] = working["reference_close"].where(working["reference_close"].notna(), close)
    return working


def _first_index_on_or_after(prices: pd.DataFrame, target_day: date) -> int | None:
    if prices.empty or "date_dt" not in prices:
        return None
    matches = prices.index[prices["date_dt"].dt.date >= target_day].tolist()
    if not matches:
        return None
    return int(matches[0])


def _record_missing_bucket_prices(
    *,
    db_path: Path,
    ticker: str,
    bucket_dates: Iterable[date],
    reason: str,
) -> None:
    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "ticker": market_symbol_for_ticker(ticker),
            "date": bucket_day.isoformat(),
            "data_source": "yfinance",
            "fetched_at": now,
            "price_available": False,
            "missing_reason": reason[:240],
        }
        for bucket_day in bucket_dates
    ]
    upsert_market_prices(db_path, rows)


def _correlation_value(x: pd.Series, y: pd.Series, method_key: str) -> float:
    if scipy_stats is not None:
        try:
            if method_key == "pearson":
                return float(scipy_stats.pearsonr(x, y).statistic)
            if method_key == "kendall":
                return float(scipy_stats.kendalltau(x, y, nan_policy="omit").statistic)
            return float(scipy_stats.spearmanr(x, y, nan_policy="omit").statistic)
        except Exception:
            pass
    if method_key == "kendall":
        return _kendall_tau_b(x, y)
    if method_key == "spearman":
        return _pearson_correlation(x.rank(method="average"), y.rank(method="average"))
    return _pearson_correlation(x, y)


def _correlation_p_value(x: pd.Series, y: pd.Series, method_key: str, correlation_value: float) -> float:
    if pd.isna(correlation_value):
        return np.nan
    if scipy_stats is not None:
        try:
            if method_key == "pearson":
                return float(scipy_stats.pearsonr(x, y).pvalue)
            if method_key == "kendall":
                return float(scipy_stats.kendalltau(x, y, nan_policy="omit").pvalue)
            return float(scipy_stats.spearmanr(x, y, nan_policy="omit").pvalue)
        except Exception:
            pass
    n = len(x)
    if n < 3 or abs(correlation_value) >= 1:
        return np.nan
    if method_key == "kendall":
        z_value = correlation_value * math.sqrt((9 * n * (n - 1)) / (2 * (2 * n + 5)))
        return _two_sided_normal_p_value(z_value)
    t_value = correlation_value * math.sqrt((n - 2) / max(1 - correlation_value ** 2, 1e-12))
    return _two_sided_normal_p_value(t_value)


def _correlation_confidence_interval(correlation_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or pd.isna(correlation_value) or abs(correlation_value) >= 1:
        return np.nan, np.nan
    z_value = math.atanh(float(correlation_value))
    margin = 1.96 / math.sqrt(n - 3)
    return math.tanh(z_value - margin), math.tanh(z_value + margin)


def _two_sided_normal_p_value(test_statistic: float) -> float:
    if pd.isna(test_statistic):
        return np.nan
    return float(math.erfc(abs(float(test_statistic)) / math.sqrt(2)))


def _outlier_sensitive(pair: pd.DataFrame, x_column: str, y_column: str, original_correlation: float) -> bool:
    x_z = _z_scores(pair[x_column])
    y_z = _z_scores(pair[y_column])
    extreme = (x_z.abs() > 3.5) | (y_z.abs() > 3.5)
    if not extreme.any() or len(pair) - int(extreme.sum()) < 8:
        return False
    trimmed = pair[~extreme].copy()
    trimmed_correlation = _pearson_correlation(
        trimmed[x_column].rank(method="average"),
        trimmed[y_column].rank(method="average"),
    )
    if pd.isna(trimmed_correlation):
        return False
    return abs(float(original_correlation) - float(trimmed_correlation)) >= 0.25


def _pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x_values) | np.isnan(y_values))
    x_values = x_values[mask]
    y_values = y_values[mask]
    if len(x_values) < 2:
        return np.nan
    x_std = float(np.std(x_values))
    y_std = float(np.std(y_values))
    if x_std == 0 or y_std == 0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x_values) | np.isnan(y_values))
    x_values = x_values[mask]
    y_values = y_values[mask]
    n = len(x_values)
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for left in range(n - 1):
        dx = x_values[left + 1:] - x_values[left]
        dy = y_values[left + 1:] - y_values[left]
        product = dx * dy
        concordant += int((product > 0).sum())
        discordant += int((product < 0).sum())
        ties_x += int(((dx == 0) & (dy != 0)).sum())
        ties_y += int(((dy == 0) & (dx != 0)).sum())
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denominator == 0:
        return np.nan
    return float((concordant - discordant) / denominator)


def _z_scores(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std()
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (numeric - numeric.mean()) / std


def _volume_change(prices: pd.DataFrame, start_index: int, future_index: int) -> float | None:
    start_volume = _coerce_float(prices.iloc[start_index].get("volume"), np.nan)
    future_volume = _coerce_float(prices.iloc[future_index].get("volume"), np.nan)
    if pd.isna(start_volume) or start_volume == 0 or pd.isna(future_volume):
        return None
    return _rounded((future_volume / start_volume) - 1)


def _volatility(prices: pd.DataFrame, start_index: int, future_index: int) -> float | None:
    window = prices.iloc[start_index:future_index + 1].copy()
    if len(window) < 2:
        return None
    returns = pd.to_numeric(window["reference_close"], errors="coerce").pct_change().dropna()
    if returns.empty:
        return None
    return _rounded(float(returns.std()))


def _drawdown(prices: pd.DataFrame, start_index: int, future_index: int, start_close: float) -> float | None:
    if not start_close:
        return None
    window = prices.iloc[start_index:future_index + 1].copy()
    closes = pd.to_numeric(window["reference_close"], errors="coerce").dropna()
    if closes.empty:
        return None
    return _rounded((float(closes.min()) / start_close) - 1)


def _difference(left: Any, right: Any) -> float | None:
    left_value = _coerce_float(left, np.nan)
    right_value = _coerce_float(right, np.nan)
    if pd.isna(left_value) or pd.isna(right_value):
        return None
    return _rounded(left_value - right_value)


def _direction(value: Any) -> int | None:
    numeric = _coerce_float(value, np.nan)
    if pd.isna(numeric):
        return None
    return int(numeric > 0)


def _rounded(value: Any, digits: int = 6) -> float | None:
    numeric = _coerce_float(value, np.nan)
    if pd.isna(numeric):
        return None
    return round(float(numeric), digits)


def _none_if_nan(value: Any) -> float | None:
    numeric = _coerce_float(value, np.nan)
    if pd.isna(numeric):
        return None
    return float(numeric)


def _coerce_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return round(float(values.mean()), 4)


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].dropna().astype(str).replace("", np.nan).dropna().nunique())


def _count_item_type(frame: pd.DataFrame, item_type: str) -> int:
    if "item_type" not in frame:
        return 0
    return int(frame["item_type"].fillna("").astype(str).str.lower().eq(item_type).sum())


def _first_non_empty(series: pd.Series, default: str) -> str:
    for value in series.dropna().astype(str).tolist():
        if value.strip():
            return value
    return default
