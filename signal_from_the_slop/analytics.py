from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from signal_from_the_slop.scoring import compute_alpha_signal_score


FINANCIAL_TERMS = (
    "revenue",
    "margin",
    "margins",
    "cash flow",
    "free cash flow",
    "guidance",
    "earnings",
    "contract",
    "contracts",
    "customer",
    "customers",
    "valuation",
    "capex",
    "tam",
    "filing",
    "10-k",
    "10-q",
    "gross margin",
    "ebitda",
)
MEGA_CAP_TICKERS = {"TSLA", "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL"}
LINK_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\d")
REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)\b(?:\W+\1\b)+", re.IGNORECASE)


def author_hash(author: str) -> str:
    return hashlib.sha1((author or "[deleted]").encode("utf-8")).hexdigest()[:12]


def coerce_claim_specificity_score(value: Any, text: str) -> int:
    if value not in (None, ""):
        try:
            return int(max(min(float(value), 10), 0))
        except (TypeError, ValueError):
            pass
    return derive_claim_specificity_score(text)


def coerce_repetition_score(value: Any, text: str) -> int:
    if value not in (None, ""):
        try:
            return int(max(min(float(value), 10), 0))
        except (TypeError, ValueError):
            pass
    return derive_repetition_score(text)


def derive_claim_specificity_score(text: str) -> int:
    lowered = (text or "").lower()
    term_hits = sum(term in lowered for term in FINANCIAL_TERMS)
    numbers_bonus = 2 if NUMBER_PATTERN.search(lowered) else 0
    link_bonus = 1 if LINK_PATTERN.search(lowered) else 0
    score = term_hits + numbers_bonus + link_bonus
    return int(max(min(score, 10), 0))


def derive_repetition_score(text: str) -> int:
    lowered = (text or "").lower()
    repeated_words = len(REPEATED_WORD_PATTERN.findall(lowered))
    repeated_phrases = sum(
        phrase in lowered
        for phrase in ("to the moon", "easy money", "next tesla", "guaranteed", "can't lose", "obviously")
    )
    all_caps_words = sum(word.isupper() and len(word) > 3 for word in text.split())
    score = repeated_words * 2 + repeated_phrases * 3 + min(all_caps_words, 3)
    return int(max(min(score, 10), 0))


def compute_hype_adjusted_signal_score(alpha_signal_score: float, hype_score: int, repetition_score: int) -> float:
    adjusted = alpha_signal_score - (hype_score * 4.5) - (repetition_score * 1.5)
    return round(max(min(adjusted, 100), 0), 2)


def build_classification_record(
    item: dict[str, Any],
    classification_payload: dict[str, Any],
    *,
    classifier_mode: str,
    model_name: str,
    deeper_analysis_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = " ".join([item.get("thread_title", ""), item.get("body_text", "")]).strip()
    ticker_company_pairs = normalize_ticker_company_pairs(
        classification_payload.get("tickers") or [],
        classification_payload.get("company_names") or [],
    )
    claim_specificity_score = coerce_claim_specificity_score(
        classification_payload.get("claim_specificity_score"),
        text,
    )
    repetition_score = coerce_repetition_score(
        classification_payload.get("repetition_score"),
        text,
    )
    alpha_signal_score = compute_alpha_signal_score(classification_payload, item)
    hype_adjusted_signal_score = compute_hype_adjusted_signal_score(
        alpha_signal_score,
        int(classification_payload.get("hype_score", 0)),
        repetition_score,
    )
    claims_to_verify = classification_payload.get("claims_to_verify") or derive_claims_to_verify(text)
    if isinstance(claims_to_verify, str):
        claims_to_verify = [claims_to_verify]

    return {
        "item_id": item["item_id"],
        "sentiment": classification_payload.get("sentiment", "irrelevant"),
        "confidence": float(classification_payload.get("confidence", 0.0)),
        "tickers": [ticker for ticker, _ in ticker_company_pairs],
        "company_names": [company_name for _, company_name in ticker_company_pairs],
        "summary": str(classification_payload.get("summary", "")).strip(),
        "bull_case": str(classification_payload.get("bull_case", "")).strip(),
        "bear_case": str(classification_payload.get("bear_case", "")).strip(),
        "evidence_quality": str(classification_payload.get("evidence_quality", "low")).strip().lower(),
        "depth_score": int(classification_payload.get("depth_score", 0)),
        "hype_score": int(classification_payload.get("hype_score", 0)),
        "claim_specificity_score": claim_specificity_score,
        "repetition_score": repetition_score,
        "needs_deeper_analysis": bool(classification_payload.get("needs_deeper_analysis", False)),
        "red_flags": classification_payload.get("red_flags") or [],
        "claims_to_verify": claims_to_verify,
        "alpha_signal_score": alpha_signal_score,
        "hype_adjusted_signal_score": hype_adjusted_signal_score,
        "model_name": model_name,
        "classifier_mode": classifier_mode,
        "deeper_analysis_json": deeper_analysis_json or {},
    }


def normalize_ticker_company_pairs(tickers: list[Any], company_names: list[Any]) -> list[tuple[str, str]]:
    pairs: dict[str, str] = {}
    for index, raw_ticker in enumerate(tickers):
        ticker = str(raw_ticker or "").strip().upper()
        if not ticker:
            continue
        company_name = str(company_names[index]).strip() if index < len(company_names) else "Unknown"
        company_name = company_name or "Unknown"
        if ticker not in pairs:
            pairs[ticker] = company_name
        elif pairs[ticker] == "Unknown" and company_name != "Unknown":
            pairs[ticker] = company_name
    return list(pairs.items())


def build_mention_records(item: dict[str, Any], classification_record: dict[str, Any]) -> list[dict[str, Any]]:
    ticker_company_pairs = normalize_ticker_company_pairs(
        classification_record["tickers"],
        classification_record.get("company_names") or [],
    )
    if not ticker_company_pairs:
        return []

    created = parse_created_time(item["created_time"])
    created_date = created.date().isoformat()
    time_bucket = week_bucket_start(created).date().isoformat()
    text = item.get("body_text", "")
    text_length = len(text)
    word_count = len(text.split())
    contains_link = bool(LINK_PATTERN.search(text))
    contains_numbers = bool(NUMBER_PATTERN.search(text))
    contains_financial_terms = any(term in text.lower() for term in FINANCIAL_TERMS)

    records = []
    for ticker, company_name in ticker_company_pairs:
        records.append(
            {
                "item_id": item["item_id"],
                "source_id": item.get("source_id"),
                "source_type": item.get("source_type", ""),
                "source_name": item.get("source_name", ""),
                "subreddit": item.get("subreddit", ""),
                "thread_id": item.get("thread_id", ""),
                "thread_title": item.get("thread_title", ""),
                "comment_id": item.get("comment_id"),
                "parent_id": item.get("parent_id"),
                "author_hash": item.get("author_hash", author_hash(item.get("author", ""))),
                "created_time": item["created_time"],
                "created_date": created_date,
                "time_bucket": time_bucket,
                "item_type": item.get("item_type", "post"),
                "ticker": ticker,
                "company_name": company_name or "Unknown",
                "sentiment": classification_record["sentiment"],
                "sentiment_numeric": sentiment_numeric(classification_record["sentiment"]),
                "confidence": classification_record["confidence"],
                "alpha_signal_score": classification_record["alpha_signal_score"],
                "hype_adjusted_signal_score": classification_record["hype_adjusted_signal_score"],
                "depth_score": classification_record["depth_score"],
                "hype_score": classification_record["hype_score"],
                "evidence_quality": classification_record["evidence_quality"],
                "evidence_quality_numeric": evidence_quality_numeric(classification_record["evidence_quality"]),
                "claim_specificity_score": classification_record["claim_specificity_score"],
                "source_diversity_score": 0.0,
                "controversy_score": 0.0,
                "repetition_score": classification_record["repetition_score"],
                "reddit_score": int(item.get("score", 0)),
                "comment_depth": int(item.get("comment_depth", 0)),
                "text_length": text_length,
                "word_count": word_count,
                "contains_link": contains_link,
                "contains_numbers": contains_numbers,
                "contains_financial_terms": contains_financial_terms,
                "needs_deeper_analysis": classification_record["needs_deeper_analysis"],
                "low_mentions_high_signal": False,
                "new_ticker_detected": False,
                "red_flag_count": len(classification_record["red_flags"]),
                "claims_to_verify": classification_record["claims_to_verify"],
                "summary": classification_record["summary"],
                "bull_case": classification_record["bull_case"],
                "bear_case": classification_record["bear_case"],
                "red_flags": classification_record["red_flags"],
                "deeper_analysis_json": classification_record["deeper_analysis_json"],
                "body_text": text,
                "permalink": item.get("permalink", ""),
            }
        )
    return records


def build_ticker_summaries(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (ticker, company_name), group in long_df.groupby(["ticker", "company_name"], dropna=False):
        group = group.sort_values("created_time")
        bucket_frame = _bucket_stats(group)
        current_bucket = bucket_frame.iloc[-1]
        previous_bucket = bucket_frame.iloc[-2] if len(bucket_frame) > 1 else None

        bullish_mentions = int((group["sentiment"] == "bullish").sum())
        bearish_mentions = int((group["sentiment"] == "bearish").sum())
        neutral_mentions = int((group["sentiment"] == "neutral").sum())
        total_mentions = int(len(group))

        bullish_ratio = bullish_mentions / total_mentions if total_mentions else 0.0
        bearish_ratio = bearish_mentions / total_mentions if total_mentions else 0.0
        net_sentiment_score = group["sentiment_numeric"].dropna().mean() if group["sentiment_numeric"].notna().any() else None
        source_diversity_score = compute_source_diversity_score(group)
        controversy_score = compute_controversy_score(bullish_mentions, bearish_mentions)
        average_alpha = round(float(group["alpha_signal_score"].mean()), 2)
        average_hype = round(float(group["hype_score"].mean()), 2)
        average_claim_specificity = round(float(group["claim_specificity_score"].mean()), 2)
        average_evidence_numeric = round(float(group["evidence_quality_numeric"].mean()), 2)
        average_repetition = round(float(group["repetition_score"].mean()), 2)

        mentions_this_period = int(current_bucket["total_mentions"])
        mentions_previous_period = int(previous_bucket["total_mentions"]) if previous_bucket is not None else 0
        mention_change = mentions_this_period - mentions_previous_period
        mention_growth_rate = safe_growth_rate(mentions_this_period, mentions_previous_period)

        bullish_this = int(current_bucket["bullish_mentions"])
        bullish_previous = int(previous_bucket["bullish_mentions"]) if previous_bucket is not None else 0
        bullish_change = bullish_this - bullish_previous
        bullish_growth_rate = safe_growth_rate(bullish_this, bullish_previous)

        unique_authors_this = int(current_bucket["unique_authors"])
        unique_authors_previous = int(previous_bucket["unique_authors"]) if previous_bucket is not None else 0
        unique_threads_this = int(current_bucket["unique_threads"])
        unique_threads_previous = int(previous_bucket["unique_threads"]) if previous_bucket is not None else 0
        unique_author_growth = unique_authors_this - unique_authors_previous
        unique_thread_growth = unique_threads_this - unique_threads_previous
        rolling_mention_average = round(float(bucket_frame["total_mentions"].tail(3).mean()), 2)

        new_ticker_detected = mentions_this_period > 0 and mentions_previous_period == 0
        low_mentions_high_signal = (
            total_mentions < 10
            and average_alpha >= 75
            and average_hype < 4
        )
        mega_popularity_penalty = compute_mega_popularity_penalty(ticker, total_mentions)
        acceleration_score = compute_acceleration_score(
            mentions_this_period=mentions_this_period,
            mentions_previous_period=mentions_previous_period,
            mention_growth_rate=mention_growth_rate,
            unique_author_growth=unique_author_growth,
            unique_thread_growth=unique_thread_growth,
            average_hype_score=average_hype,
            repetition_score=average_repetition,
            new_ticker_detected=new_ticker_detected,
        )
        emerging_ticker_score = compute_emerging_ticker_score(
            acceleration_score=acceleration_score,
            source_diversity_score=source_diversity_score,
            average_alpha_signal_score=average_alpha,
            average_evidence_quality_numeric=average_evidence_numeric,
            average_claim_specificity_score=average_claim_specificity,
            average_hype_score=average_hype,
            repetition_score=average_repetition,
            mega_popularity_penalty=mega_popularity_penalty,
            low_mentions_high_signal=low_mentions_high_signal,
            new_ticker_detected=new_ticker_detected,
            mention_change=mention_change,
            total_mentions=total_mentions,
            unique_sources=int(group["source_name"].nunique()),
        )

        records.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "total_mentions": total_mentions,
                "bullish_mentions": bullish_mentions,
                "bearish_mentions": bearish_mentions,
                "neutral_mentions": neutral_mentions,
                "bullish_ratio": round(bullish_ratio, 4),
                "bearish_ratio": round(bearish_ratio, 4),
                "net_sentiment_score": round(float(net_sentiment_score), 4) if net_sentiment_score is not None else None,
                "average_confidence": round(float(group["confidence"].mean()), 4),
                "average_alpha_signal_score": average_alpha,
                "average_hype_adjusted_signal_score": round(float(group["hype_adjusted_signal_score"].mean()), 2),
                "average_depth_score": round(float(group["depth_score"].mean()), 2),
                "average_hype_score": average_hype,
                "average_claim_specificity_score": average_claim_specificity,
                "average_evidence_quality_numeric": average_evidence_numeric,
                "source_diversity_score": source_diversity_score,
                "controversy_score": controversy_score,
                "repetition_score": average_repetition,
                "unique_authors": int(group["author_hash"].nunique()),
                "unique_threads": int(group["thread_id"].nunique()),
                "first_seen_date": str(group["created_date"].min()),
                "most_recent_seen_date": str(group["created_date"].max()),
                "mentions_this_period": mentions_this_period,
                "mentions_previous_period": mentions_previous_period,
                "mention_change": mention_change,
                "mention_growth_rate": mention_growth_rate,
                "bullish_change": bullish_change,
                "bullish_growth_rate": bullish_growth_rate,
                "unique_author_growth": unique_author_growth,
                "unique_thread_growth": unique_thread_growth,
                "rolling_mention_average": rolling_mention_average,
                "acceleration_score": acceleration_score,
                "emerging_ticker_score": emerging_ticker_score,
                "low_mentions_high_signal": low_mentions_high_signal,
                "new_ticker_detected": new_ticker_detected,
                "mega_popularity_penalty": mega_popularity_penalty,
            }
        )

    return pd.DataFrame(records).sort_values(
        ["emerging_ticker_score", "average_alpha_signal_score"],
        ascending=[False, False],
    )


def build_time_bucket_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    bucket_rows: list[dict[str, Any]] = []
    for (ticker, company_name), ticker_group in long_df.groupby(["ticker", "company_name"], dropna=False):
        previous_mentions = 0
        for _, bucket_group in ticker_group.groupby("time_bucket", sort=True):
            bucket_start = bucket_group["time_bucket"].iloc[0]
            bucket_start_dt = parse_created_time(bucket_start)
            bucket_end_dt = bucket_start_dt + timedelta(days=6)
            total_mentions = int(len(bucket_group))
            bullish_mentions = int((bucket_group["sentiment"] == "bullish").sum())
            bearish_mentions = int((bucket_group["sentiment"] == "bearish").sum())
            neutral_mentions = int((bucket_group["sentiment"] == "neutral").sum())
            source_diversity_score = compute_source_diversity_score(bucket_group)
            average_alpha = round(float(bucket_group["alpha_signal_score"].mean()), 2)
            average_hype = round(float(bucket_group["hype_score"].mean()), 2)
            average_claim_specificity = round(float(bucket_group["claim_specificity_score"].mean()), 2)
            average_evidence_numeric = round(float(bucket_group["evidence_quality_numeric"].mean()), 2)
            average_repetition = round(float(bucket_group["repetition_score"].mean()), 2)
            mention_growth_rate = safe_growth_rate(total_mentions, previous_mentions)
            new_ticker_detected = total_mentions > 0 and previous_mentions == 0
            low_mentions_high_signal = total_mentions < 10 and average_alpha >= 75 and average_hype < 4
            acceleration_score = compute_acceleration_score(
                mentions_this_period=total_mentions,
                mentions_previous_period=previous_mentions,
                mention_growth_rate=mention_growth_rate,
                unique_author_growth=int(bucket_group["author_hash"].nunique()),
                unique_thread_growth=int(bucket_group["thread_id"].nunique()),
                average_hype_score=average_hype,
                repetition_score=average_repetition,
                new_ticker_detected=new_ticker_detected,
            )
            emerging_ticker_score = compute_emerging_ticker_score(
                acceleration_score=acceleration_score,
                source_diversity_score=source_diversity_score,
                average_alpha_signal_score=average_alpha,
                average_evidence_quality_numeric=average_evidence_numeric,
                average_claim_specificity_score=average_claim_specificity,
                average_hype_score=average_hype,
                repetition_score=average_repetition,
                mega_popularity_penalty=compute_mega_popularity_penalty(ticker, total_mentions),
                low_mentions_high_signal=low_mentions_high_signal,
                new_ticker_detected=new_ticker_detected,
                mention_change=total_mentions - previous_mentions,
                total_mentions=total_mentions,
                unique_sources=int(bucket_group["source_name"].nunique()),
            )
            bucket_rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "time_bucket": bucket_start,
                    "bucket_start": bucket_start_dt.date().isoformat(),
                    "bucket_end": bucket_end_dt.date().isoformat(),
                    "total_mentions": total_mentions,
                    "bullish_mentions": bullish_mentions,
                    "bearish_mentions": bearish_mentions,
                    "neutral_mentions": neutral_mentions,
                    "unique_authors": int(bucket_group["author_hash"].nunique()),
                    "unique_threads": int(bucket_group["thread_id"].nunique()),
                    "source_diversity_score": source_diversity_score,
                    "average_alpha_signal_score": average_alpha,
                    "average_hype_adjusted_signal_score": round(float(bucket_group["hype_adjusted_signal_score"].mean()), 2),
                    "average_depth_score": round(float(bucket_group["depth_score"].mean()), 2),
                    "average_hype_score": average_hype,
                    "average_claim_specificity_score": average_claim_specificity,
                    "average_evidence_quality_numeric": average_evidence_numeric,
                    "mention_growth_rate": mention_growth_rate,
                    "acceleration_score": acceleration_score,
                    "emerging_ticker_score": emerging_ticker_score,
                    "low_mentions_high_signal": low_mentions_high_signal,
                    "new_ticker_detected": new_ticker_detected,
                }
            )
            previous_mentions = total_mentions

    return pd.DataFrame(bucket_rows).sort_values(["bucket_start", "ticker"])


def apply_ticker_flags_to_mentions(long_df: pd.DataFrame, ticker_summary_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty or ticker_summary_df.empty:
        return long_df
    merged = long_df.merge(
        ticker_summary_df[
            [
                "ticker",
                "source_diversity_score",
                "controversy_score",
                "low_mentions_high_signal",
                "new_ticker_detected",
            ]
        ],
        on="ticker",
        how="left",
        suffixes=("", "_ticker"),
    )
    for column in ("source_diversity_score", "controversy_score", "low_mentions_high_signal", "new_ticker_detected"):
        ticker_column = f"{column}_ticker"
        if ticker_column in merged.columns:
            merged[column] = merged[ticker_column]
            merged = merged.drop(columns=[ticker_column])
    return merged


def build_run_summary(long_df: pd.DataFrame, ticker_summary_df: pd.DataFrame) -> dict[str, Any]:
    if long_df.empty:
        return {
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
        }

    item_sentiments = long_df.sort_values("confidence", ascending=False).drop_duplicates("item_id")
    summary = {
        "items_analyzed": int(item_sentiments["item_id"].nunique()),
        "ticker_mentions": int(len(long_df)),
        "tickers_found": int(long_df["ticker"].nunique()),
        "bullish_items": int((item_sentiments["sentiment"] == "bullish").sum()),
        "bearish_items": int((item_sentiments["sentiment"] == "bearish").sum()),
        "neutral_items": int((item_sentiments["sentiment"] == "neutral").sum()),
        "irrelevant_items": int((item_sentiments["sentiment"] == "irrelevant").sum()),
        "top_mentioned_tickers": ticker_summary_df.head(5)[["ticker", "total_mentions"]].to_dict(orient="records"),
        "top_accelerating_tickers": ticker_summary_df.sort_values("acceleration_score", ascending=False)
        .head(5)[["ticker", "acceleration_score"]]
        .to_dict(orient="records"),
        "high_depth_posts_found": int((item_sentiments["depth_score"] >= 7).sum()),
        "low_mentions_high_signal_count": int(ticker_summary_df["low_mentions_high_signal"].sum()) if not ticker_summary_df.empty else 0,
        "newly_detected_tickers_count": int(ticker_summary_df["new_ticker_detected"].sum()) if not ticker_summary_df.empty else 0,
    }
    return summary


def derive_claims_to_verify(text: str) -> list[str]:
    lowered = (text or "").lower()
    claims = []
    if "revenue" in lowered:
        claims.append("Verify the revenue claim against recent company filings.")
    if "margin" in lowered:
        claims.append("Check whether the margin claim is supported by reported results.")
    if "guidance" in lowered:
        claims.append("Confirm management guidance in the latest earnings materials.")
    if "contract" in lowered or "customer" in lowered:
        claims.append("Validate customer or contract claims using company disclosures.")
    if not claims and lowered:
        claims.append("Check the core thesis against filings, transcripts, or reliable reporting.")
    return claims[:3]


def build_deeper_analysis_fallback(item: dict[str, Any], classification_record: dict[str, Any]) -> dict[str, Any]:
    text = item.get("body_text", "")
    key_claims = derive_claims_to_verify(text)
    return {
        "stronger_summary": classification_record["summary"] or text[:220],
        "bull_thesis": classification_record["bull_case"],
        "bear_thesis": classification_record["bear_case"],
        "key_claims": key_claims,
        "claims_to_verify": classification_record["claims_to_verify"],
        "possible_catalysts": _extract_catalysts(text),
        "possible_risks": _extract_risks(text),
        "next_sources_to_check": [
            "Latest 10-Q or 10-K",
            "Most recent earnings call transcript",
            "Press releases or customer announcements",
        ],
    }


def parse_created_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(UTC)


def week_bucket_start(value: datetime) -> datetime:
    return (value - timedelta(days=value.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def sentiment_numeric(sentiment: str) -> float | None:
    mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0, "irrelevant": None}
    return mapping.get(sentiment, None)


def evidence_quality_numeric(evidence_quality: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(evidence_quality, 1)


def safe_growth_rate(current: int, previous: int) -> float | None:
    if previous <= 0:
        return None
    return round((current - previous) / previous, 4)


def compute_source_diversity_score(group: pd.DataFrame) -> float:
    source_count = int(group["source_name"].nunique())
    subreddit_count = int(group["subreddit"].nunique())
    score = min((source_count * 2.5) + (subreddit_count * 1.5), 10)
    return round(score, 2)


def compute_controversy_score(bullish_mentions: int, bearish_mentions: int) -> float:
    decisive_mentions = bullish_mentions + bearish_mentions
    if decisive_mentions == 0:
        return 0.0
    balance = (2 * min(bullish_mentions, bearish_mentions)) / decisive_mentions
    return round(balance * 10, 2)


def compute_mega_popularity_penalty(ticker: str, total_mentions: int) -> float:
    penalty = max(total_mentions - 12, 0) * 0.8
    if ticker in MEGA_CAP_TICKERS:
        penalty += 10
    return round(min(penalty, 25), 2)


def compute_acceleration_score(
    *,
    mentions_this_period: int,
    mentions_previous_period: int,
    mention_growth_rate: float | None,
    unique_author_growth: int,
    unique_thread_growth: int,
    average_hype_score: float,
    repetition_score: float,
    new_ticker_detected: bool,
) -> float:
    if new_ticker_detected:
        score = 30 + (mentions_this_period * 8) + max(unique_author_growth, 0) * 4 + max(unique_thread_growth, 0) * 3
    else:
        growth_component = 0 if mention_growth_rate is None else mention_growth_rate * 25
        score = (
            (mentions_this_period - mentions_previous_period) * 9
            + growth_component
            + max(unique_author_growth, 0) * 5
            + max(unique_thread_growth, 0) * 4
        )
    score -= max(average_hype_score - 4, 0) * 4
    score -= repetition_score * 1.2
    return round(max(min(score, 100), 0), 2)


def compute_emerging_ticker_score(
    *,
    acceleration_score: float,
    source_diversity_score: float,
    average_alpha_signal_score: float,
    average_evidence_quality_numeric: float,
    average_claim_specificity_score: float,
    average_hype_score: float,
    repetition_score: float,
    mega_popularity_penalty: float,
    low_mentions_high_signal: bool,
    new_ticker_detected: bool,
    mention_change: int,
    total_mentions: int,
    unique_sources: int,
) -> float:
    low_volume_bonus = 10 if total_mentions <= 12 and mention_change > 0 else 0
    multi_source_bonus = min(max(unique_sources - 1, 0) * 5, 15)
    score = (
        acceleration_score * 0.45
        + source_diversity_score * 2.8
        + average_alpha_signal_score * 0.28
        + average_evidence_quality_numeric * 6
        + average_claim_specificity_score * 2.5
        + low_volume_bonus
        + multi_source_bonus
        + (12 if low_mentions_high_signal else 0)
        + (10 if new_ticker_detected else 0)
    )
    score -= average_hype_score * 2.2
    score -= repetition_score * 1.4
    score -= mega_popularity_penalty
    return round(max(min(score, 100), 0), 2)


def _bucket_stats(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, bucket_group in group.groupby("time_bucket", sort=True):
        rows.append(
            {
                "time_bucket": bucket,
                "total_mentions": int(len(bucket_group)),
                "bullish_mentions": int((bucket_group["sentiment"] == "bullish").sum()),
                "unique_authors": int(bucket_group["author_hash"].nunique()),
                "unique_threads": int(bucket_group["thread_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _extract_catalysts(text: str) -> list[str]:
    lowered = (text or "").lower()
    catalysts = []
    if "contract" in lowered or "customer" in lowered:
        catalysts.append("Customer or contract wins")
    if "guidance" in lowered or "earnings" in lowered:
        catalysts.append("Upcoming earnings or guidance updates")
    if "launch" in lowered or "product" in lowered:
        catalysts.append("Product or launch execution milestones")
    return catalysts[:3]


def _extract_risks(text: str) -> list[str]:
    lowered = (text or "").lower()
    risks = []
    for keyword, label in (
        ("dilution", "Dilution risk"),
        ("execution", "Execution risk"),
        ("capex", "Capital intensity"),
        ("liquidity", "Liquidity risk"),
        ("customer concentration", "Customer concentration"),
        ("margin", "Margin pressure"),
    ):
        if keyword in lowered:
            risks.append(label)
    return risks[:4]
