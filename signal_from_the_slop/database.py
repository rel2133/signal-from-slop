from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


JSON_COLUMNS = {
    "selected_source_ids",
    "tickers",
    "company_names",
    "red_flags",
    "claims_to_verify",
}


def init_db(db_path: str | Path, schema_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        schema = Path(schema_path).read_text(encoding="utf-8")
        conn.executescript(schema)
        conn.commit()


def add_source(
    db_path: str | Path,
    *,
    source_key: str,
    source_type: str,
    display_name: str,
    normalized_value: str,
    url: str,
    notes: str = "",
    active: bool = True,
) -> None:
    payload = (
        source_key,
        source_type,
        display_name,
        normalized_value,
        url,
        int(active),
        _now_iso(),
        notes,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sources (
                source_key, source_type, display_name, normalized_value,
                url, active, date_added, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                source_type=excluded.source_type,
                display_name=excluded.display_name,
                normalized_value=excluded.normalized_value,
                url=excluded.url,
                notes=excluded.notes
            """,
            payload,
        )
        conn.commit()


def load_sources(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                source_id,
                source_key,
                source_type,
                display_name,
                normalized_value,
                url,
                active,
                date_added,
                notes
            FROM sources
            ORDER BY active DESC, source_type, display_name
            """,
            conn,
        )
    if frame.empty:
        return frame
    frame["active"] = frame["active"].astype(bool)
    return frame


def update_sources(db_path: str | Path, records: list[dict[str, Any]]) -> None:
    payload = [
        (
            int(bool(record.get("active", True))),
            str(record.get("notes", "")),
            int(record["source_id"]),
        )
        for record in records
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE sources SET active = ?, notes = ? WHERE source_id = ?",
            payload,
        )
        conn.commit()


def delete_sources(db_path: str | Path, source_ids: list[int]) -> None:
    if not source_ids:
        return
    placeholders = ",".join("?" for _ in source_ids)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DELETE FROM sources WHERE source_id IN ({placeholders})", source_ids)
        conn.commit()


def create_analysis_run(db_path: str | Path, config: dict[str, Any]) -> str:
    analysis_run_id = config.get("analysis_run_id") or f"run_{uuid.uuid4().hex[:12]}"
    payload = (
        analysis_run_id,
        _now_iso(),
        None,
        "running",
        config.get("data_mode", "live"),
        json.dumps(config.get("selected_source_ids", []), ensure_ascii=True),
        config.get("time_window_label", ""),
        config.get("time_window_start", ""),
        config.get("time_window_end", ""),
        int(config.get("max_posts_per_source", 0)),
        int(config.get("max_comments_per_thread", 0)),
        int(bool(config.get("include_comments", True))),
        int(bool(config.get("run_deeper_analysis", False))),
        config.get("ollama_model", ""),
        config.get("ollama_url", ""),
        json.dumps({}, ensure_ascii=True),
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                analysis_run_id, started_at, completed_at, status, data_mode,
                selected_source_ids, time_window_label, time_window_start,
                time_window_end, max_posts_per_source, max_comments_per_thread,
                include_comments, run_deeper_analysis, ollama_model, ollama_url,
                summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()
    return analysis_run_id


def complete_analysis_run(db_path: str | Path, analysis_run_id: str, summary: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE analysis_runs
            SET completed_at = ?, status = ?, summary_json = ?
            WHERE analysis_run_id = ?
            """,
            (_now_iso(), "completed", json.dumps(summary, ensure_ascii=True), analysis_run_id),
        )
        conn.commit()


def fail_analysis_run(db_path: str | Path, analysis_run_id: str, summary: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE analysis_runs
            SET completed_at = ?, status = ?, summary_json = ?
            WHERE analysis_run_id = ?
            """,
            (_now_iso(), "failed", json.dumps(summary, ensure_ascii=True), analysis_run_id),
        )
        conn.commit()


def load_analysis_runs(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                analysis_run_id,
                started_at,
                completed_at,
                status,
                data_mode,
                selected_source_ids,
                time_window_label,
                time_window_start,
                time_window_end,
                max_posts_per_source,
                max_comments_per_thread,
                include_comments,
                run_deeper_analysis,
                ollama_model,
                ollama_url,
                summary_json
            FROM analysis_runs
            ORDER BY started_at DESC
            """,
            conn,
        )
    return _decode_json_columns(frame)


def latest_analysis_run_id(db_path: str | Path) -> str | None:
    runs = load_analysis_runs(db_path)
    if runs.empty:
        return None
    return str(runs.iloc[0]["analysis_run_id"])


def load_previously_analyzed_item_ids(db_path: str | Path, *, data_mode: str = "live") -> set[str]:
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
    return {str(row[0]) for row in rows}


def upsert_reddit_items(db_path: str | Path, items: list[dict[str, Any]]) -> None:
    payload = [
        (
            item["item_id"],
            item.get("source_id"),
            item.get("source_type", ""),
            item.get("source_name", ""),
            item.get("subreddit", ""),
            item.get("thread_id", ""),
            item.get("thread_title", ""),
            item.get("thread_url", ""),
            item.get("comment_id"),
            item.get("parent_id"),
            item.get("item_type", "post"),
            item.get("body_text", ""),
            item.get("author", ""),
            item.get("author_hash", ""),
            int(item.get("score", 0)),
            item.get("created_time", ""),
            item.get("permalink", ""),
            int(item.get("comment_depth", 0)),
            json.dumps(item.get("raw_json", item), ensure_ascii=True),
        )
        for item in items
    ]
    if not payload:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO reddit_items (
                item_id, source_id, source_type, source_name, subreddit,
                thread_id, thread_title, thread_url, comment_id, parent_id,
                item_type, body_text, author, author_hash, score, created_time,
                permalink, comment_depth, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                source_id=excluded.source_id,
                source_type=excluded.source_type,
                source_name=excluded.source_name,
                subreddit=excluded.subreddit,
                thread_id=excluded.thread_id,
                thread_title=excluded.thread_title,
                thread_url=excluded.thread_url,
                comment_id=excluded.comment_id,
                parent_id=excluded.parent_id,
                item_type=excluded.item_type,
                body_text=excluded.body_text,
                author=excluded.author,
                author_hash=excluded.author_hash,
                score=excluded.score,
                created_time=excluded.created_time,
                permalink=excluded.permalink,
                comment_depth=excluded.comment_depth,
                raw_json=excluded.raw_json
            """,
            payload,
        )
        conn.commit()


def replace_run_classifications(db_path: str | Path, analysis_run_id: str, rows: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM classifications WHERE analysis_run_id = ?", (analysis_run_id,))
        if rows:
            payload = [
                (
                    analysis_run_id,
                    row["item_id"],
                    row["sentiment"],
                    float(row["confidence"]),
                    json.dumps(row["tickers"], ensure_ascii=True),
                    json.dumps(row["company_names"], ensure_ascii=True),
                    row["summary"],
                    row["bull_case"],
                    row["bear_case"],
                    row["evidence_quality"],
                    int(row["depth_score"]),
                    int(row["hype_score"]),
                    int(row["claim_specificity_score"]),
                    int(row["repetition_score"]),
                    int(bool(row["needs_deeper_analysis"])),
                    json.dumps(row["red_flags"], ensure_ascii=True),
                    json.dumps(row["claims_to_verify"], ensure_ascii=True),
                    float(row["alpha_signal_score"]),
                    float(row["hype_adjusted_signal_score"]),
                    row["model_name"],
                    row["classifier_mode"],
                    json.dumps(row.get("deeper_analysis_json", {}), ensure_ascii=True),
                    _now_iso(),
                )
                for row in rows
            ]
            conn.executemany(
                """
                INSERT INTO classifications (
                    analysis_run_id, item_id, sentiment, confidence, tickers,
                    company_names, summary, bull_case, bear_case, evidence_quality,
                    depth_score, hype_score, claim_specificity_score, repetition_score,
                    needs_deeper_analysis, red_flags, claims_to_verify,
                    alpha_signal_score, hype_adjusted_signal_score, model_name,
                    classifier_mode, deeper_analysis_json, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        conn.commit()


def replace_run_mentions(db_path: str | Path, analysis_run_id: str, rows: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM item_ticker_mentions WHERE analysis_run_id = ?", (analysis_run_id,))
        if rows:
            payload = [
                (
                    analysis_run_id,
                    row["item_id"],
                    row.get("source_id"),
                    row["source_type"],
                    row["source_name"],
                    row["subreddit"],
                    row["thread_id"],
                    row["thread_title"],
                    row.get("comment_id"),
                    row.get("parent_id"),
                    row["author_hash"],
                    row["created_time"],
                    row["created_date"],
                    row["time_bucket"],
                    row["item_type"],
                    row["ticker"],
                    row["company_name"],
                    row["sentiment"],
                    row.get("sentiment_numeric"),
                    float(row["confidence"]),
                    float(row["alpha_signal_score"]),
                    float(row["hype_adjusted_signal_score"]),
                    int(row["depth_score"]),
                    int(row["hype_score"]),
                    row["evidence_quality"],
                    int(row["evidence_quality_numeric"]),
                    int(row["claim_specificity_score"]),
                    float(row["source_diversity_score"]),
                    float(row["controversy_score"]),
                    int(row["repetition_score"]),
                    int(row["reddit_score"]),
                    int(row["comment_depth"]),
                    int(row["text_length"]),
                    int(row["word_count"]),
                    int(bool(row["contains_link"])),
                    int(bool(row["contains_numbers"])),
                    int(bool(row["contains_financial_terms"])),
                    int(bool(row["needs_deeper_analysis"])),
                    int(bool(row["low_mentions_high_signal"])),
                    int(bool(row["new_ticker_detected"])),
                    int(row["red_flag_count"]),
                    json.dumps(row["claims_to_verify"], ensure_ascii=True),
                    row["summary"],
                    row["bull_case"],
                    row["bear_case"],
                    json.dumps(row["red_flags"], ensure_ascii=True),
                    json.dumps(row.get("deeper_analysis_json", {}), ensure_ascii=True),
                    row["body_text"],
                    row["permalink"],
                )
                for row in rows
            ]
            conn.executemany(
                """
                INSERT INTO item_ticker_mentions (
                    analysis_run_id, item_id, source_id, source_type, source_name,
                    subreddit, thread_id, thread_title, comment_id, parent_id,
                    author_hash, created_time, created_date, time_bucket, item_type,
                    ticker, company_name, sentiment, sentiment_numeric, confidence,
                    alpha_signal_score, hype_adjusted_signal_score, depth_score,
                    hype_score, evidence_quality, evidence_quality_numeric,
                    claim_specificity_score, source_diversity_score, controversy_score,
                    repetition_score, reddit_score, comment_depth, text_length,
                    word_count, contains_link, contains_numbers,
                    contains_financial_terms, needs_deeper_analysis,
                    low_mentions_high_signal, new_ticker_detected, red_flag_count,
                    claims_to_verify, summary, bull_case, bear_case, red_flags,
                    deeper_analysis_json, body_text, permalink
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        conn.commit()


def replace_run_ticker_summaries(db_path: str | Path, analysis_run_id: str, rows: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM ticker_summaries WHERE analysis_run_id = ?", (analysis_run_id,))
        if rows:
            payload = [
                (
                    analysis_run_id,
                    row["ticker"],
                    row["company_name"],
                    int(row["total_mentions"]),
                    int(row["bullish_mentions"]),
                    int(row["bearish_mentions"]),
                    int(row["neutral_mentions"]),
                    float(row["bullish_ratio"]),
                    float(row["bearish_ratio"]),
                    _nullable_float(row.get("net_sentiment_score")),
                    float(row["average_confidence"]),
                    float(row["average_alpha_signal_score"]),
                    float(row["average_hype_adjusted_signal_score"]),
                    float(row["average_depth_score"]),
                    float(row["average_hype_score"]),
                    float(row["average_claim_specificity_score"]),
                    float(row["average_evidence_quality_numeric"]),
                    float(row["source_diversity_score"]),
                    float(row["controversy_score"]),
                    float(row["repetition_score"]),
                    int(row["unique_authors"]),
                    int(row["unique_threads"]),
                    row["first_seen_date"],
                    row["most_recent_seen_date"],
                    int(row["mentions_this_period"]),
                    int(row["mentions_previous_period"]),
                    int(row["mention_change"]),
                    _nullable_float(row.get("mention_growth_rate")),
                    int(row["bullish_change"]),
                    _nullable_float(row.get("bullish_growth_rate")),
                    int(row["unique_author_growth"]),
                    int(row["unique_thread_growth"]),
                    float(row["rolling_mention_average"]),
                    float(row["acceleration_score"]),
                    float(row["emerging_ticker_score"]),
                    int(bool(row["low_mentions_high_signal"])),
                    int(bool(row["new_ticker_detected"])),
                    float(row["mega_popularity_penalty"]),
                )
                for row in rows
            ]
            conn.executemany(
                """
                INSERT INTO ticker_summaries (
                    analysis_run_id, ticker, company_name, total_mentions,
                    bullish_mentions, bearish_mentions, neutral_mentions,
                    bullish_ratio, bearish_ratio, net_sentiment_score,
                    average_confidence, average_alpha_signal_score,
                    average_hype_adjusted_signal_score, average_depth_score,
                    average_hype_score, average_claim_specificity_score,
                    average_evidence_quality_numeric, source_diversity_score,
                    controversy_score, repetition_score, unique_authors,
                    unique_threads, first_seen_date, most_recent_seen_date,
                    mentions_this_period, mentions_previous_period, mention_change,
                    mention_growth_rate, bullish_change, bullish_growth_rate,
                    unique_author_growth, unique_thread_growth,
                    rolling_mention_average, acceleration_score,
                    emerging_ticker_score, low_mentions_high_signal,
                    new_ticker_detected, mega_popularity_penalty
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        conn.commit()


def replace_run_time_buckets(db_path: str | Path, analysis_run_id: str, rows: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM ticker_time_buckets WHERE analysis_run_id = ?", (analysis_run_id,))
        if rows:
            payload = [
                (
                    analysis_run_id,
                    row["ticker"],
                    row["company_name"],
                    row["time_bucket"],
                    row["bucket_start"],
                    row["bucket_end"],
                    int(row["total_mentions"]),
                    int(row["bullish_mentions"]),
                    int(row["bearish_mentions"]),
                    int(row["neutral_mentions"]),
                    int(row["unique_authors"]),
                    int(row["unique_threads"]),
                    float(row["source_diversity_score"]),
                    float(row["average_alpha_signal_score"]),
                    float(row["average_hype_adjusted_signal_score"]),
                    float(row["average_depth_score"]),
                    float(row["average_hype_score"]),
                    float(row["average_claim_specificity_score"]),
                    float(row["average_evidence_quality_numeric"]),
                    _nullable_float(row.get("mention_growth_rate")),
                    float(row["acceleration_score"]),
                    float(row["emerging_ticker_score"]),
                    int(bool(row["low_mentions_high_signal"])),
                    int(bool(row["new_ticker_detected"])),
                )
                for row in rows
            ]
            conn.executemany(
                """
                INSERT INTO ticker_time_buckets (
                    analysis_run_id, ticker, company_name, time_bucket,
                    bucket_start, bucket_end, total_mentions, bullish_mentions,
                    bearish_mentions, neutral_mentions, unique_authors,
                    unique_threads, source_diversity_score,
                    average_alpha_signal_score,
                    average_hype_adjusted_signal_score, average_depth_score,
                    average_hype_score, average_claim_specificity_score,
                    average_evidence_quality_numeric, mention_growth_rate,
                    acceleration_score, emerging_ticker_score,
                    low_mentions_high_signal, new_ticker_detected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        conn.commit()


def load_run_mentions(db_path: str | Path, analysis_run_id: str) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT * FROM item_ticker_mentions
        WHERE analysis_run_id = ?
          AND ticker != 'UNKNOWN'
        ORDER BY created_time DESC, alpha_signal_score DESC
        """,
        (analysis_run_id,),
    )


def load_run_ticker_summaries(db_path: str | Path, analysis_run_id: str) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT * FROM ticker_summaries
        WHERE analysis_run_id = ?
          AND ticker != 'UNKNOWN'
        ORDER BY emerging_ticker_score DESC, average_alpha_signal_score DESC
        """,
        (analysis_run_id,),
    )


def load_historical_ticker_summaries(
    db_path: str | Path,
    *,
    data_mode: str,
    selected_source_ids_json: str,
) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT
            ts.*,
            ar.started_at,
            ar.completed_at,
            ar.data_mode,
            ar.selected_source_ids,
            ar.summary_json
        FROM ticker_summaries ts
        JOIN analysis_runs ar
            ON ar.analysis_run_id = ts.analysis_run_id
        WHERE ar.status = 'completed'
          AND ar.data_mode = ?
          AND ar.selected_source_ids = ?
          AND ts.ticker != 'UNKNOWN'
        ORDER BY ar.completed_at, ts.ticker
        """,
        (data_mode, selected_source_ids_json),
    )


def load_run_time_buckets(db_path: str | Path, analysis_run_id: str) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT * FROM ticker_time_buckets
        WHERE analysis_run_id = ?
          AND ticker != 'UNKNOWN'
        ORDER BY bucket_start, ticker
        """,
        (analysis_run_id,),
    )


def load_run_classification_quality(db_path: str | Path, analysis_run_id: str) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT
            item_id,
            classifier_mode,
            model_name,
            analyzed_at
        FROM classifications
        WHERE analysis_run_id = ?
        """,
        (analysis_run_id,),
    )


def load_run_source_activity(db_path: str | Path, analysis_run_id: str) -> pd.DataFrame:
    return _load_table(
        db_path,
        """
        SELECT
            ri.source_id,
            ri.source_name,
            COUNT(DISTINCT c.item_id) AS items_analyzed,
            COUNT(itm.item_id) AS ticker_mentions,
            MAX(ri.created_time) AS newest_item_time
        FROM classifications c
        JOIN reddit_items ri
            ON ri.item_id = c.item_id
        LEFT JOIN item_ticker_mentions itm
            ON itm.analysis_run_id = c.analysis_run_id
           AND itm.item_id = c.item_id
        WHERE c.analysis_run_id = ?
        GROUP BY ri.source_id, ri.source_name
        ORDER BY items_analyzed DESC, ri.source_name
        """,
        (analysis_run_id,),
    )


def load_run_summary(db_path: str | Path, analysis_run_id: str) -> dict[str, Any]:
    runs = _load_table(
        db_path,
        "SELECT summary_json FROM analysis_runs WHERE analysis_run_id = ?",
        (analysis_run_id,),
    )
    if runs.empty:
        return {}
    return runs.iloc[0].get("summary_json", {}) or {}


def load_full_results_records(db_path: str | Path, analysis_run_id: str) -> list[dict[str, Any]]:
    mentions = load_run_mentions(db_path, analysis_run_id)
    if mentions.empty:
        return []
    return mentions.to_dict(orient="records")


def db_overview(db_path: str | Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {
            "sources": _count_rows(conn, "sources"),
            "analysis_runs": _count_rows(conn, "analysis_runs"),
            "reddit_items": _count_rows(conn, "reddit_items"),
            "item_ticker_mentions": _count_rows(conn, "item_ticker_mentions"),
        }


def _count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def _load_table(db_path: str | Path, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    return _decode_json_columns(frame)


def _decode_json_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    for column in working.columns:
        if column in JSON_COLUMNS:
            working[column] = working[column].fillna("[]").apply(_decode_json_cell)
        elif column.endswith("_json"):
            working[column] = working[column].fillna("{}").apply(_decode_json_cell)
    for flag_column in ("active", "include_comments", "run_deeper_analysis", "needs_deeper_analysis", "low_mentions_high_signal", "new_ticker_detected"):
        if flag_column in working.columns:
            working[flag_column] = working[flag_column].fillna(0).astype(bool)
    return working


def _decode_json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return []
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _nullable_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
