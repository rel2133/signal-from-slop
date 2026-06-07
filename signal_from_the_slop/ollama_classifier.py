from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from signal_from_the_slop.analytics import (
    build_deeper_analysis_fallback,
    derive_claim_specificity_score,
    derive_claims_to_verify,
    derive_repetition_score,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_SENTIMENT = {
    "sentiment": "irrelevant",
    "confidence": 0.3,
    "tickers": [],
    "company_names": [],
    "summary": "",
    "bull_case": "",
    "bear_case": "",
    "evidence_quality": "low",
    "depth_score": 0,
    "hype_score": 0,
    "claim_specificity_score": 0,
    "repetition_score": 0,
    "needs_deeper_analysis": False,
    "red_flags": [],
    "claims_to_verify": [],
}

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral", "irrelevant"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "tickers": {"type": "array", "items": {"type": "string"}},
        "company_names": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "bull_case": {"type": "string"},
        "bear_case": {"type": "string"},
        "evidence_quality": {"type": "string", "enum": ["low", "medium", "high"]},
        "depth_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "hype_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "claim_specificity_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "repetition_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "needs_deeper_analysis": {"type": "boolean"},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
    },
    "required": list(DEFAULT_SENTIMENT.keys()),
    "additionalProperties": False,
}

DEEPER_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "stronger_summary": {"type": "string"},
        "bull_thesis": {"type": "string"},
        "bear_thesis": {"type": "string"},
        "key_claims": {"type": "array", "items": {"type": "string"}},
        "claims_to_verify": {"type": "array", "items": {"type": "string"}},
        "possible_catalysts": {"type": "array", "items": {"type": "string"}},
        "possible_risks": {"type": "array", "items": {"type": "string"}},
        "next_sources_to_check": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "stronger_summary",
        "bull_thesis",
        "bear_thesis",
        "key_claims",
        "claims_to_verify",
        "possible_catalysts",
        "possible_risks",
        "next_sources_to_check",
    ],
    "additionalProperties": False,
}


@dataclass
class ClassifierResult:
    payload: dict[str, Any]
    mode: str


class OllamaClassifier:
    def __init__(self, model: str, endpoint: str, timeout: int = 90, retries: int = 2) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries

    def classify(self, item: dict[str, Any], extraction: dict[str, list[str]]) -> ClassifierResult:
        payload = self._call_ollama(
            schema=CLASSIFICATION_SCHEMA,
            system_prompt=(
                "Classify Reddit stock posts. Return strict JSON only. "
                "Use only tickers or company names that are already allowed. "
                "If the stock reference is uncertain, leave tickers/company_names empty."
            ),
            user_payload=self._build_stage_one_payload(item, extraction),
        )
        if payload is None:
            return ClassifierResult(payload=self._heuristic_classification(item, extraction), mode="fallback")
        return ClassifierResult(payload=self._normalize_payload(payload, extraction, item), mode="ollama")

    def deep_analyze(self, item: dict[str, Any], classification_payload: dict[str, Any]) -> ClassifierResult:
        payload = self._call_ollama(
            schema=DEEPER_ANALYSIS_SCHEMA,
            system_prompt=(
                "Deepen the research analysis of this Reddit stock item. Return strict JSON only. "
                "Focus on key claims, risks, catalysts, and what should be verified next."
            ),
            user_payload=self._build_stage_two_payload(item, classification_payload),
        )
        if payload is None:
            return ClassifierResult(
                payload=build_deeper_analysis_fallback(item, classification_payload),
                mode="fallback",
            )
        return ClassifierResult(payload=self._normalize_deeper_payload(payload, classification_payload), mode="ollama")

    def list_available_models(self) -> list[str]:
        response = requests.get(self._tags_endpoint(), timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models if model.get("name")]

    def _call_ollama(self, *, schema: dict[str, Any], system_prompt: str, user_payload: str) -> dict[str, Any] | None:
        attempt = 0
        while attempt <= self.retries:
            attempt += 1
            try:
                response = requests.post(
                    self.endpoint,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_payload},
                        ],
                        "format": schema,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=self.timeout,
                )
                if response.status_code in {429, 503} and attempt <= self.retries:
                    time.sleep(attempt)
                    continue
                response.raise_for_status()
                return self._parse_json_content(response.json()["message"]["content"])
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
                LOGGER.warning("Ollama request failed on attempt %s: %s", attempt, exc)
                if attempt > self.retries:
                    break
                time.sleep(attempt)
        return None

    def _build_stage_one_payload(self, item: dict[str, Any], extraction: dict[str, list[str]]) -> str:
        return json.dumps(
            {
                "task": "Classify this Reddit stock item using only the supplied text and metadata.",
                "allowed_tickers": extraction["tickers"],
                "allowed_company_names": extraction["company_names"],
                "required_output_schema": CLASSIFICATION_SCHEMA,
                "item": {
                    "item_type": item.get("item_type"),
                    "subreddit": item.get("subreddit"),
                    "thread_title": item.get("thread_title"),
                    "body_text": item.get("body_text"),
                    "score": item.get("score"),
                    "comment_depth": item.get("comment_depth"),
                },
            },
            ensure_ascii=True,
        )

    def _build_stage_two_payload(self, item: dict[str, Any], classification_payload: dict[str, Any]) -> str:
        return json.dumps(
            {
                "task": "Perform deeper research triage on this Reddit stock item.",
                "required_output_schema": DEEPER_ANALYSIS_SCHEMA,
                "classification": classification_payload,
                "item": {
                    "subreddit": item.get("subreddit"),
                    "thread_title": item.get("thread_title"),
                    "body_text": item.get("body_text"),
                    "score": item.get("score"),
                },
            },
            ensure_ascii=True,
        )

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in Ollama response.")
            return json.loads(content[start : end + 1])

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        extraction: dict[str, list[str]],
        item: dict[str, Any],
    ) -> dict[str, Any]:
        result = dict(DEFAULT_SENTIMENT)
        for key in DEFAULT_SENTIMENT:
            if key in payload:
                result[key] = payload[key]

        text = " ".join([item.get("thread_title", ""), item.get("body_text", "")]).strip()
        result["sentiment"] = self._coerce_sentiment(result["sentiment"])
        result["confidence"] = self._coerce_float(result.get("confidence"), minimum=0.0, maximum=1.0)
        result["tickers"] = self._filter_allowed_values(self._coerce_string_list(result.get("tickers")), extraction["tickers"])
        result["company_names"] = self._filter_allowed_values(
            self._coerce_string_list(result.get("company_names")),
            extraction["company_names"],
        )
        result["summary"] = str(result.get("summary") or "").strip()
        result["bull_case"] = self._coerce_case_text(result.get("bull_case"))
        result["bear_case"] = self._coerce_case_text(result.get("bear_case"))
        result["evidence_quality"] = self._coerce_quality(result.get("evidence_quality"))
        result["depth_score"] = int(self._coerce_float(result.get("depth_score"), minimum=0, maximum=10))
        result["hype_score"] = int(self._coerce_float(result.get("hype_score"), minimum=0, maximum=10))
        result["claim_specificity_score"] = int(
            self._coerce_float(result.get("claim_specificity_score"), minimum=0, maximum=10)
        )
        result["repetition_score"] = int(self._coerce_float(result.get("repetition_score"), minimum=0, maximum=10))
        result["needs_deeper_analysis"] = self._coerce_bool(result.get("needs_deeper_analysis"))
        result["red_flags"] = self._coerce_string_list(result.get("red_flags"))
        result["claims_to_verify"] = self._coerce_string_list(result.get("claims_to_verify"))

        if not result["tickers"]:
            result["tickers"] = extraction["tickers"]
        if not result["company_names"]:
            result["company_names"] = extraction["company_names"]
        if not result["claim_specificity_score"]:
            result["claim_specificity_score"] = derive_claim_specificity_score(text)
        if not result["repetition_score"]:
            result["repetition_score"] = derive_repetition_score(text)
        if not result["claims_to_verify"]:
            result["claims_to_verify"] = derive_claims_to_verify(text)
        return result

    def _normalize_deeper_payload(self, payload: dict[str, Any], classification_payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        for key in (
            "key_claims",
            "claims_to_verify",
            "possible_catalysts",
            "possible_risks",
            "next_sources_to_check",
        ):
            normalized[key] = self._coerce_string_list(normalized.get(key))
        normalized["stronger_summary"] = str(normalized.get("stronger_summary") or classification_payload.get("summary") or "").strip()
        normalized["bull_thesis"] = str(normalized.get("bull_thesis") or classification_payload.get("bull_case") or "").strip()
        normalized["bear_thesis"] = str(normalized.get("bear_thesis") or classification_payload.get("bear_case") or "").strip()
        return normalized

    def _heuristic_classification(self, item: dict[str, Any], extraction: dict[str, list[str]]) -> dict[str, Any]:
        text = " ".join([item.get("thread_title", ""), item.get("body_text", "")]).lower()
        result = dict(DEFAULT_SENTIMENT)
        result["tickers"] = extraction["tickers"]
        result["company_names"] = extraction["company_names"]
        result["summary"] = (item.get("body_text") or item.get("thread_title") or "")[:220].strip()

        bullish_terms = ("undervalued", "growth", "winning", "margin", "cash flow", "bull", "upside", "improving")
        bearish_terms = ("risk", "dilution", "overvalued", "thin", "skeptical", "bear", "execution", "concern")
        hype_terms = ("moon", "vertical", "obviously", "shorts", "wrecked", "insane", "easy money", "guaranteed")

        bullish_hits = sum(term in text for term in bullish_terms)
        bearish_hits = sum(term in text for term in bearish_terms)
        hype_hits = sum(term in text for term in hype_terms)
        result["claim_specificity_score"] = derive_claim_specificity_score(text)
        result["repetition_score"] = derive_repetition_score(text)
        result["claims_to_verify"] = derive_claims_to_verify(text)

        if not extraction["tickers"] and not extraction["company_names"]:
            result["sentiment"] = "irrelevant"
            result["confidence"] = 0.85
            result["summary"] = "No confident stock ticker or company reference was identified."
            result["red_flags"] = ["unknown ticker/company"]
            return result

        if bullish_hits > bearish_hits:
            result["sentiment"] = "bullish"
        elif bearish_hits > bullish_hits:
            result["sentiment"] = "bearish"
        else:
            result["sentiment"] = "neutral"

        result["confidence"] = 0.68
        result["bull_case"] = "Post argues for improving fundamentals, upside, or business momentum." if bullish_hits else ""
        result["bear_case"] = "Post points to downside risk, dilution, or execution concerns." if bearish_hits else ""
        result["evidence_quality"] = "high" if result["claim_specificity_score"] >= 6 else "medium" if bullish_hits or bearish_hits else "low"
        result["depth_score"] = min(2 + bullish_hits + bearish_hits + (result["claim_specificity_score"] // 2), 10)
        result["hype_score"] = min(hype_hits * 3, 10)
        result["needs_deeper_analysis"] = (
            result["depth_score"] >= 7
            or result["evidence_quality"] == "high"
            or result["claim_specificity_score"] >= 7
        )
        red_flags = []
        if hype_hits:
            red_flags.append("pump language")
        if result["evidence_quality"] == "low":
            red_flags.append("no evidence")
        if "next tesla" in text:
            red_flags.append("comparison hype")
        result["red_flags"] = red_flags
        return result

    def _coerce_sentiment(self, value: Any) -> str:
        allowed = {"bullish", "bearish", "neutral", "irrelevant"}
        normalized = str(value or "irrelevant").strip().lower()
        return normalized if normalized in allowed else "irrelevant"

    def _coerce_quality(self, value: Any) -> str:
        allowed = {"low", "medium", "high"}
        normalized = str(value or "low").strip().lower()
        return normalized if normalized in allowed else "low"

    def _coerce_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _coerce_float(self, value: Any, minimum: float, maximum: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = minimum
        return max(min(parsed, maximum), minimum)

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    def _filter_allowed_values(self, values: list[str], allowed: list[str]) -> list[str]:
        allowed_set = set(allowed)
        return [value for value in values if value in allowed_set]

    def _tags_endpoint(self) -> str:
        if self.endpoint.endswith("/api/chat"):
            return f"{self.endpoint[:-len('/api/chat')]}/api/tags"
        return f"{self.endpoint.rstrip('/')}/api/tags"

    def _coerce_case_text(self, value: Any) -> str:
        normalized = str(value or "").strip()
        return "" if normalized.lower() in {"true", "false", "null", "none"} else normalized
