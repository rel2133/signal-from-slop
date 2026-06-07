from __future__ import annotations


QUALITY_WEIGHTS = {"low": 5, "medium": 15, "high": 25}
HYPE_TERMS = (
    "moon",
    "vertical",
    "obviously",
    "shorts get wrecked",
    "insane",
    "everyone knows",
    "no spreadsheet needed",
)
SPECIFICITY_TERMS = (
    "revenue",
    "margin",
    "cash flow",
    "free cash flow",
    "guidance",
    "capex",
    "liquidity",
    "customer concentration",
    "dilution",
    "commercial",
)
RISK_TERMS = ("risk", "dilution", "execution", "liquidity", "margin", "capex", "customer concentration")
NO_EVIDENCE_TERMS = ("no spreadsheet needed", "everyone knows", "obviously")


def compute_alpha_signal_score(result: dict, item: dict) -> float:
    reddit_score = int(item.get("score", 0))
    text = " ".join([item.get("thread_title", ""), item.get("body_text", "")]).lower()

    reddit_component = min(max(reddit_score, 0), 250) / 25
    depth_component = max(min(result.get("depth_score", 0), 10), 0) * 2.5
    quality_component = QUALITY_WEIGHTS.get(result.get("evidence_quality", "low"), 5)
    confidence_component = max(min(float(result.get("confidence", 0.0)), 1.0), 0.0) * 10
    claim_specificity_component = max(min(int(result.get("claim_specificity_score", 0)), 10), 0) * 3.2

    specificity_hits = sum(term in text for term in SPECIFICITY_TERMS)
    risk_hits = sum(term in text for term in RISK_TERMS)
    hype_hits = sum(term in text for term in HYPE_TERMS)
    no_evidence_hits = sum(term in text for term in NO_EVIDENCE_TERMS)

    specificity_bonus = min(specificity_hits * 3, 12)
    risk_bonus = 6 if risk_hits or result.get("bear_case") else 0
    hype_penalty = max(min(result.get("hype_score", 0), 10), 0) * 4 + min(hype_hits * 8, 32)
    red_flag_penalty = min(len(result.get("red_flags", [])) * 5, 20)
    no_evidence_penalty = min(no_evidence_hits * 12, 24)

    summary = (result.get("summary") or "").lower()
    summary_bonus = 4 if any(token in summary for token in ("margin", "revenue", "cash", "risk", "guidance")) else 0

    score = (
        reddit_component
        + depth_component
        + quality_component
        + confidence_component
        + claim_specificity_component
        + specificity_bonus
        + risk_bonus
        + summary_bonus
        - hype_penalty
        - red_flag_penalty
        - no_evidence_penalty
    )
    if hype_hits >= 2 and specificity_hits == 0:
        score = min(score, 35)
    if no_evidence_hits and risk_hits == 0:
        score = min(score, 25)
    return round(max(min(score, 100), 0), 2)
