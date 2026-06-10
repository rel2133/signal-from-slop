from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


AMBIGUOUS_WORDS = {
    "A",
    "AI",
    "ALL",
    "AM",
    "ARE",
    "AT",
    "BE",
    "CAN",
    "FOR",
    "GO",
    "HAS",
    "IT",
    "NOW",
    "ON",
    "SO",
    "TO",
    "UP",
    "WE",
}


@dataclass(frozen=True)
class ExtractionResult:
    tickers: list[str]
    company_names: list[str]


class TickerExtractor:
    def __init__(self, ticker_catalog_path: str | Path) -> None:
        self.ticker_catalog_path = Path(ticker_catalog_path)
        self.catalog = self._load_catalog()
        self.company_lookup = self._build_company_lookup()

    def extract(self, text: str) -> ExtractionResult:
        text = text or ""
        found_tickers = []

        for ticker in self._find_cash_tag_tickers(text):
            if ticker not in found_tickers:
                found_tickers.append(ticker)

        for ticker in self._find_plain_tickers(text):
            if ticker not in found_tickers:
                found_tickers.append(ticker)

        lowered = text.lower()
        for alias, payload in self.company_lookup.items():
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, lowered):
                ticker = payload["ticker"]
                if ticker not in found_tickers:
                    found_tickers.append(ticker)

        found_companies = [self.catalog[ticker]["company_name"] or "Unknown" for ticker in found_tickers]

        return ExtractionResult(tickers=found_tickers, company_names=found_companies)

    def _find_cash_tag_tickers(self, text: str) -> list[str]:
        matches = re.findall(r"\$([A-Z]{1,5})\b", text)
        return [ticker for ticker in matches if ticker in self.catalog and ticker not in AMBIGUOUS_WORDS]

    def _find_plain_tickers(self, text: str) -> list[str]:
        matches = re.findall(r"\b([A-Z]{2,5})\b", text)
        return [ticker for ticker in matches if ticker in self.catalog and ticker not in AMBIGUOUS_WORDS]

    def _load_catalog(self) -> dict[str, dict[str, str]]:
        with self.ticker_catalog_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            catalog = {}
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                catalog[ticker] = {
                    "company_name": (row.get("company_name") or "").strip(),
                    "aliases": (row.get("aliases") or "").strip(),
                }
            return catalog

    def _build_company_lookup(self) -> dict[str, dict[str, str]]:
        company_lookup: dict[str, dict[str, str]] = {}
        for ticker, record in self.catalog.items():
            names = [record["company_name"]]
            aliases = [alias.strip() for alias in record["aliases"].split("|") if alias.strip()]
            names.extend(aliases)
            for name in names:
                company_lookup[name.lower()] = {
                    "ticker": ticker,
                    "company_name": record["company_name"],
                }
        return company_lookup
