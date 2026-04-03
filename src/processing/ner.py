from __future__ import annotations

import re
from functools import lru_cache


DEFAULT_TICKER_MAP: dict[str, str] = {
    "lockheed": "LMT",
    "raytheon": "RTX",
    "northrop": "NOC",
    "general dynamics": "GD",
    "l3harris": "LHX",
    "boeing": "BA",
    "huntington ingalls": "HII",
    "transdigm": "TDG",
    "leidos": "LDOS",
    "saic": "SAIC",
    "kratos": "KTOS",
    "palantir": "PLTR",
}


class NERProcessor:
    """Entity extraction with spaCy when available and deterministic ticker mapping."""

    def __init__(self, ticker_map: dict[str, str] | None = None) -> None:
        self._ticker_map = ticker_map or DEFAULT_TICKER_MAP
        self._nlp = self._load_model()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        try:
            import spacy

            return spacy.load("en_core_web_sm")
        except Exception:
            return None

    def extract(self, text: str) -> tuple[list[str], list[str]]:
        """Return extracted entities and resolved tickers."""
        entities: list[str] = []
        tickers: set[str] = set()
        lowered = text.lower()

        for company, ticker in self._ticker_map.items():
            if company in lowered:
                entities.append(company.title())
                tickers.add(ticker)

        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"ORG", "GPE", "PERSON", "NORP"}:
                    value = ent.text.strip()
                    if value and value not in entities:
                        entities.append(value)

        for ticker in re.findall(r"\b[A-Z]{2,5}\b", text):
            if ticker in self._ticker_map.values():
                tickers.add(ticker)

        return entities, sorted(tickers)
