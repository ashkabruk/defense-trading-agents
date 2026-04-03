from __future__ import annotations

from functools import lru_cache


class SentimentAnalyzer:
    """FinBERT-based sentiment with safe fallback when model deps are missing."""

    def __init__(self) -> None:
        self._classifier = self._load_classifier()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_classifier():
        try:
            from transformers import pipeline

            return pipeline(
                task="text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
            )
        except Exception:
            return None

    def analyze(self, text: str) -> float:
        """Return normalized sentiment score in [-1, 1]."""
        if not text.strip():
            return 0.0

        if self._classifier is None:
            return 0.0

        raw = self._classifier(text[:2000], top_k=None)

        # transformers output varies by version/model wrapper:
        # - list[dict] when top_k=None
        # - list[list[dict]] in some pipeline paths
        # - dict for single top label fallback
        if isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
            items = raw
        elif isinstance(raw, list) and raw and isinstance(raw[0], list):
            items = raw[0]
        else:
            return 0.0

        scores = {
            str(item.get("label", "")).lower(): float(item.get("score", 0.0))
            for item in items
            if isinstance(item, dict)
        }
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)
        return max(min(positive - negative, 1.0), -1.0)
