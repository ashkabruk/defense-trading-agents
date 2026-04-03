from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.models.config import Settings
from src.models.core import Event
from src.storage.fact_store import ChromaFactStore


class ImportanceScorer:
    """Computes a bounded [0, 1] score using sentiment magnitude, novelty, and staleness detection."""

    def __init__(self, settings: Settings, fact_store: ChromaFactStore) -> None:
        self._settings = settings
        self._fact_store = fact_store

    def score(self, event: Event) -> float:
        """
        Higher score means more likely to trigger discussion.
        Applies staleness penalty: events >2 hours old get 50% reduction, >24 hours skipped entirely.
        """
        # Check signal staleness
        now = datetime.now(timezone.utc)
        age = now - event.timestamp
        
        # Skip events older than 24 hours
        if age > timedelta(hours=24):
            return 0.0
        
        # Apply 50% penalty for events 2-24 hours old
        staleness_multiplier = 1.0
        if age > timedelta(hours=2):
            staleness_multiplier = 0.5
        
        sentiment_component = min(abs(event.sentiment_score), 1.0)

        novelty = 1.0
        similar = self._fact_store.search(event.headline, max_results=1)
        if similar and similar[0].get("distance") is not None:
            distance = float(similar[0]["distance"])
            novelty = max(0.0, min(distance, 1.0))

        source_weight = 1.0
        if event.source in {"sam_gov", "sec_edgar", "dsca_arms_sales"}:
            source_weight = 1.1

        raw_score = (0.55 * novelty) + (0.45 * sentiment_component)
        weighted = raw_score * source_weight * staleness_multiplier
        return round(max(0.0, min(weighted, 1.0)), 4)
