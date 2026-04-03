from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import structlog

from src.exceptions import ScannerFetchError, ScannerParseError
from src.models.config import Settings, SourceConfig
from src.models.core import Event
from src.processing.ner import NERProcessor
from src.processing.scoring import ImportanceScorer
from src.processing.sentiment import SentimentAnalyzer
from src.storage.db import SQLiteRepository
from src.storage.fact_store import ChromaFactStore

logger = structlog.get_logger(__name__)


class BaseScanner(ABC):
    """Shared scanner workflow for fetch/parse/filter/store."""

    def __init__(
        self,
        source: SourceConfig,
        settings: Settings,
        repository: SQLiteRepository,
        fact_store: ChromaFactStore,
        sentiment: SentimentAnalyzer,
        ner: NERProcessor,
        scorer: ImportanceScorer,
    ) -> None:
        self.source = source
        self.settings = settings
        self.repository = repository
        self.fact_store = fact_store
        self.sentiment = sentiment
        self.ner = ner
        self.scorer = scorer

    async def run(self, warmup: bool = False) -> list[Event]:
        """Execute scanner pipeline and return events above threshold.

        When warmup=True, new events are persisted but not queued for discussion.
        """
        raw_payload = await self.fetch_with_retries()
        parsed = self.parse(raw_payload)

        queued_events: list[Event] = []
        for event in parsed[: self.settings.max_events_per_scan]:
            if self.repository.event_exists(event):
                continue

            sentiment_score = self.sentiment.analyze(f"{event.headline}\n\n{event.body}")
            entities, tickers = self.ner.extract(f"{event.headline}\n{event.body}")

            enriched = event.model_copy(
                update={
                    "sentiment_score": sentiment_score,
                    "entities": entities,
                    "tickers": tickers,
                }
            )
            importance = self.scorer.score(enriched)
            enriched = enriched.model_copy(update={"importance_score": importance})

            inserted = self.repository.save_event(enriched)
            if not inserted:
                continue

            self.fact_store.add_event(enriched)
            if not warmup and enriched.importance_score >= self.settings.min_importance_score:
                queued_events.append(enriched)

        logger.info(
            "scanner_completed",
            source=self.source.name,
            scanned=len(parsed),
            queued=len(queued_events),
        )
        return queued_events

    async def fetch_with_retries(self, retries: int = 3, timeout_seconds: int = 20) -> dict[str, Any]:
        """Fetch source payload with retry and timeout behavior."""
        attempt = 0
        last_error: Exception | None = None
        while attempt < retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    return await self.fetch(client)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "scanner_fetch_retry",
                    source=self.source.name,
                    attempt=attempt,
                    error=str(exc),
                )
                # Exponential backoff: 2s, 4s, 8s, ... capped at 30s
                await asyncio.sleep(min(2**attempt, 30))

        raise ScannerFetchError(f"Failed to fetch {self.source.name}") from last_error

    def _default_event(
        self,
        headline: str,
        body: str,
        url: str,
        raw_data: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> Event:
        """Build default event shell for parser implementations."""
        return Event(
            id=str(uuid4()),
            source=self.source.name,
            timestamp=timestamp or datetime.now(timezone.utc),
            headline=headline,
            body=body,
            url=url,
            entities=[],
            tickers=[],
            sentiment_score=0.0,
            importance_score=0.0,
            raw_data=raw_data,
        )

    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Fetch raw payload from source endpoint."""

    @abstractmethod
    def parse(self, payload: dict[str, Any]) -> list[Event]:
        """Convert payload into normalized Event objects."""

    def ensure_list(self, payload: Any, key: str) -> list[dict[str, Any]]:
        """Utility to validate expected list payload shape."""
        value = payload.get(key, []) if isinstance(payload, dict) else []
        if not isinstance(value, list):
            raise ScannerParseError(f"Expected list at key '{key}' for source {self.source.name}")
        return [item for item in value if isinstance(item, dict)]
