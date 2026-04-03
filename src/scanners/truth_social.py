from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class TruthSocialScanner(BaseScanner):
    """Scanner for Truth Social policy posts with keyword filtering."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.url:
            return {"rss": ""}
        response = await client.get(self.source.url)
        response.raise_for_status()
        return {"rss": response.text}

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        rss = str(payload.get("rss") or "")
        parsed = feedparser.parse(rss)
        events: list[Event] = []
        keywords = [item.lower() for item in self.source.keywords_filter]

        for entry in parsed.entries:
            title = str(getattr(entry, "title", "Truth Social post"))
            summary = str(getattr(entry, "summary", ""))
            link = str(getattr(entry, "link", self.source.url or ""))
            combined = f"{title}\n{summary}".lower()
            if keywords and not any(keyword in combined for keyword in keywords):
                continue

            timestamp = datetime.now(timezone.utc)
            published = getattr(entry, "published_parsed", None)
            if published is not None:
                timestamp = datetime(*published[:6], tzinfo=timezone.utc)

            events.append(self._default_event(title, summary, link, dict(entry), timestamp=timestamp))

        # Fallback when feed parser cannot extract entries from HTML proxy responses.
        if not events and rss:
            lowered = rss.lower()
            if not keywords or any(keyword in lowered for keyword in keywords):
                snippet = rss[:1500]
                events.append(
                    self._default_event(
                        headline="Truth Social feed update",
                        body=snippet,
                        url=self.source.url or "",
                        raw_data={"raw": snippet},
                        timestamp=datetime.now(timezone.utc),
                    )
                )

        return events
