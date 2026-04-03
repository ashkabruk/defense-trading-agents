from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class DefenseGovRssScanner(BaseScanner):
    """Scanner for defense.gov RSS headlines."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.url:
            return {"entries": []}
        response = await client.get(self.source.url, follow_redirects=True)
        response.raise_for_status()
        return {"rss": response.text}

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        rss = str(payload.get("rss") or "")
        parsed = feedparser.parse(rss)
        events: list[Event] = []
        for entry in parsed.entries:
            title = str(getattr(entry, "title", "Defense.gov update"))
            summary = str(getattr(entry, "summary", ""))
            link = str(getattr(entry, "link", ""))
            timestamp = datetime.now(timezone.utc)
            published = getattr(entry, "published_parsed", None)
            if published is not None:
                timestamp = datetime(*published[:6], tzinfo=timezone.utc)
            events.append(self._default_event(title, summary, link, dict(entry), timestamp=timestamp))
        return events
