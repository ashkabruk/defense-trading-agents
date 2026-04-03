from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class CapitolTradesScanner(BaseScanner):
    """Scanner for congressional defense-sector trade disclosures."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.url:
            return {"html": ""}
        response = await client.get(self.source.url)
        response.raise_for_status()
        return {"html": response.text}

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        html = str(payload.get("html") or "")
        events: list[Event] = []

        # Pull candidate links and nearby text blocks that often contain issuer/trade info.
        pattern = re.compile(
            r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>[^<]{8,200})</a>',
            flags=re.IGNORECASE,
        )

        for match in pattern.finditer(html):
            title = " ".join(match.group("title").split())
            if not title:
                continue
            href = match.group("href")
            url = href if href.startswith("http") else f"https://www.capitoltrades.com{href}"
            if "/trades/" not in url and "trade" not in title.lower():
                continue
            body = f"Congressional trade disclosure candidate: {title}"
            events.append(
                self._default_event(
                    headline=f"Capitol Trades: {title}",
                    body=body,
                    url=url,
                    raw_data={"title": title, "url": url},
                    timestamp=datetime.now(timezone.utc),
                )
            )
            if len(events) >= 25:
                break

        if not events and html:
            snippet = re.sub(r"\s+", " ", html)[:1200]
            events.append(
                self._default_event(
                    headline="Capitol Trades page update",
                    body=snippet,
                    url=self.source.url or "",
                    raw_data={"raw": snippet},
                    timestamp=datetime.now(timezone.utc),
                )
            )

        return events
