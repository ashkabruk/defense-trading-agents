from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class DscaScanner(BaseScanner):
    """Scanner for DSCA major arms sales announcements."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.url:
            return {"html": ""}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = await client.get(self.source.url, headers=headers)
        response.raise_for_status()
        return {"html": response.text}

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        html = str(payload.get("html") or "")
        events: list[Event] = []

        link_pattern = re.compile(
            r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>[^<]{8,250})</a>',
            flags=re.IGNORECASE,
        )

        for match in link_pattern.finditer(html):
            title = " ".join(match.group("title").split())
            href = match.group("href")
            if "arms" not in title.lower() and "sale" not in title.lower() and "major" not in title.lower():
                continue
            url = href if href.startswith("http") else f"https://www.dsca.mil{href}"
            events.append(
                self._default_event(
                    headline=f"DSCA: {title}",
                    body=title,
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
                    headline="DSCA major arms sales page update",
                    body=snippet,
                    url=self.source.url or "",
                    raw_data={"raw": snippet},
                    timestamp=datetime.now(timezone.utc),
                )
            )

        return events
