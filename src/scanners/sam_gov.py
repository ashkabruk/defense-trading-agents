from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.exceptions import ScannerFetchError
from src.models.core import Event
from src.scanners.base import BaseScanner, logger


class SamGovScanner(BaseScanner):
    """Scanner for SAM.gov opportunity feed."""

    async def fetch_with_retries(self, retries: int = 3, timeout_seconds: int = 20) -> dict[str, Any]:
        """SAM.gov-specific retry policy: 30s, 60s, 120s backoff to survive strict rate limits."""
        attempt = 0
        last_error: Exception | None = None
        backoff_seconds = [30, 60, 120]

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
                if attempt < retries:
                    await asyncio.sleep(backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)])

        raise ScannerFetchError(f"Failed to fetch {self.source.name}") from last_error

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.endpoint:
            return {"opportunitiesData": []}

        params = dict(self.source.params)
        api_key_env = params.pop("api_key_env", None)
        if api_key_env:
            import os

            api_key = os.environ.get(str(api_key_env), "")
            if api_key:
                params["api_key"] = api_key

        # Replace date placeholders
        if "postedFrom" in params and params["postedFrom"] == "{today_minus_1}":
            today_minus_1 = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            params["postedFrom"] = today_minus_1

        response = await client.get(self.source.endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        events: list[Event] = []
        for item in self.ensure_list(payload, "opportunitiesData"):
            title = str(item.get("title") or item.get("solicitationNumber") or "SAM Opportunity")
            description = str(item.get("description") or "")
            link = str(item.get("uiLink") or item.get("link") or "")
            posted = item.get("postedDate")
            timestamp = datetime.now(timezone.utc)
            if isinstance(posted, str):
                try:
                    timestamp = datetime.fromisoformat(posted.replace("Z", "+00:00"))
                except ValueError:
                    pass
            events.append(self._default_event(title, description, link, item, timestamp=timestamp))
        return events
