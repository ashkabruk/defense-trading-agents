from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class FredScanner(BaseScanner):
    """Scanner for FRED macro series observations."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.endpoint:
            return {"series": []}

        params = dict(self.source.params)
        api_key_env = params.pop("api_key_env", None)
        series_ids = params.pop("series_ids", [])

        import os

        api_key = os.environ.get(str(api_key_env), "") if api_key_env else ""

        series_payload: list[dict[str, Any]] = []
        for series_id in series_ids:
            query = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            response = await client.get(self.source.endpoint, params=query)
            response.raise_for_status()
            body = response.json()
            series_payload.append({"series_id": series_id, "payload": body})

        return {"series": series_payload}

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        events: list[Event] = []
        for series_item in self.ensure_list(payload, "series"):
            series_id = str(series_item.get("series_id") or "UNKNOWN")
            body = series_item.get("payload", {})
            observations = body.get("observations", []) if isinstance(body, dict) else []
            if not observations:
                continue
            latest = observations[0]
            value = latest.get("value", "")
            date_value = str(latest.get("date", ""))
            timestamp = datetime.now(timezone.utc)
            if date_value:
                try:
                    timestamp = datetime.fromisoformat(f"{date_value}T00:00:00+00:00")
                except ValueError:
                    pass
            headline = f"FRED update {series_id}: {value}"
            text = f"Latest observation for {series_id} on {date_value} is {value}."
            events.append(
                self._default_event(
                    headline=headline,
                    body=text,
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    raw_data=series_item,
                    timestamp=timestamp,
                )
            )
        return events
