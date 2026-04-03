from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.models.core import Event
from src.scanners.base import BaseScanner


class SecEdgarScanner(BaseScanner):
    """Scanner for SEC EDGAR EFTS full-text search endpoint."""

    async def fetch(self, client: httpx.AsyncClient) -> dict[str, Any]:
        if not self.source.endpoint:
            return {"hits": {"hits": []}}

        tickers = self.source.params.get("tickers", [])
        forms = self.source.params.get("forms", "4,8-K,10-Q")
        if isinstance(forms, list):
            forms = ",".join(forms)
        else:
            forms = str(forms)

        # Calculate date range (last 7 days)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        start_dt = start_date.strftime("%Y-%m-%d")
        end_dt = end_date.strftime("%Y-%m-%d")

        # Use GET endpoint for EFTS full-text search with browser-like User-Agent
        search_query = " OR ".join(tickers)
        params = {
            "q": search_query,
            "forms": forms,
            "dateRange": "custom",
            "startdt": start_dt,
            "enddt": end_dt,
        }

        headers = {
            "User-Agent": "DefenseTrader/1.0 (defense-trader@example.com)"
        }

        response = await client.get(self.source.endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def parse(self, payload: dict[str, Any]) -> list[Event]:
        hits = payload.get("hits", {}) if isinstance(payload, dict) else {}
        items = hits.get("hits", []) if isinstance(hits, dict) else []

        events: list[Event] = []
        for item in items:
            source = item.get("_source", {}) if isinstance(item, dict) else {}
            filed_at = str(source.get("filedAt") or "")
            form = str(source.get("form") or "FILING")
            ticker = str(source.get("ticker") or "")
            company = str(source.get("displayNames") or source.get("companyName") or ticker)
            title = f"SEC {form} filing: {company}"
            description = str(source.get("description") or "")
            link = str(source.get("linkToFilingDetails") or source.get("linkToTxt") or "")
            timestamp = datetime.now(timezone.utc)
            if filed_at:
                try:
                    timestamp = datetime.fromisoformat(filed_at.replace("Z", "+00:00"))
                except ValueError:
                    pass
            events.append(self._default_event(title, description, link, source, timestamp=timestamp))
        return events
