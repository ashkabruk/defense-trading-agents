from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import httpx

from src.models.config import RiskConfig, Settings
from src.storage.db import SQLiteRepository
from src.storage.fact_store import ChromaFactStore

ToolFunc = Callable[..., Awaitable[dict[str, Any] | list[dict[str, Any]] | None]]


@dataclass(slots=True)
class ToolSpec:
    """Registered callable tool with schema metadata."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolFunc


@dataclass(slots=True)
class ToolContext:
    """Shared dependencies available to all tools."""

    repository: SQLiteRepository
    fact_store: ChromaFactStore
    settings: Settings
    risk: RiskConfig


class ToolRegistry:
    """In-memory tool registry used by discussion agents."""

    def __init__(self, context: ToolContext) -> None:
        self._context = context
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable[[ToolFunc], ToolFunc]:
        """Decorator-style tool registration."""

        def _decorator(func: ToolFunc) -> ToolFunc:
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
            )
            return func

        return _decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_specs(self, allowed_names: list[str]) -> list[ToolSpec]:
        return [spec for name, spec in self._tools.items() if name in allowed_names]

    async def invoke(self, name: str, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        spec = self.get(name)
        if spec is None:
            return {"error": f"Unknown tool: {name}"}
        return await spec.handler(**kwargs)


def create_default_tool_registry(context: ToolContext) -> ToolRegistry:
    """Create and populate default tool implementations for agent use."""
    registry = ToolRegistry(context)

    @registry.register(
        name="fact_store_search",
        description="Search the fact store for related events",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    )
    async def fact_store_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return context.fact_store.search(query=query, max_results=max_results)

    @registry.register(
        name="ticker_lookup",
        description="Map a company name to its stock ticker",
        parameters={
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"],
        },
    )
    async def ticker_lookup(company_name: str) -> dict[str, Any] | None:
        map_data = {
            "lockheed martin": "LMT",
            "raytheon": "RTX",
            "northrop grumman": "NOC",
            "general dynamics": "GD",
            "l3harris": "LHX",
            "boeing": "BA",
            "huntington ingalls": "HII",
            "transdigm": "TDG",
            "leidos": "LDOS",
            "saic": "SAIC",
            "kratos": "KTOS",
            "palantir": "PLTR",
        }
        lowered = company_name.lower().strip()
        for company, ticker in map_data.items():
            if lowered == company or lowered in company or company in lowered:
                return {"company": company.title(), "ticker": ticker}
        return None

    @registry.register(
        name="contract_history",
        description="Query past contracts for a company",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "months_back": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["ticker"],
        },
    )
    async def contract_history(ticker: str, months_back: int = 12) -> list[dict[str, Any]]:
        events = context.repository.fetch_contract_history(ticker=ticker.upper(), months_back=months_back)
        return [
            {
                "id": item.id,
                "timestamp": item.timestamp.isoformat(),
                "headline": item.headline,
                "source": item.source,
                "importance_score": item.importance_score,
            }
            for item in events[:20]
        ]

    @registry.register(
        name="portfolio_status",
        description="Get current portfolio positions and P&L",
        parameters={"type": "object", "properties": {}},
    )
    async def portfolio_status() -> dict[str, Any]:
        return {
            "environment": context.settings.environment,
            "max_open_positions": context.risk.max_open_positions,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "note": "IBKR execution layer not connected in Phase 3 demo",
        }

    @registry.register(
        name="fred_data_lookup",
        description="Get latest value for a FRED series",
        parameters={
            "type": "object",
            "properties": {"series_id": {"type": "string"}},
            "required": ["series_id"],
        },
    )
    async def fred_data_lookup(series_id: str) -> dict[str, Any]:
        source = next((s for s in context.settings.models if s == "scanner"), None)
        _ = source  # keep interface stable; series endpoint is configured in sources.yaml
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            return {"series_id": series_id, "error": "FRED_API_KEY not set"}

        endpoint = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            payload = response.json()

        obs = payload.get("observations", [])
        latest = obs[0] if obs else {}
        return {
            "series_id": series_id,
            "date": latest.get("date"),
            "value": latest.get("value"),
        }

    @registry.register(
        name="news_search",
        description="Search recent news for a topic",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "days_back": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["query"],
        },
    )
    async def news_search(query: str, days_back: int = 7) -> list[dict[str, Any]]:
        min_timestamp = datetime.now(timezone.utc) - timedelta(days=days_back)
        results = context.fact_store.search(query=query, max_results=15)
        filtered: list[dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            timestamp = metadata.get("timestamp") if isinstance(metadata, dict) else None
            if isinstance(timestamp, str):
                try:
                    parsed_ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if parsed_ts < min_timestamp:
                        continue
                except ValueError:
                    pass
            filtered.append(item)
        return filtered[:10]

    @registry.register(
        name="risk_calculator",
        description="Calculate risk metrics for a proposed trade",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "direction": {"type": "string", "enum": ["long", "short"]},
                "position_pct": {"type": "number"},
                "stop_loss_pct": {"type": "number"},
            },
            "required": ["ticker", "direction", "position_pct", "stop_loss_pct"],
        },
    )
    async def risk_calculator(
        ticker: str,
        direction: str,
        position_pct: float,
        stop_loss_pct: float,
    ) -> dict[str, Any]:
        max_loss_pct = round((position_pct * stop_loss_pct) / 100.0, 4)
        reward_pct = context.risk.default_take_profit_pct
        risk_reward = round(reward_pct / max(stop_loss_pct, 0.0001), 4)
        return {
            "ticker": ticker.upper(),
            "direction": direction,
            "position_pct": position_pct,
            "stop_loss_pct": stop_loss_pct,
            "estimated_max_portfolio_loss_pct": max_loss_pct,
            "default_take_profit_pct": reward_pct,
            "risk_reward_ratio": risk_reward,
            "within_position_limit": position_pct <= context.risk.max_position_pct,
        }

    @registry.register(
        name="materiality_calculator",
        description="Calculate contract value materiality relative to company fundamentals",
        parameters={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "contract_value": {"type": "number", "minimum": 0},
            },
            "required": ["ticker", "contract_value"],
        },
    )
    async def materiality_calculator(ticker: str, contract_value: float) -> dict[str, Any]:
        """Calculate materiality score and estimated stock price impact."""
        import json as json_module
        
        ticker_upper = ticker.upper()
        
        # Load company fundamentals
        try:
            with open("src/data/company_fundamentals.json", "r") as f:
                data = json_module.load(f)
                fundamentals_map = {f["ticker"]: f for f in data.get("fundamentals", [])}
        except Exception as e:
            return {
                "ticker": ticker_upper,
                "contract_value": contract_value,
                "error": f"Failed to load fundamentals: {str(e)}"
            }
        
        if ticker_upper not in fundamentals_map:
            return {
                "ticker": ticker_upper,
                "contract_value": contract_value,
                "error": f"Ticker {ticker_upper} not in fundamentals database"
            }
        
        fund = fundamentals_map[ticker_upper]
        annual_revenue = fund.get("annual_revenue", 1)
        market_cap = fund.get("market_cap", 1)
        
        # Calculate materiality ratios
        revenue_ratio = (contract_value / annual_revenue) if annual_revenue > 0 else 0
        
        # Estimate price impact using rough formula:
        # price_impact ≈ (contract_value / market_cap) * 100 * sensitivity_factor
        sensitivity_factor = 0.05  # 5% of market cap change = 100% stock move (rough estimate)
        estimated_price_impact_pct = (contract_value / market_cap * sensitivity_factor * 100) if market_cap > 0 else 0
        
        # Materiality score: 0-1 based on revenue ratio
        # >5% of annual revenue = very material (0.9)
        # 1-5% = material (0.6)
        # 0.1-1% = moderate (0.4)
        # <0.1% = low (0.1)
        if revenue_ratio > 0.05:
            materiality_score = 0.9
        elif revenue_ratio > 0.01:
            materiality_score = 0.6
        elif revenue_ratio > 0.001:
            materiality_score = 0.4
        else:
            materiality_score = 0.1
        
        return {
            "ticker": ticker_upper,
            "contract_value": round(contract_value, 2),
            "annual_revenue": fund.get("annual_revenue"),
            "market_cap": market_cap,
            "revenue_ratio_pct": round(revenue_ratio * 100, 4),
            "estimated_price_impact_pct": round(estimated_price_impact_pct, 4),
            "materiality_score": materiality_score,
            "materiality_level": "very_high" if materiality_score >= 0.8 else "high" if materiality_score >= 0.5 else "moderate" if materiality_score >= 0.3 else "low",
        }

    return registry


def format_tool_result_for_prompt(result: dict[str, Any] | list[dict[str, Any]] | None) -> str:
    """Serialize tool outputs consistently for LLM tool messages."""
    if result is None:
        return "null"
    return json.dumps(result, default=str)
