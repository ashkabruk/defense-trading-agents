from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .agent import AgentVote


class Event(BaseModel):
    """Normalized event emitted by source scanners."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    timestamp: datetime
    headline: str
    body: str
    url: str
    entities: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    importance_score: float = Field(ge=0.0, le=1.0)
    raw_data: dict[str, object] = Field(default_factory=dict)


class TradeProposal(BaseModel):
    """Structured trade recommendation after agent discussion."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    ticker: str
    direction: Literal["long", "short"]
    conviction: float = Field(ge=1.0, le=10.0)
    entry_rationale: str
    risk_factors: list[str] = Field(default_factory=list)
    position_size_pct: float = Field(ge=0.0, le=100.0)
    stop_loss_pct: float = Field(gt=0.0, le=100.0)
    take_profit_pct: float = Field(gt=0.0, le=100.0)
    max_holding_days: int = Field(gt=0)
    source_events: list[str] = Field(default_factory=list)
    agent_votes: dict[str, AgentVote] = Field(default_factory=dict)
    ragas_score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: Literal["proposed", "approved", "rejected", "executed", "closed"] = "proposed"


class FailedClaim(BaseModel):
    """Unsupported or conflicting claim from evaluator."""

    claim: str
    reason: str
    supporting_doc_ids: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """RAGAS-inspired evaluation output for one proposal."""

    proposal_id: str
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    failed_claims: list[FailedClaim] = Field(default_factory=list)
    passed: bool


class Position(BaseModel):
    """Open or recently closed brokerage position."""

    ticker: str
    direction: Literal["long", "short"]
    shares: int = Field(gt=0)
    entry_price: float = Field(gt=0.0)
    entry_time: datetime
    stop_loss: float = Field(gt=0.0)
    take_profit: float = Field(gt=0.0)
    max_close_time: datetime
    current_price: float | None = Field(default=None, gt=0.0)
    unrealized_pnl: float | None = None
    proposal_id: str
