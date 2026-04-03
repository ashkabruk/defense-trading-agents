from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """One model provider configuration from settings.yaml."""

    provider: str
    model: str
    api_key_env: str
    base_url: str | None = None
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)


class Settings(BaseModel):
    """Global runtime settings loaded from config/settings.yaml."""

    environment: str = Field(pattern="^(paper|live)$")
    scan_interval_seconds: int = Field(gt=0)
    max_events_per_scan: int = Field(gt=0)
    max_queue_size: int = Field(default=5, gt=0)
    discussions_per_day: int = Field(gt=0)
    discussion_rounds: int = Field(gt=0)
    min_importance_score: float = Field(ge=0.0, le=1.0)
    escalation_threshold: float = Field(ge=0.0, le=1.0)
    ragas_min_score: float = Field(ge=0.0, le=1.0)
    ragas_max_retries: int = Field(ge=0)
    models: dict[str, ModelConfig]
    log_level: str = "INFO"
    log_format: str = Field(default="json", pattern="^(json|console)$")
    log_file: str


class SourceConfig(BaseModel):
    """One source entry from config/sources.yaml."""

    name: str
    enabled: bool = True
    type: str
    endpoint: str | None = None
    url: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int = Field(gt=0)
    parser: str
    keywords_filter: list[str] = Field(default_factory=list)


class SourcesConfig(BaseModel):
    """Container for all source entries."""

    sources: list[SourceConfig]


class AgentConfig(BaseModel):
    """Agent entry from config/agents.yaml."""

    name: str
    role: str
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    model_override: str | None = None
    enabled: bool = True


class OrchestratorConfig(BaseModel):
    """Orchestrator-specific settings from config/agents.yaml."""

    speaking_order: list[str] = Field(default_factory=list)
    summary_prompt: str
    output_format: str = "TradeProposal"


class AgentsConfig(BaseModel):
    """Container for agents and orchestrator config."""

    agents: list[AgentConfig]
    orchestrator: OrchestratorConfig


class IBKRConfig(BaseModel):
    """IBKR connection parameters from risk config."""

    host: str
    port_paper: int = Field(gt=0)
    port_live: int = Field(gt=0)
    client_id: int = Field(ge=0)
    account: str = ""
    read_only: bool = False


class RiskConfig(BaseModel):
    """Hard risk limits loaded from config/risk.yaml."""

    max_position_pct: float = Field(gt=0.0, le=100.0)
    max_daily_loss_pct: float = Field(gt=0.0, le=100.0)
    max_open_positions: int = Field(gt=0)
    max_sector_concentration_pct: float = Field(gt=0.0, le=100.0)
    max_holding_days: int = Field(gt=0)
    default_stop_loss_pct: float = Field(gt=0.0, le=100.0)
    default_take_profit_pct: float = Field(gt=0.0, le=100.0)
    min_conviction_score: int = Field(ge=1, le=10)
    min_ragas_score: float = Field(ge=0.0, le=1.0)
    cool_down_minutes: int = Field(ge=0)
    max_trades_per_day: int = Field(gt=0)
    allowed_tickers: list[str] = Field(default_factory=list)
    ibkr: IBKRConfig
