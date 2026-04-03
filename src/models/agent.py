from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents one tool invocation made by an agent."""

    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    result_summary: str | None = None


class AgentVote(BaseModel):
    """Per-agent vote emitted during discussion."""

    agent_name: str
    conviction: int = Field(ge=1, le=10)
    direction: str = Field(pattern="^(long|short|abstain)$")
    reasoning: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
