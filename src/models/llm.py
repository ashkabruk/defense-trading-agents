from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Portable chat message used by all providers."""

    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, object]] | None = None


class ToolDefinition(BaseModel):
    """Simplified tool schema used for provider tool-calling."""

    name: str
    description: str
    parameters: dict[str, object] = Field(default_factory=dict)


class ToolInvocation(BaseModel):
    """Tool call emitted by the model."""

    id: str | None = None
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Unified output from any provider."""

    model: str
    provider: str
    content: str
    finish_reason: str | None = None
    tool_calls: list[ToolInvocation] = Field(default_factory=list)
    raw: dict[str, object] | None = None
