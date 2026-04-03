from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agents.tools import ToolRegistry, format_tool_result_for_prompt
from src.models.agent import AgentVote, ToolCall
from src.models.config import AgentConfig, ModelConfig
from src.models.llm import ChatMessage, ToolDefinition
from src.llm.client import LLMClient


@dataclass(slots=True)
class AgentTurnResult:
    """Single turn output from one agent."""

    message: str
    vote: AgentVote


class DiscussionAgent:
    """Runtime agent that uses LLM + allowed tools from config."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        model_config: ModelConfig,
        tool_registry: ToolRegistry,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.model_config = model_config
        self.tool_registry = tool_registry

    async def run_turn(
        self,
        event_text: str,
        transcript: list[dict[str, str]],
        round_index: int,
        model_config_override: ModelConfig | None = None,
    ) -> AgentTurnResult:
        """Run one discussion turn, executing tool calls when requested."""
        tool_specs = [
            ToolDefinition(name=s.name, description=s.description, parameters=s.parameters)
            for s in self.tool_registry.list_specs(self.config.tools)
        ]

        transcript_text = self._format_transcript(transcript)
        tool_defs_text = json.dumps([tool.model_dump() for tool in tool_specs], sort_keys=True)
        cacheable_prefix = [
            ChatMessage(role="system", content=self.config.system_prompt),
            ChatMessage(
                role="system",
                content=(
                    f"Agent persona: {self.config.name} ({self.config.role}). "
                    "Always provide a concise market view and explicitly state conviction 1-10 and "
                    "direction (long/short/abstain)."
                ),
            ),
            ChatMessage(
                role="system",
                content=f"Available tool definitions (JSON schema): {tool_defs_text}",
            ),
        ]
        variable_payload = ChatMessage(
            role="user",
            content=(
                f"Round {round_index}. Analyze this event and provide your view.\n"
                f"Event: {event_text}\n\n"
                "Discussion transcript so far:\n"
                f"{transcript_text}"
            ),
        )

        messages = [*cacheable_prefix, variable_payload]
        executed_tools: list[ToolCall] = []
        
        # Use override model if provided, otherwise use default
        active_model_config = model_config_override or self.model_config

        for iteration in range(5):
            response = await self.llm_client.chat(
                messages=messages,
                model_config=active_model_config,
                tools=tool_specs or None,
            )

            if not response.tool_calls:
                vote = self._extract_vote(response.content, executed_tools)
                return AgentTurnResult(message=response.content, vote=vote)

            assistant_tool_calls = []
            tool_results: list[tuple[str, str, str]] = []
            for call in response.tool_calls:
                result = await self.tool_registry.invoke(call.name, **call.arguments)
                result_text = format_tool_result_for_prompt(result)
                executed_tools.append(
                    ToolCall(name=call.name, arguments=call.arguments, result_summary=result_text[:500])
                )
                tool_call_id = call.id or f"tool_{call.name}"
                tool_results.append((call.name, tool_call_id, result_text))
                assistant_tool_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": format_tool_result_for_prompt(call.arguments),
                        },
                    }
                )

            messages.append(
                ChatMessage(role="assistant", content=response.content or "", tool_calls=assistant_tool_calls)
            )

            for call_name, tool_call_id, result_text in tool_results:
                messages.append(
                    ChatMessage(
                        role="tool",
                        name=call_name,
                        tool_call_id=tool_call_id,
                        content=result_text,
                    )
                )

            # If this is the last iteration before loop ends, force synthesis
            if iteration == 4:
                synthesis_response = await self.llm_client.chat(
                    messages=messages,
                    model_config=active_model_config,
                    tools=None,  # Disable tools to force synthesis
                )
                vote = self._extract_vote(synthesis_response.content, executed_tools)
                return AgentTurnResult(message=synthesis_response.content, vote=vote)

        fallback_message = "Unable to complete response after tool calls. Conviction 5, direction abstain."
        vote = self._extract_vote(fallback_message, executed_tools)
        return AgentTurnResult(message=fallback_message, vote=vote)

    def _format_transcript(self, transcript: list[dict[str, str]]) -> str:
        if not transcript:
            return "No previous discussion."
        return "\n".join([f"[{item['speaker']}] {item['content']}" for item in transcript])

    def _extract_vote(self, message: str, tool_calls: list[ToolCall]) -> AgentVote:
        conviction = 5
        direction = "abstain"

        conviction_match = re.search(r"conviction\D{0,8}(10|[1-9])", message, flags=re.IGNORECASE)
        if conviction_match:
            conviction = int(conviction_match.group(1))

        if re.search(r"\blong\b", message, flags=re.IGNORECASE):
            direction = "long"
        elif re.search(r"\bshort\b", message, flags=re.IGNORECASE):
            direction = "short"

        return AgentVote(
            agent_name=self.config.name,
            conviction=conviction,
            direction=direction,
            reasoning=message,
            tool_calls=tool_calls,
        )
