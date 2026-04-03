from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from src.agents.agent import DiscussionAgent
from src.agents.tools import ToolContext, ToolRegistry, create_default_tool_registry
from src.config.loader import load_config_bundle
from src.llm.client import LLMClient
from src.models.config import AgentsConfig
from src.models.config import AgentConfig, Settings, ModelConfig
from src.models.llm import ChatMessage
from src.models.agent import AgentVote
from src.models.core import Event, TradeProposal
from src.storage.db import SQLiteRepository
from src.storage.fact_store import ChromaFactStore


logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class DiscussionResult:
    """Complete result from a multi-round discussion."""

    transcript: list[dict[str, str]]
    proposal: TradeProposal


class DiscussionOrchestrator:
    """Runs configured agents in speaking order across multiple rounds."""

    def __init__(
        self,
        settings: Settings,
        agents_config: AgentsConfig,
        llm_client: LLMClient,
        tool_context: ToolContext,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.agents_config = agents_config
        self.llm_client = llm_client
        self.tool_registry = tool_registry or create_default_tool_registry(tool_context)
        self._agent_lookup = {agent.name: agent for agent in agents_config.agents if agent.enabled}

    @classmethod
    def from_config(
        cls,
        config_dir: str,
        repository: SQLiteRepository,
        fact_store: ChromaFactStore,
        llm_client: LLMClient | None = None,
    ) -> "DiscussionOrchestrator":
        """Build orchestrator directly from YAML configs."""
        bundle = load_config_bundle(config_dir)
        context = ToolContext(
            repository=repository,
            fact_store=fact_store,
            settings=bundle.settings,
            risk=bundle.risk,
        )
        return cls(
            settings=bundle.settings,
            agents_config=bundle.agents,
            llm_client=llm_client or LLMClient(),
            tool_context=context,
        )

    def _resolve_model_for_agent(self, agent_cfg: AgentConfig):
        if agent_cfg.model_override:
            override = self.settings.models.get(agent_cfg.model_override)
            if override is not None:
                return override
        return self.settings.models["discussion"]

    def _build_runtime_agent(self, agent_cfg: AgentConfig) -> DiscussionAgent:
        return DiscussionAgent(
            config=agent_cfg,
            llm_client=self.llm_client,
            model_config=self._resolve_model_for_agent(agent_cfg),
            tool_registry=self.tool_registry,
        )

    async def discuss(self, event: Event, rounds: int | None = None) -> DiscussionResult:
        """Run N rounds with early consensus detection and tiered model escalation."""
        total_rounds = rounds or self.settings.discussion_rounds

        discussion_model = self.settings.models["discussion"]
        transcript, votes = await self._run_rounds(
            event=event,
            model_config=discussion_model,
            start_round=1,
            end_round=1,
            transcript=None,
            votes=None,
        )

        convictions = [vote.conviction for vote in votes.values()]
        conviction_stdev = self._calculate_stdev(convictions)
        if conviction_stdev < 1.5:
            logger.info(
                "early_consensus_detected",
                stdev=conviction_stdev,
                rounds_executed=1,
            )
            proposal = await self._summarize_proposal(
                event=event,
                transcript=transcript,
                votes=votes,
                model_config=discussion_model,
            )
            return DiscussionResult(transcript=transcript, proposal=proposal)

        avg_conviction = (sum(convictions) / len(convictions)) if convictions else 0.0
        should_escalate = (
            avg_conviction >= 7.0
            and float(event.importance_score) >= float(self.settings.escalation_threshold)
        )

        active_model = discussion_model
        if should_escalate:
            premium_model = self.settings.models.get("premium")
            if premium_model is not None:
                active_model = premium_model
                logger.info(
                    "discussion_escalated_to_premium",
                    avg_conviction=avg_conviction,
                    event_importance=float(event.importance_score),
                    escalation_threshold=float(self.settings.escalation_threshold),
                    rounds=total_rounds,
                )
                # Re-run full discussion with premium model to improve quality on high-signal events.
                transcript, votes = await self._run_rounds(
                    event=event,
                    model_config=active_model,
                    start_round=1,
                    end_round=total_rounds,
                    transcript=None,
                    votes=None,
                )
            else:
                logger.warning("premium_model_missing_for_escalation")
                if total_rounds > 1:
                    transcript, votes = await self._run_rounds(
                        event=event,
                        model_config=active_model,
                        start_round=2,
                        end_round=total_rounds,
                        transcript=transcript,
                        votes=votes,
                    )
        else:
            if total_rounds > 1:
                transcript, votes = await self._run_rounds(
                    event=event,
                    model_config=active_model,
                    start_round=2,
                    end_round=total_rounds,
                    transcript=transcript,
                    votes=votes,
                )

        proposal = await self._summarize_proposal(
            event=event,
            transcript=transcript,
            votes=votes,
            model_config=active_model,
        )
        return DiscussionResult(transcript=transcript, proposal=proposal)

    async def _run_rounds(
        self,
        event: Event,
        model_config: ModelConfig,
        start_round: int,
        end_round: int,
        transcript: list[dict[str, str]] | None,
        votes: dict[str, AgentVote] | None,
    ) -> tuple[list[dict[str, str]], dict[str, AgentVote]]:
        active_transcript = list(transcript or [])
        active_votes = dict(votes or {})
        speaking_order = self.agents_config.orchestrator.speaking_order
        event_text = f"{event.headline}\n{event.body}\nSource: {event.source} ({event.url})"

        for round_index in range(start_round, end_round + 1):
            agent_tasks = []
            agent_names_in_round = []

            for agent_name in speaking_order:
                agent_cfg = self._agent_lookup.get(agent_name)
                if agent_cfg is None:
                    continue

                runtime_agent = self._build_runtime_agent(agent_cfg)
                agent_tasks.append(
                    runtime_agent.run_turn(
                        event_text=event_text,
                        transcript=active_transcript,
                        round_index=round_index,
                        model_config_override=model_config,
                    )
                )
                agent_names_in_round.append(agent_name)

            turns = await asyncio.gather(*agent_tasks)

            for agent_name, turn in zip(agent_names_in_round, turns):
                agent_cfg = self._agent_lookup.get(agent_name)
                if agent_cfg is None:
                    continue
                active_transcript.append(
                    {
                        "round": str(round_index),
                        "speaker": agent_cfg.name,
                        "role": agent_cfg.role,
                        "content": turn.message,
                    }
                )
                active_votes[agent_cfg.name] = turn.vote

        return active_transcript, active_votes

    async def _summarize_proposal(
        self,
        event: Event,
        transcript: list[dict[str, str]],
        votes: dict[str, AgentVote],
        model_config: ModelConfig | None = None,
    ) -> TradeProposal:
        prefix_instruction = (
            "You output only valid JSON for TradeProposal. "
            "Follow the required schema exactly and provide no markdown or prose."
        )
        summary_instruction = (
            f"{self.agents_config.orchestrator.summary_prompt}\n\n"
            "Output a valid TradeProposal JSON object with fields: "
            "timestamp, ticker, direction (long/short), conviction (1-10), entry_rationale, risk_factors (list), "
            "position_size_pct, stop_loss_pct, take_profit_pct, max_holding_days."
        )
        variable_payload = (
            f"Event:\n{event.model_dump_json(indent=2)}\n\n"
            "Transcript:\n"
            f"{json.dumps(transcript, indent=2)}"
        )

        # Use provided model_config or default to discussion model
        active_model = model_config or self.settings.models["discussion"]

        try:
            response = await self.llm_client.chat(
                messages=[
                    ChatMessage(role="system", content=prefix_instruction),
                    ChatMessage(role="system", content=summary_instruction),
                    ChatMessage(role="user", content=variable_payload),
                ],
                model_config=active_model,
            )
            parsed = TradeProposal.model_validate_json(response.content)
            return parsed.model_copy(update={"agent_votes": votes, "source_events": [event.id]})
        except Exception as e:
            logger.warning("orchestrator_proposal_parse_error", error=str(e))
            conviction_values = [vote.conviction for vote in votes.values()]
            avg_conviction = sum(conviction_values) / len(conviction_values) if conviction_values else 5.0
            default_direction = "long"
            direction_votes = [vote.direction for vote in votes.values()]
            if direction_votes and direction_votes.count("short") > direction_votes.count("long"):
                default_direction = "short"

            ticker = event.tickers[0] if event.tickers else "LMT"
            return TradeProposal(
                timestamp=datetime.now(timezone.utc),
                ticker=ticker,
                direction=default_direction,
                conviction=max(1.0, min(float(avg_conviction), 10.0)),
                entry_rationale=event.headline,
                risk_factors=["Model fallback summary used"],
                position_size_pct=5.0,
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                max_holding_days=5,
                source_events=[event.id],
                agent_votes=votes,
                status="proposed",
            )

    def _calculate_stdev(self, values: list[float]) -> float:
        """Calculate standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        try:
            return statistics.stdev(values)
        except Exception:
            return 0.0
