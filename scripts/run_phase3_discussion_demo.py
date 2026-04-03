from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.agents.orchestrator import DiscussionOrchestrator
from src.llm.client import LLMClient
from src.models.config import ModelConfig
from src.models.core import Event
from src.models.llm import ChatMessage, LLMResponse
from src.storage.db import SQLiteRepository
from src.storage.fact_store import ChromaFactStore


class MockDiscussionLLMClient(LLMClient):
    """Fallback client used only when DeepSeek credentials are unavailable."""

    async def chat(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        tools=None,
        response_format=None,
    ) -> LLMResponse:
        last_system = next((m.content for m in messages if m.role == "system"), "")
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        content = "Conviction 6, direction long."
        if "defense contracts analyst" in last_system.lower():
            content = (
                "The $2.3B F-35 sustainment award is material and supports recurring program cash flows. "
                "Defense.gov timing suggests near-term narrative support for LMT backlog quality. "
                "Conviction 8, direction long."
            )
        elif "macroeconomic analyst" in last_system.lower():
            content = (
                "Macro regime still supports stable defense appropriations despite rate uncertainty. "
                "Large sustainment contracts are less cyclical than new platform starts. "
                "Conviction 7, direction long."
            )
        elif "sentiment and narrative analyst" in last_system.lower():
            content = (
                "Headline is likely sentiment-positive for LMT today, but some impact may be priced in quickly. "
                "Still, contract size and Pentagon attribution improve credibility. "
                "Conviction 7, direction long."
            )
        elif "devil's advocate" in last_system.lower():
            content = (
                "Counterpoint: sustainment announcements are often expected and can be low-surprise. "
                "If this is a continuation award, upside may be limited after the first reaction. "
                "Conviction 6, direction abstain."
            )
        elif "risk manager" in last_system.lower():
            content = (
                "Approved with constraints: size at 5% notional, stop-loss 2%, take-profit 4%, hold <= 5 days. "
                "Sector correlation remains high across defense names. Conviction 7, direction long."
            )
        elif "output only valid json" in last_system.lower() or "tradeproposal" in last_user.lower():
            content = (
                '{"id":"demo-proposal-1","timestamp":"2026-04-03T00:00:00Z","ticker":"LMT",'
                '"direction":"long","conviction":7.0,"entry_rationale":"$2.3B F-35 sustainment contract '
                'supports Lockheed recurring defense revenue and sentiment.",'
                '"risk_factors":["Potentially priced-in news","Defense sector correlation"],'
                '"position_size_pct":5.0,"stop_loss_pct":2.0,"take_profit_pct":4.0,'
                '"max_holding_days":5,"source_events":["demo-event-1"],"agent_votes":{},'
                '"ragas_score":null,"status":"proposed"}'
            )

        return LLMResponse(
            model=model_config.model,
            provider=model_config.provider,
            content=content,
            finish_reason="stop",
            tool_calls=[],
            raw=None,
        )


async def run_demo(require_live: bool) -> None:
    repository = SQLiteRepository("data/trading.db")
    fact_store = ChromaFactStore("data/chroma")

    use_live = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if require_live and not use_live:
        raise RuntimeError("DEEPSEEK_API_KEY is required for --require-live mode")

    llm_client = LLMClient() if use_live else MockDiscussionLLMClient()
    orchestrator = DiscussionOrchestrator.from_config(
        config_dir="config",
        repository=repository,
        fact_store=fact_store,
        llm_client=llm_client,
    )

    event = Event(
        id="demo-event-1",
        source="defense_gov",
        timestamp=datetime.now(timezone.utc),
        headline="Lockheed Martin awarded $2.3B contract for F-35 sustainment by the Pentagon, announced on defense.gov today.",
        body="Lockheed Martin awarded $2.3B contract for F-35 sustainment by the Pentagon, announced on defense.gov today.",
        url="https://www.defense.gov/",
        entities=["Lockheed Martin", "Pentagon"],
        tickers=["LMT"],
        sentiment_score=0.6,
        importance_score=0.9,
        raw_data={"demo": True},
    )

    result = await orchestrator.discuss(event=event, rounds=2)

    mode = "LIVE_DEEPSEEK" if use_live else "MOCK_NO_DEEPSEEK_KEY"
    print(f"DISCUSSION_MODE {mode}")
    for msg in result.transcript:
        print(f"ROUND {msg['round']} | {msg['speaker']} ({msg['role']})")
        print(msg["content"])
        print("-" * 80)

    print("FINAL_PROPOSAL")
    print(result.proposal.model_dump_json(indent=2))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-live", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_demo(require_live=args.require_live))


if __name__ == "__main__":
    main()
