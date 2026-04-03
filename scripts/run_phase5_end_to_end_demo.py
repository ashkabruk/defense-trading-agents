from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.agents.orchestrator import DiscussionOrchestrator
from src.config.loader import load_config_bundle
from src.evaluation.ragas import RAGASEvaluator
from src.execution import IBKRConnector, OrderManager, RiskManager
from src.llm.client import LLMClient
from src.models.config import ModelConfig
from src.models.core import Event, TradeProposal
from src.models.llm import ChatMessage, LLMResponse
from src.processing import ImportanceScorer, NERProcessor, SentimentAnalyzer
from src.storage import ChromaFactStore, SQLiteRepository


class DeterministicDemoLLMClient(LLMClient):
    """Deterministic LLM for offline-safe end-to-end demo runs."""

    async def chat(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        tools=None,
        response_format=None,
    ) -> LLMResponse:
        system_text = "\n".join(msg.content for msg in messages if msg.role == "system").lower()
        user_text = "\n".join(msg.content for msg in messages if msg.role == "user").lower()

        # Agent turns
        if "defense contracts specialist" in system_text:
            content = (
                "NOC winning a $4.8B next-gen ICBM award is material and likely revenue-accretive over multi-year horizons. "
                "This is a competitive win versus LMT and supports a long thesis. Conviction 9, direction long."
            )
        elif "macroeconomic analyst" in system_text:
            content = (
                "Geopolitical backdrop and defense appropriations remain supportive for strategic deterrence programs. "
                "Budget risk exists but probability-adjusted impact remains positive. Conviction 8, direction long."
            )
        elif "sentiment and narrative analyst" in system_text:
            content = (
                "Headline surprise and competitive angle versus Lockheed should drive near-term positive sentiment in NOC. "
                "Some move may be priced quickly, but narrative remains constructive. Conviction 8, direction long."
            )
        elif "devil's advocate" in system_text:
            content = (
                "Counterpoint: contract timing and execution risk can delay earnings realization, and market may fade initial hype. "
                "Still no hard contradiction to long bias. Conviction 6, direction abstain."
            )
        elif "risk manager" in system_text:
            content = (
                "Risk view: allow trade if capped to configured max position, with stop-loss 2% and take-profit 4%, "
                "holding <= 10 days and no daily loss limit breach. Conviction 7, direction long."
            )
        # Proposal synthesis
        elif "you output only valid json for tradeproposal" in system_text:
            proposal = {
                "id": "phase5-demo-proposal-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": "NOC",
                "direction": "long",
                "conviction": 8.2,
                "entry_rationale": (
                    "Northrop Grumman won a $4.8B next-gen ICBM contract from the US Air Force, "
                    "beating Lockheed Martin in a major strategic award."
                ),
                "risk_factors": [
                    "Execution/timeline risk on large defense programs",
                    "Potential near-term overreaction in price",
                ],
                "position_size_pct": 5.0,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 4.0,
                "max_holding_days": 7,
                "source_events": ["phase5-demo-event-1"],
                "agent_votes": {},
                "ragas_score": None,
                "status": "proposed",
            }
            content = json.dumps(proposal)
        # RAGAS claim decomposition
        elif "extract all explicit and implicit claims" in user_text:
            content = json.dumps(
                {
                    "claims": [
                        "Northrop Grumman was awarded a $4.8B next-gen ICBM contract by the US Air Force.",
                        "Northrop Grumman beat Lockheed Martin in the bid competition.",
                        "The award is likely material for Northrop Grumman's defense backlog and revenue outlook.",
                    ]
                }
            )
        # RAGAS scoring
        elif "score" in user_text and "json" in user_text:
            if "relevant" in user_text or "relevance" in user_text:
                content = json.dumps({"score": 0.86, "explanation": "Retrieved docs are highly relevant"})
            else:
                content = json.dumps({"score": 0.84, "explanation": "Claim is well-supported"})
        else:
            content = "Conviction 6, direction long."

        return LLMResponse(
            model=model_config.model,
            provider=model_config.provider,
            content=content,
            finish_reason="stop",
            tool_calls=[],
            raw=None,
        )


async def main() -> None:
    print("=" * 100)
    print("PHASE 5 END-TO-END DEMO")
    print("Pipeline: scanner ingest -> FinBERT -> agent discussion -> RAGAS -> risk -> IBKR paper order")
    print("=" * 100)

    # Setup shared dependencies
    bundle = load_config_bundle("config")
    repository = SQLiteRepository("data/trading.db")
    fact_store = ChromaFactStore("data/chroma")
    sentiment = SentimentAnalyzer()
    ner = NERProcessor()
    scorer = ImportanceScorer(bundle.settings, fact_store)
    llm_client = DeterministicDemoLLMClient()

    # Single IB Gateway connection per requirement.
    ibkr = IBKRConnector(
        mode="paper",
        host=bundle.risk.ibkr.host,
        port=4002,
        client_id=1,
    )

    print("\n[STEP 1] Create fake high-conviction event")
    raw_headline = (
        "Northrop Grumman awarded $4.8B next-gen ICBM contract by US Air Force, "
        "beating Lockheed Martin bid"
    )
    raw_body = (
        "The U.S. Air Force awarded Northrop Grumman a $4.8 billion next-generation ICBM contract. "
        "The award was reported as a competitive win over Lockheed Martin's bid."
    )

    entities, tickers = ner.extract(f"{raw_headline}\n{raw_body}")
    sentiment_score = sentiment.analyze(f"{raw_headline}\n{raw_body}")

    event = Event(
        id="phase5-demo-event-1",
        source="defense_gov",
        timestamp=datetime.now(timezone.utc),
        headline=raw_headline,
        body=raw_body,
        url="https://example.local/phase5/noc-icbm-award",
        entities=entities,
        tickers=tickers or ["NOC", "LMT"],
        sentiment_score=sentiment_score,
        importance_score=0.0,
        raw_data={"demo": True},
    )
    event.importance_score = scorer.score(event)

    print(f"  Event ID: {event.id}")
    print(f"  Entities: {event.entities}")
    print(f"  Tickers: {event.tickers}")
    print(f"  FinBERT sentiment: {event.sentiment_score:.3f}")
    print(f"  Importance score: {event.importance_score:.3f}")

    print("\n[STEP 2] Scanner/storage stage: persist to SQLite + Chroma")
    inserted = repository.save_event(event)
    if inserted:
        print("  SQLite: event inserted")
    else:
        print("  SQLite: duplicate event detected, skipped insert")
    fact_store.add_event(event)
    print("  Chroma: event embedded and stored")
    print(f"  SQLite total event count: {repository.get_event_count()}")

    print("\n[STEP 3] Agent discussion stage")
    orchestrator = DiscussionOrchestrator.from_config(
        config_dir="config",
        repository=repository,
        fact_store=fact_store,
        llm_client=llm_client,
    )
    discussion = await orchestrator.discuss(event=event, rounds=2)

    for msg in discussion.transcript:
        print(f"  Round {msg['round']} | {msg['speaker']} -> {msg['content']}")

    proposal: TradeProposal = discussion.proposal
    print("\n  Proposed trade:")
    print(f"    ticker={proposal.ticker} direction={proposal.direction} conviction={proposal.conviction:.2f}")
    print(
        f"    size={proposal.position_size_pct:.2f}% stop={proposal.stop_loss_pct:.2f}% "
        f"take={proposal.take_profit_pct:.2f}% hold_days={proposal.max_holding_days}"
    )

    print("\n[STEP 4] RAGAS evaluation stage")
    evaluator = RAGASEvaluator(
        llm_client=llm_client,
        model_config=bundle.settings.models["discussion"],
        fact_store=fact_store,
    )
    ragas = await evaluator.evaluate(proposal, context_limit=8)
    proposal = proposal.model_copy(update={"ragas_score": ragas.ragas_score})
    repository.save_trade_proposal(proposal)

    print(f"  Claims evaluated: {len(ragas.claim_contexts)}")
    print(f"  Faithfulness: {ragas.average_faithfulness:.3f}")
    print(f"  Relevance:    {ragas.average_relevance:.3f}")
    print(f"  Consistency:  {ragas.average_consistency:.3f}")
    print(f"  RAGAS score:  {ragas.ragas_score:.3f}")
    print(f"  Threshold:    {bundle.risk.min_ragas_score:.3f}")

    print("\n[STEP 5] IBKR connect + last-price retrieval (single Gateway connection, clientId=1)")
    connected = await ibkr.connect()
    if not connected:
        print("  ERROR: Could not connect to IB Gateway on localhost:4002")
        return
    print("  Connected to IB Gateway")

    try:
        # Use last price for order pricing per requirement.
        try:
            last_price = await ibkr.get_quote(proposal.ticker)
            print(f"  Last price for {proposal.ticker}: {last_price:.4f}")
        except Exception as exc:
            print(f"  Quote unavailable for {proposal.ticker}: {exc}")
            print("  Cannot continue to risk/order placement without last price.")
            return

        print("\n[STEP 6] Risk manager checks (enforcing config/risk.yaml)")
        risk_manager = RiskManager(bundle.risk, ibkr)
        approved, reasons = await risk_manager.validate_trade_for_risk(
            proposal=proposal,
            current_price=last_price,
            account_value=await ibkr.get_account_value(),
        )

        ragas_pass = proposal.ragas_score is not None and proposal.ragas_score >= bundle.risk.min_ragas_score
        print(f"  RAGAS pass: {ragas_pass}")
        print(f"  Risk pass:  {approved}")
        if not approved:
            for reason in reasons:
                print(f"    - {reason}")

        if not ragas_pass or not approved:
            print("\n[STEP 7] Decision: REJECTED (no order placed)")
            return

        sizing = await risk_manager.calculate_position_size(
            proposal=proposal,
            current_price=last_price,
            account_value=await ibkr.get_account_value(),
        )
        if sizing is None:
            print("\n[STEP 7] Decision: REJECTED (position sizing failed)")
            return

        print("\n[STEP 7] Decision: APPROVED -> place paper bracket order")
        print(
            f"  Shares: {sizing.shares} | account_pct={sizing.account_pct:.3f}% "
            f"| stop_loss_price={sizing.stop_loss_price:.4f}"
        )

        order_manager = OrderManager(ibkr)
        entry_side = "BUY" if proposal.direction == "long" else "SELL"
        entry_order_id = await order_manager.place_bracket_order(
            ticker=proposal.ticker,
            side=entry_side,
            quantity=sizing.shares,
            entry_limit_price=last_price,
            stop_loss_pct=proposal.stop_loss_pct,
            take_profit_pct=proposal.take_profit_pct,
        )

        print(f"  Entry order id: {entry_order_id}")
        await order_manager.monitor_brackets()
        bracket_summary = order_manager.get_bracket_summary()
        print(f"  Bracket status summary: {bracket_summary}")

        print("\n[STEP 8] Final result")
        print("  Trade placed on IBKR paper account.")
    finally:
        await ibkr.disconnect()
        print("\nDisconnected from IB Gateway.")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
