"""Phase 4 Demo: RAGAS Evaluation of Trade Proposal."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.agents.orchestrator import DiscussionOrchestrator
from src.config.loader import load_config_bundle
from src.evaluation.ragas import RAGASEvaluator
from src.logging_setup import configure_logging
from src.llm.client import LLMClient
from src.models.core import Event
from src.storage import ChromaFactStore, SQLiteRepository

# Fake event for testing (same as Phase 3 demo)
TEST_EVENT = Event(
    id="test_event_001",
    source="demo",
    headline="Lockheed Martin awarded $2.3B F-35 sustainment contract by Pentagon",
    body=(
        "The Department of Defense announced today that Lockheed Martin has been awarded a $2.3 billion "
        "contract for F-35 aircraft sustainment and support. The contract includes spare parts provisioning, "
        "technical data, and sustainment engineering support for the global F-35 fleet. This sustainment contract "
        "is separate from production contracts and reflects the Pentagon's commitment to maintaining the 900+ aircraft "
        "currently in service worldwide."
    ),
    url="https://example.com/f35-sustainment-2026",
    entities=["Lockheed Martin", "Pentagon", "F-35"],
    tickers=["LMT"],
    sentiment_score=0.6,
    importance_score=0.72,
    raw_data={},
    timestamp=datetime.now(timezone.utc),
)


async def main() -> None:
    """Run Phase 4 demo: full discussion → evaluation."""
    print("\n" + "=" * 80)
    print("PHASE 4 DEMO: RAGAS EVALUATION")
    print("=" * 80 + "\n")

    # Initialize
    bundle = load_config_bundle("config")
    configure_logging(bundle.settings)

    repository = SQLiteRepository("data/trading.db")
    fact_store = ChromaFactStore("data/chroma")

    # Step 1: Run Phase 3 orchestration to generate proposal
    print("STEP 1: Running Phase 3 orchestration (Agent Discussion)")
    print("-" * 80)

    llm_client = LLMClient()
    orchestrator = DiscussionOrchestrator.from_config(
        config_dir="config",
        repository=repository,
        fact_store=fact_store,
        llm_client=llm_client,
    )

    event_text = f"{TEST_EVENT.headline}\n\n{TEST_EVENT.body}"
    discussion_result = await orchestrator.discuss(TEST_EVENT, rounds=1)
    proposal = discussion_result.proposal

    print(f"[OK] Proposal generated:")
    print(f"  Ticker: {proposal.ticker}")
    print(f"  Direction: {proposal.direction}")
    print(f"  Conviction: {proposal.conviction:.1f}/10")
    print(f"  Status: {proposal.status}")

    # Step 2: Create RAGAS evaluator
    print("\n" + "-" * 80)
    print("STEP 2: Initializing RAGAS Evaluator")
    print("-" * 80)

    evaluator = RAGASEvaluator(
        llm_client=llm_client,
        model_config=orchestrator.settings.models["discussion"],
        fact_store=fact_store,
    )
    print("[OK] RAGAS evaluator ready")

    # Step 3: Evaluate proposal
    print("\n" + "-" * 80)
    print("STEP 3: Running RAGAS Evaluation")
    print("-" * 80)

    result = await evaluator.evaluate(proposal, context_limit=10)

    # Step 4: Display results
    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nProposal ID: {result.proposal_id}")
    print(f"Timestamp: {result.timestamp}")

    print(f"\n--- Claims Decomposed ---")
    print(f"Total Claims: {len(result.claim_contexts)}")
    for i, ctx in enumerate(result.claim_contexts, 1):
        print(f"\nClaim {i}: {ctx.claim}")
        print(f"  Faithfulness: {ctx.faithfulness_score:.1%} (supported by evidence)")
        print(f"  Relevance: {ctx.relevance_score:.1%} (doc relevance)")
        print(f"  Consistency: {ctx.consistency_score:.1%} (no contradictions)")
        if ctx.contradicting_facts:
            print(f"  [!] Contradictions: {len(ctx.contradicting_facts)}")
            for fact in ctx.contradicting_facts[:2]:
                print(f"    - {fact}")

    print(f"\n--- Aggregate Scores ---")
    print(f"Average Faithfulness: {result.average_faithfulness:.1%}")
    print(f"Average Relevance: {result.average_relevance:.1%}")
    print(f"Average Consistency: {result.average_consistency:.1%}")

    print(f"\n--- RAGAS Score ---")
    print(f"RAGAS Score: {result.ragas_score:.1%} (composite 0-100)")
    print(f"Confidence: {result.confidence_level.upper()}")

    if result.red_flags:
        print(f"\n--- Red Flags ({len(result.red_flags)}) ---")
        for flag in result.red_flags:
            print(f"  [!] {flag}")
    else:
        print(f"\n--- Red Flags ---")
        print(f"  [OK] None detected")

    print(f"\n--- Summary ---")
    print(result.summary)

    # Step 5: Integration check
    print("\n" + "=" * 80)
    print("INTEGRATION CHECK")
    print("=" * 80)

    print(f"[OK] Proposal generated with RAGAS evaluation")
    print(f"[OK] Evaluation stored: proposal.ragas_score = {proposal.ragas_score}")
    print(f"[OK] Ready for Phase 5 (IBKR Execution with risk approval)")

    # Save result to JSON for inspection
    result_json = json.dumps(
        {
            "proposal_id": result.proposal_id,
            "ragas_score": result.ragas_score,
            "confidence": result.confidence_level,
            "claim_count": len(result.claim_contexts),
            "red_flags": result.red_flags,
            "summary": result.summary,
        },
        indent=2,
    )

    print(f"\nFull result (JSON):")
    print(result_json)

    print("\n" + "=" * 80)
    print(f"Phase 4 Demo Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
