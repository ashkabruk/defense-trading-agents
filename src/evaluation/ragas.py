"""RAGAS-inspired evaluation engine for trade proposals."""
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from src.llm.client import LLMClient
from src.models.config import ModelConfig
from src.models.core import TradeProposal
from src.models.evaluation import ClaimContext, ClaimDecomposition, RAGASEvaluationResult
from src.models.llm import ChatMessage

if TYPE_CHECKING:
    from src.storage.fact_store import ChromaFactStore

logger = structlog.get_logger(__name__)


class ClaimsResponse(BaseModel):
    """Response format for claim decomposition."""

    claims: list[str] = Field(default_factory=list, description="Extracted claims")


class ScoreResponse(BaseModel):
    """Response format for claim scoring."""

    score: float = Field(default=0.5, ge=0, le=1, description="Score 0-1")
    explanation: str = Field(default="", description="Reasoning")


class RAGASEvaluator:
    """RAGAS-inspired evaluator using claim decomposition and fact grounding."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_config: ModelConfig,
        fact_store: ChromaFactStore,
    ) -> None:
        self.llm_client = llm_client
        self.model_config = model_config
        self.fact_store = fact_store

    async def evaluate(self, proposal: TradeProposal, context_limit: int = 10) -> RAGASEvaluationResult:
        """Evaluate a trade proposal using RAGAS framework."""
        logger.info("ragas_eval_start", proposal_id=proposal.id, ticker=proposal.ticker)

        # Step 1: Decompose proposal into individual claims
        claims = await self._decompose_claims(proposal)
        logger.info("ragas_claims_decomposed", proposal_id=proposal.id, claim_count=len(claims.claims))

        # Step 2: For each claim, retrieve supporting facts and evaluate concurrently
        claim_tasks = [self._evaluate_claim(claim_text, proposal, context_limit) for claim_text in claims.claims]
        claim_contexts = await asyncio.gather(*claim_tasks) if claim_tasks else []

        # Step 3: Compute aggregate scores
        avg_faithfulness = sum(c.faithfulness_score for c in claim_contexts) / len(claim_contexts) if claim_contexts else 0.5
        avg_relevance = sum(c.relevance_score for c in claim_contexts) / len(claim_contexts) if claim_contexts else 0.5
        avg_consistency = sum(c.consistency_score for c in claim_contexts) / len(claim_contexts) if claim_contexts else 0.5

        # RAGAS composite: equally weighted average of three components
        ragas_score = (avg_faithfulness + avg_relevance + avg_consistency) / 3

        # Step 4: Generate summary and identify red flags
        summary, red_flags, confidence = await self._generate_summary(
            proposal, claim_contexts, avg_faithfulness, avg_relevance, avg_consistency
        )

        result = RAGASEvaluationResult(
            proposal_id=proposal.id,
            claim_contexts=claim_contexts,
            average_faithfulness=avg_faithfulness,
            average_relevance=avg_relevance,
            average_consistency=avg_consistency,
            ragas_score=ragas_score,
            summary=summary,
            red_flags=red_flags,
            confidence_level=confidence,
        )

        logger.info(
            "ragas_eval_complete",
            proposal_id=proposal.id,
            ragas_score=ragas_score,
            confidence=confidence,
        )

        return result

    async def _decompose_claims(self, proposal: TradeProposal) -> ClaimDecomposition:
        """Use LLM to decompose proposal reasoning into individual claims."""
        decompose_prompt = f"""You are an expert fact-checker. Analyze the following trade proposal and extract ALL explicit and implicit claims.

PROPOSAL:
Ticker: {proposal.ticker}
Direction: {proposal.direction}
Conviction: {proposal.conviction}/10
Rationale: {proposal.entry_rationale}

Your task: List each factual claim made in the rationale that can be independently verified. Be thorough but concise.

RESPOND ONLY WITH VALID JSON, NO MARKDOWN, NO EXPLANATION. Format: {{"claims": ["claim1", "claim2", ...]}}"""

        message = ChatMessage(role="user", content=decompose_prompt)
        response = await self.llm_client.chat(
            messages=[message],
            model_config=self.model_config,
        )

        try:
            claims_data = json.loads(response.content)
            return ClaimDecomposition(
                claims=claims_data.get("claims", []),
                proposal_id=proposal.id,
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("ragas_decompose_json_error", proposal_id=proposal.id, error=str(e))
            # Fallback: extract claim-like sentences
            sentences = [s.strip() for s in proposal.entry_rationale.split(".") if s.strip()]
            return ClaimDecomposition(claims=sentences, proposal_id=proposal.id)

    async def _evaluate_claim(
        self,
        claim: str,
        proposal: TradeProposal,
        context_limit: int,
    ) -> ClaimContext:
        """Evaluate a single claim against retrieved facts."""
        context = ClaimContext(claim=claim)

        # Step 1: Retrieve supporting documents
        try:
            supporting_docs = self.fact_store.search(
                query=f"{proposal.ticker} {claim}",
                max_results=context_limit,
            )
            context.supporting_docs = supporting_docs
        except Exception as e:
            logger.warning("ragas_fact_retrieval_error", claim=claim, error=str(e))

        # Step 2: Score faithfulness (does evidence support claim?)
        faithfulness_prompt = f"""You are an expert evaluator. Given a claim and supporting documents, rate how well the evidence supports the claim.

CLAIM: {claim}

SUPPORTING DOCUMENTS:
{self._format_documents(supporting_docs)}

Rate the faithfulness of this claim given the evidence:
- 0.0-0.3: Unsupported or contradicted by evidence
- 0.3-0.6: Weakly supported; some evidence exists but gaps
- 0.6-0.8: Well-supported; evidence clearly backs claim
- 0.8-1.0: Strongly supported; multiple credible sources

Return ONLY a JSON object like {{"score": 0.75, "explanation": "..."}}
"""

        context.faithfulness_score = await self._score_claim(faithfulness_prompt, default=0.5)

        # Step 3: Score relevance (are retrieved docs relevant to claim?)
        if supporting_docs:
            relevance_prompt = f"""Rate how relevant the retrieved documents are to evaluating this claim:

CLAIM: {claim}

RETRIEVED DOCS:
{self._format_documents(supporting_docs, max_docs=3)}

Score relevance on 0-1 scale. Return JSON: {{"score": 0.85, "explanation": "..."}}
"""
            context.relevance_score = await self._score_claim(relevance_prompt, default=0.5)
        else:
            context.relevance_score = 0.0  # No docs = not relevant

        # Step 4: Check for contradictions
        contradictions = self._detect_contradictions(claim, supporting_docs)
        context.contradicting_facts = contradictions

        # Adjust consistency score based on contradictions
        if contradictions:
            context.consistency_score = max(0.0, context.faithfulness_score - (0.3 * len(contradictions)))
        else:
            context.consistency_score = context.faithfulness_score

        logger.info(
            "ragas_claim_scored",
            claim_length=len(claim),
            faithfulness=context.faithfulness_score,
            relevance=context.relevance_score,
        )

        return context

    async def _score_claim(self, prompt: str, default: float = 0.5) -> float:
        """Use LLM to score a claim and return numeric score."""
        # Ensure JSON instruction in prompt
        json_prompt = (
            prompt
            + "\n\nRESPOND ONLY WITH VALID JSON, NO MARKDOWN, NO EXPLANATION. "
            + 'Format: {"score": 0.85, "explanation": "..."}'
        )
        message = ChatMessage(role="user", content=json_prompt)
        try:
            response = await self.llm_client.chat(
                messages=[message],
                model_config=self.model_config,
            )
            data = json.loads(response.content)
            score = float(data.get("score", default))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception as e:
            logger.warning("ragas_score_claim_error", error=str(e))
            return default

    async def _generate_summary(
        self,
        proposal: TradeProposal,
        claim_contexts: list[ClaimContext],
        avg_faithfulness: float,
        avg_relevance: float,
        avg_consistency: float,
    ) -> tuple[str, list[str], str]:
        """Generate evaluation summary and identify red flags."""
        if not claim_contexts:
            return "No claims to evaluate.", [], "low"

        # Identify red flags
        red_flags: list[str] = []

        # Flag 1: Low faithfulness
        if avg_faithfulness < 0.5:
            red_flags.append(
                f"Low faithfulness ({avg_faithfulness:.1%}): Claims are not well-supported by available evidence"
            )

        # Flag 2: Low relevance
        if avg_relevance < 0.4:
            red_flags.append(
                f"Low relevance ({avg_relevance:.1%}): Retrieved evidence may not be directly relevant to claims"
            )

        # Flag 3: Contradictions found
        contradicting_claims = [c.claim for c in claim_contexts if c.contradicting_facts]
        if contradicting_claims:
            red_flags.append(f"Contradictions detected in {len(contradicting_claims)} claims (see details)")

        # Determine confidence level
        data_density = len(claim_contexts) * avg_relevance  # More claims + higher relevance = higher confidence
        if avg_faithfulness >= 0.7 and data_density > 2:
            confidence = "high"
        elif avg_faithfulness >= 0.5 and data_density > 1:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate summary
        summary_lines = [
            f"RAGAS Evaluation for {proposal.ticker} {proposal.direction.upper()} proposal",
            f"Decomposed into {len(claim_contexts)} claims",
            f"Faithfulness: {avg_faithfulness:.1%} (how well claims are supported)",
            f"Relevance: {avg_relevance:.1%} (how relevant evidence is)",
            f"Consistency: {avg_consistency:.1%} (absence of contradictions)",
            f"Confidence: {confidence.upper()}",
        ]

        if red_flags:
            summary_lines.append("\nRed Flags:")
            summary_lines.extend(f"  - {flag}" for flag in red_flags)

        summary = "\n".join(summary_lines)
        return summary, red_flags, confidence

    def _format_documents(self, docs: list[dict[str, object]], max_docs: int = 5) -> str:
        """Format retrieved documents for display."""
        if not docs:
            return "[No supporting documents found]"

        formatted = []
        for i, doc in enumerate(docs[:max_docs]):
            doc_text = str(doc.get("document", ""))[:250]
            source = doc.get("metadata", {}).get("source", "unknown") if isinstance(doc.get("metadata"), dict) else "unknown"
            formatted.append(f"[Doc {i+1}] ({source}): {doc_text}...")

        return "\n".join(formatted)

    def _detect_contradictions(self, claim: str, docs: list[dict[str, object]]) -> list[str]:
        """Simple heuristic contradiction detection."""
        contradictions: list[str] = []

        # Check for explicit negations in retrieved documents
        negation_markers = ["not ", "no ", "cannot ", "won't ", "doesn't ", "failed to", "unable to"]
        claim_lower = claim.lower()

        for doc in docs:
            doc_text = str(doc.get("document", "")).lower()
            # Simple check: if document contains negation of key terms in claim
            claim_terms = claim_lower.split()
            for term in claim_terms:
                if len(term) > 4:  # Skip short words
                    for neg in negation_markers:
                        if f"{neg}{term}" in doc_text:
                            contradictions.append(
                                f"Document contradicts claim: found '{neg}{term}' in source"
                            )

        return contradictions
