"""RAGAS-inspired evaluation models for claim validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field


@dataclass
class ClaimDecomposition:
    """Decomposed claims from a trade proposal."""

    claims: list[str]
    """Individual claims extracted from proposal reasoning."""

    proposal_id: str
    """ID of the source proposal."""


class ClaimContext(BaseModel):
    """Single claim with supporting and contradicting evidence."""

    claim: str = Field(..., description="The claim to evaluate")
    supporting_docs: list[dict[str, object]] = Field(default_factory=list, description="Retrieved facts supporting claim")
    contradicting_facts: list[str] = Field(default_factory=list, description="Facts that contradict the claim")
    faithfulness_score: float = Field(default=0.5, ge=0, le=1, description="0-1 score: how well supported is the claim")
    relevance_score: float = Field(default=0.5, ge=0, le=1, description="0-1 score: relevance of retrieved docs")
    consistency_score: float = Field(default=0.5, ge=0, le=1, description="0-1 score: internal consistency")


class RAGASEvaluationResult(BaseModel):
    """RAGAS evaluation result for a trade proposal."""

    proposal_id: str = Field(..., description="ID of evaluated proposal")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When evaluation ran")
    
    # Per-claim evaluations
    claim_contexts: list[ClaimContext] = Field(default_factory=list, description="Evaluated claims")
    
    # Aggregate scores
    average_faithfulness: float = Field(default=0.5, ge=0, le=1, description="Mean faithfulness across claims")
    average_relevance: float = Field(default=0.5, ge=0, le=1, description="Mean relevance across claims")
    average_consistency: float = Field(default=0.5, ge=0, le=1, description="Mean consistency across claims")
    
    # Final composite score
    ragas_score: float = Field(default=0.5, ge=0, le=1, description="Composite RAGAS score (0-1)")
    
    # Reasoning
    summary: str = Field(default="", description="Human-readable evaluation summary")
    red_flags: list[str] = Field(default_factory=list, description="Flagged concerns or contradictions")
    confidence_level: str = Field(default="medium", description="high/medium/low based on data density")


@dataclass
class EvaluationStatistics:
    """Statistics from evaluation run."""

    total_proposals_evaluated: int = 0
    """How many proposals were evaluated."""

    total_claims_decomposed: int = 0
    """Total number of claims extracted."""

    avg_claims_per_proposal: float = 0.0
    """Average claims decomposed per proposal."""

    avg_ragas_score: float = 0.0
    """Average RAGAS score across all proposals."""

    high_confidence_count: int = 0
    """Proposals with high confidence (score >= 0.7)."""

    low_confidence_count: int = 0
    """Proposals with low confidence (score < 0.4)."""

    flagged_proposals: list[tuple[str, str]] = field(default_factory=list)
    """[(proposal_id, red_flag_reason)] tuples."""
