from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerConfidence:
    """Explainable, local confidence assessment for one answer."""

    score: int | None
    level: str
    retrieval_relevance: float | None
    groundedness: float | None
    citation_coverage: float | None
    evidence_consistency: float | None
    scope_match: float | None
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def unavailable_confidence(reason: str = "There is not enough evidence to calculate confidence.") -> AnswerConfidence:
    return AnswerConfidence(
        score=None,
        level="Unavailable",
        retrieval_relevance=None,
        groundedness=None,
        citation_coverage=None,
        evidence_consistency=None,
        scope_match=None,
        explanation=reason,
    )
