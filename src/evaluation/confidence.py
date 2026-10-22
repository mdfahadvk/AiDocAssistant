"""Lightweight answer confidence scoring.

This module is deliberately independent of Chroma and the LLM. It evaluates
only evidence and citations already produced by the RAG search.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.evaluation.models import AnswerConfidence, unavailable_confidence

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
# Keep a citation written after terminal punctuation attached to its claim.
_SENTENCE_RE = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z])")
_FILLER = {
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "and", "or",
    "to", "of", "in", "for", "with", "it", "as", "be", "on", "by", "from", "at",
    "into", "than", "then", "also", "can", "may", "will", "allows", "allow", "using",
}
_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|(?:[A-Z][A-Za-z /&-]{1,50}:\s*$))")
_NEGATION = {"not", "no", "never", "cannot", "can't", "without", "unsupported", "false"}
_CONTRADICTION_PAIRS = (("enabled", "disabled"), ("active", "inactive"), ("allowed", "not allowed"), ("yes", "no"), ("true", "false"))


@dataclass(frozen=True)
class ConfidenceWeights:
    retrieval: float = 0.30
    groundedness: float = 0.30
    citation: float = 0.20
    consistency: float = 0.10
    scope: float = 0.10

    def __post_init__(self) -> None:
        values = (self.retrieval, self.groundedness, self.citation, self.consistency, self.scope)
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("Answer confidence weights must be non-negative and total 1.0")

    @classmethod
    def from_env(cls) -> "ConfidenceWeights":
        return cls(
            retrieval=float(os.getenv("CONFIDENCE_RETRIEVAL_WEIGHT", "0.30")),
            groundedness=float(os.getenv("CONFIDENCE_GROUNDEDNESS_WEIGHT", "0.30")),
            citation=float(os.getenv("CONFIDENCE_CITATION_WEIGHT", "0.20")),
            consistency=float(os.getenv("CONFIDENCE_CONSISTENCY_WEIGHT", "0.10")),
            scope=float(os.getenv("CONFIDENCE_SCOPE_WEIGHT", "0.10")),
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokens(text: str) -> set[str]:
    return {
        word.casefold()
        for word in _WORD_RE.findall(text)
        if not word.isdigit() and word.casefold() not in _FILLER
    }


def _claim_parts(answer: str) -> list[tuple[str, set[str]]]:
    claims: list[tuple[str, set[str]]] = []
    for part in _SENTENCE_RE.split(answer):
        cleaned = re.sub(r"^[\s\-*•\d.)]+", "", part)
        citation_ids = set(re.findall(r"\[(\d+)\]", cleaned))
        cleaned = re.sub(r"\s*\[\d+\]\s*", " ", cleaned).strip()
        if not cleaned or _HEADING_RE.match(cleaned):
            continue
        tokens = _tokens(cleaned)
        if tokens:
            claims.append((cleaned, citation_ids))
    return claims


def _claims(answer: str) -> list[str]:
    return [claim for claim, _ in _claim_parts(answer)]


def _source_text(source: dict[str, Any]) -> str:
    chunks = source.get("chunks") or []
    return "\n".join(str(item) for item in chunks) or str(source.get("text", ""))


class AnswerConfidenceService:
    def __init__(self, weights: ConfidenceWeights | None = None, retrieval_max_distance: float | None = None):
        self.weights = weights or ConfidenceWeights.from_env()
        self.retrieval_max_distance = retrieval_max_distance or float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.65"))
        if self.retrieval_max_distance <= 0:
            raise ValueError("RETRIEVAL_MAX_DISTANCE must be greater than zero")

    def calculate_retrieval_relevance(self, chunks: Iterable[dict[str, Any]]) -> float | None:
        distances = []
        for chunk in chunks:
            try:
                distance = float(chunk.get("distance"))
            except (TypeError, ValueError):
                continue
            if distance >= 0:
                distances.append(distance)
        if not distances:
            return None
        qualities = sorted(
            (_clamp(1 - distance / self.retrieval_max_distance) for distance in distances),
            reverse=True,
        )
        best = qualities[0]
        average = sum(qualities) / len(qualities)
        # A single strong chunk should remain strong. Additional chunks add
        # support, but a weaker secondary chunk cannot erase the best match.
        support = min(len(qualities) / 2.0, 1.0)
        return _clamp(best * 0.55 + average * 0.25 + support * 0.20)

    def calculate_groundedness(self, answer: str, sources: Iterable[dict[str, Any]]) -> float | None:
        evidence = _tokens(" ".join(_source_text(source) for source in sources))
        claims = _claims(answer)
        if not evidence or not claims:
            return None
        scores = []
        for claim in claims:
            claim_tokens = _tokens(claim)
            if not claim_tokens:
                continue
            overlap = len(claim_tokens & evidence) / len(claim_tokens)
            # Common connective language is already excluded from tokens. A
            # claim is supported when its meaningful terms substantially match
            # the retrieved evidence; partial overlap receives partial credit.
            scores.append(_clamp(overlap))
        return sum(scores) / len(scores) if scores else None

    def calculate_citation_coverage(self, answer: str, citations: Iterable[dict[str, Any]]) -> float | None:
        sources = list(citations)
        claims = _claim_parts(answer)
        if not sources or not claims:
            return None
        valid_ids = {str(source.get("id")) for source in sources if source.get("id") is not None}
        covered = 0
        factual = 0
        for claim, ids in claims:
            if _tokens(claim):
                factual += 1
                covered += bool(ids & valid_ids)
        return covered / factual if factual else None

    def calculate_evidence_consistency(self, sources: Iterable[dict[str, Any]]) -> float | None:
        raw_texts = [_source_text(source).casefold() for source in sources if _tokens(_source_text(source))]
        texts = [_tokens(text) for text in raw_texts]
        if len(texts) < 2:
            return 1.0 if texts else None
        conflicts = 0
        comparisons = 0
        for index, left in enumerate(texts):
            for right_index, right in enumerate(texts[index + 1 :], start=index + 1):
                comparisons += 1
                shared_subject_terms = len((left & right) - _NEGATION)
                explicit_conflict = any(
                    (first in raw_texts[index] and second in raw_texts[right_index])
                    or (second in raw_texts[index] and first in raw_texts[right_index])
                    for first, second in _CONTRADICTION_PAIRS
                )
                if explicit_conflict and shared_subject_terms >= 1:
                    conflicts += 1
        return 1.0 if not comparisons else _clamp(1 - conflicts / comparisons)

    def calculate_scope_match(self, sources: Iterable[dict[str, Any]], project: str | None = None, document: str | None = None) -> float | None:
        sources = list(sources)
        if not sources or not project and not document:
            return None if not sources else 0.5
        project_name = project.casefold() if project else None
        document_name = os.path.basename(document.replace("\\", "/")).casefold() if document else None
        matches = 0
        for source in sources:
            source_project = str(source.get("project", "")).casefold()
            source_document = os.path.basename(str(source.get("document", "")).replace("\\", "/")).casefold()
            matches += bool((not project_name or source_project == project_name) and (not document_name or source_document == document_name))
        return matches / len(sources)

    def calculate_confidence(self, answer: str, retrieved_chunks: list[dict[str, Any]], citations: list[dict[str, Any]], project: str | None = None, document: str | None = None) -> AnswerConfidence:
        if not retrieved_chunks or not citations:
            return unavailable_confidence("Confidence unavailable because no cited evidence was available.")
        # Only sources exposed to the user are used as evidence for the score.
        citation_ids = {str(source.get("id")) for source in citations}
        evidence = [chunk for chunk in retrieved_chunks if str(chunk.get("source_id")) in citation_ids]
        if not evidence:
            evidence = retrieved_chunks
        factors = {
            "retrieval": self.calculate_retrieval_relevance(evidence),
            "groundedness": self.calculate_groundedness(answer, citations),
            "citation": self.calculate_citation_coverage(answer, citations),
            "consistency": self.calculate_evidence_consistency(citations),
            "scope": self.calculate_scope_match(citations, project, document),
        }
        factor_weights = dict(zip(factors, self.weights.__dict__.values()))
        available = {
            name: (value, factor_weights[name])
            for name, value in factors.items()
            if value is not None
        }
        if not available:
            return unavailable_confidence("Confidence unavailable because no evidence signals could be measured reliably.")
        total_weight = sum(weight for _, weight in available.values())
        score = round(100 * sum(value * weight for value, weight in available.values()) / total_weight)
        level = "High" if score >= 90 else "Medium" if score >= 75 else "Low" if score >= 50 else "Very Low"
        if score >= 85:
            explanation = "Strong document match with supporting source citations and consistent evidence."
        elif score >= 65:
            explanation = "The evidence supports the answer, with some limitations in retrieval or claim coverage."
        else:
            explanation = "Weak document match or several unsupported claims were detected."
        return AnswerConfidence(score, level, factors["retrieval"], factors["groundedness"], factors["citation"], factors["consistency"], factors["scope"], explanation)
