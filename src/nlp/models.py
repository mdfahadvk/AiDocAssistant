"""Models shared by query-understanding providers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    intent: str = "unknown"
    keywords: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    project: str | None = None
    document: str | None = None
    is_overview_query: bool = False
    is_comparison_query: bool = False
    is_followup_query: bool = False
    retrieval_queries: list[str] = field(default_factory=list)
    query_types: list[str] = field(default_factory=list)
    is_short_query: bool = False
    is_broad_query: bool = False
    expansion_required: bool = False
    corrected_query: str | None = None

    def as_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "keywords": self.keywords,
            "entities": self.entities,
            "project": self.project,
            "document": self.document,
            "is_overview_query": self.is_overview_query,
            "is_comparison_query": self.is_comparison_query,
            "is_followup_query": self.is_followup_query,
            "retrieval_queries": self.retrieval_queries,
            "query_types": self.query_types,
            "is_short_query": self.is_short_query,
            "is_broad_query": self.is_broad_query,
            "expansion_required": self.expansion_required,
            "corrected_query": self.corrected_query,
        }
