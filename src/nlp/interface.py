"""Provider interface for local and future LLM query analyzers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.nlp.models import QueryAnalysis


class QueryAnalyzer(Protocol):
    def analyze(
        self, query: str, conversation_history: Sequence[object] | None = None
    ) -> QueryAnalysis:
        ...
