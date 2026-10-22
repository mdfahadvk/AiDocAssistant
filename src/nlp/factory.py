"""Factory for selecting the query-understanding provider."""
from __future__ import annotations

import os
from functools import lru_cache

from src.nlp.interface import QueryAnalyzer
from src.nlp.local import LocalQueryAnalyzer


class QueryAnalyzerFactory:
    @staticmethod
    @lru_cache(maxsize=4)
    def create(provider: str | None = None) -> QueryAnalyzer:
        selected = (provider or os.getenv("NLP_PROVIDER", "local")).strip().casefold()
        if selected == "local":
            return LocalQueryAnalyzer()
        if selected == "llm":
            raise ValueError("NLP_PROVIDER=llm is reserved for a future provider.")
        raise ValueError(f"Unsupported NLP_PROVIDER: {selected}")
