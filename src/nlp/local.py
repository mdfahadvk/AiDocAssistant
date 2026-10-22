"""Local spaCy query analysis with a conservative rule-based layer."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from collections.abc import Sequence

from src.nlp.models import QueryAnalysis

LOGGER = logging.getLogger(__name__)
_FOLLOW_UP_RE = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|former|latter)\b|"
    r"^\s*(and then|also|what about|how about|why is that|where is that)\b",
    re.IGNORECASE,
)
_SAFE_CORRECTIONS = {
    "avialble": "available", "comission": "commission", "comissions": "commissions",
    "introducton": "introduction", "purpse": "purpose", "prerequsites": "prerequisites",
}


def _vocabulary_path() -> Path:
    return Path(os.getenv("NLP_TERMS_FILE", "config/nlp_terms.json")).expanduser()


def load_vocabulary() -> dict[str, list[str]]:
    try:
        value = json.loads(_vocabulary_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(category): [str(term).strip() for term in terms if str(term).strip()]
        for category, terms in value.items()
        if isinstance(terms, list)
    }


class LocalQueryAnalyzer:
    def __init__(self, vocabulary: dict[str, list[str]] | None = None, model_name: str | None = None):
        self.vocabulary = vocabulary if vocabulary is not None else load_vocabulary()
        self.model_name = model_name or os.getenv("SPACY_MODEL", "en_core_web_sm")
        self._nlp = None
        self._load_attempted = False

    @property
    def nlp(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                import spacy

                self._nlp = spacy.load(self.model_name)
            except Exception as exc:  # spaCy must never break chat.
                LOGGER.warning("Local NLP unavailable; using safe query fallback: %s", exc)
                self._nlp = None
        return self._nlp

    def analyze(
        self, query: str, conversation_history: Sequence[object] | None = None
    ) -> QueryAnalysis:
        original = str(query or "").strip()
        if not original:
            return QueryAnalysis(original_query=original, normalized_query=original)
        try:
            return self._analyze(original, conversation_history)
        except Exception as exc:
            LOGGER.warning("Local NLP analysis failed; using query as-is: %s", exc)
            return QueryAnalysis(
                original_query=original,
                normalized_query=original,
                retrieval_queries=[original],
            )

    def _analyze(self, query: str, history: Sequence[object] | None) -> QueryAnalysis:
        corrected = self._correct_spelling(query)
        doc = self.nlp(corrected) if self.nlp is not None else None
        entities = self._domain_entities(corrected)
        project_values = entities.get("project", []) + entities.get("application", [])
        projects = self._unique(project_values)
        is_followup = bool(_FOLLOW_UP_RE.search(corrected))
        if not projects and is_followup:
            history_text = " ".join(self._message_text(item) for item in (history or []))
            history_entities = self._domain_entities(history_text)
            projects = self._unique(
                history_entities.get("project", []) + history_entities.get("application", [])
            )
            if projects:
                entities["project"] = projects

        intent = self._intent(corrected)
        overview = intent in {"overview", "list", "summary"}
        comparison = intent == "comparison"
        keywords = self._keywords(doc, corrected, entities)
        is_short = len(re.findall(r"[A-Za-z0-9]+", corrected)) <= 3
        section = self._is_section_lookup(corrected)
        broad = is_short or section or intent in {"overview", "list", "summary", "page_functionality"}
        query_types = []
        if section: query_types.append("section_lookup")
        if is_short: query_types.append("short_query")
        if intent in {"overview", "list", "how_to", "page_functionality", "comparison", "troubleshooting"}:
            query_types.append(intent)
        if not query_types: query_types.append("normal")
        normalized = self._normalized(corrected, intent, entities, keywords, projects)
        retrieval_queries = self._expand(corrected, normalized, intent, keywords, projects, entities) if broad else [normalized]
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            keywords=keywords,
            entities=entities,
            project=projects[0] if len(projects) == 1 else None,
            is_overview_query=overview,
            is_comparison_query=comparison,
            is_followup_query=is_followup,
            retrieval_queries=retrieval_queries,
            query_types=self._unique(query_types),
            is_short_query=is_short,
            is_broad_query=broad,
            expansion_required=broad,
            corrected_query=corrected if corrected.casefold() != query.casefold() else None,
        )

    @staticmethod
    def _correct_spelling(query: str) -> str:
        return re.sub(r"\b[A-Za-z]+\b", lambda match: _SAFE_CORRECTIONS.get(match.group(0).casefold(), match.group(0)), query)

    @staticmethod
    def _is_section_lookup(query: str) -> bool:
        text = query.casefold().strip(" ?.!\t")
        return bool(re.fullmatch(r"(?:introduction|purpose|features|api|history log|pages?|scope|color coding|prerequisites?)", text))

    def _domain_entities(self, query: str) -> dict[str, list[str]]:
        entities: dict[str, list[str]] = {}
        for category, terms in self.vocabulary.items():
            matches = []
            for term in terms:
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query, re.IGNORECASE):
                    matches.append(term)
            if matches:
                entities[category] = self._unique(matches)
        if "projects" in entities and "project" not in entities:
            entities["project"] = list(entities["projects"])
        if "applications" in entities and "project" not in entities:
            entities["project"] = list(entities["applications"])
        return entities

    @staticmethod
    def _intent(query: str) -> str:
        text = query.casefold()
        if re.search(r"\b(objective|objectives|goal|goals|requirements?)\b", text):
            return "list"
        if re.search(r"\b(compare|comparison|difference|differentiate|versus|vs\.)\b", text):
            return "comparison"
        if re.search(r"\b(database|sql|schema|table|stored procedure)\b", text):
            return "database"
        if re.search(r"\b(api|endpoint|integration|service call)\b", text):
            return "api"
        if re.search(r"\b(error|issue|problem|fail|troubleshoot|not working)\b", text):
            return "troubleshooting"
        if re.search(r"\b(page|screen|used for|functionality)\b", text):
            return "page_functionality"
        if re.search(r"\b(how|steps?|procedure|add|create|configure|set up|setup)\b", text):
            return "how_to"
        if re.search(r"\b(each|every|list|all|pages?|screens?|use cases?)\b", text):
            return "list"
        if re.search(r"\b(summary|summarize|briefly)\b", text):
            return "summary"
        if re.search(r"\b(what is|what are|explain|tell me about|describe|introduce me)\b", text):
            return "overview"
        return "unknown"

    def _keywords(self, doc, query: str, entities: dict[str, list[str]]) -> list[str]:
        terms: list[str] = []
        if doc is not None:
            for token in doc:
                if token.is_stop or token.is_punct or token.like_num or len(token.text) < 3:
                    continue
                value = token.lemma_.casefold().strip()
                if value and value not in terms:
                    terms.append(value)
            for chunk in doc.noun_chunks:
                value = re.sub(r"\s+", " ", chunk.text.strip()).casefold()
                if value and len(value.split()) > 1 and value not in terms:
                    terms.append(value)
        else:
            stop_words = {
                "a", "an", "and", "are", "do", "for", "how", "i", "in", "is", "it",
                "me", "of", "on", "please", "the", "to", "use", "what", "does", "you",
            }
            terms = [
                word.casefold()
                for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query)
                if word.casefold() not in stop_words
            ]
        for category_terms in entities.values():
            for term in category_terms:
                if term.casefold() not in terms:
                    terms.append(term.casefold())
        return terms[:20]

    @staticmethod
    def _normalized(query: str, intent: str, entities: dict[str, list[str]], keywords: list[str], projects: list[str]) -> str:
        technical = []
        for category in ("pages", "features", "technologies"):
            technical.extend(entities.get(category, []))
        project_text = " in " + ", ".join(projects) if projects else ""
        section = query.strip(" ?.!\t").casefold()
        if section == "introduction":
            return f"Introduction section of the selected project{project_text}"
        if section == "purpose":
            return f"Purpose of the selected application or project{project_text}"
        if intent == "page_functionality" and technical:
            return f"{technical[0]} page functionality and usage{project_text}"
        if intent == "overview" and re.search(r"\b(introduce|introduction|purpose)\b", query, re.I):
            return f"Introduction and purpose of the selected project{project_text}"
        if intent == "how_to" and keywords:
            project_keys = {project.casefold() for project in projects}
            usable_keywords = [keyword for keyword in keywords if keyword.casefold() not in project_keys]
            return " ".join(usable_keywords[:8]) + project_text
        return re.sub(r"\s+", " ", query).strip()

    def _expand(self, query: str, normalized: str, intent: str, keywords: list[str], projects: list[str], entities: dict[str, list[str]]) -> list[str]:
        candidates = [normalized, query]
        text = query.strip(" ?.!\t")
        if self._is_section_lookup(query):
            label = text
            candidates.extend([f"{label} section", f"Project {label}", f"Application {label}"])
            if label.casefold() in {"introduction", "purpose"}:
                candidates.append("Purpose of the application")
        if keywords:
            candidates.append(" ".join(keywords[:6]))
        if projects:
            candidates.append(" ".join(projects + keywords[:5]))
        if re.search(r"\bcolor(?:s)?\s+coding\b", query, re.IGNORECASE):
            candidates = [normalized, "color coding", "available colors", "calendar color coding", "color meanings"]
        if intent in {"overview", "list", "summary"}:
            subject = " ".join(projects) if projects else "documents"
            candidates.extend([f"{subject} pages", f"{subject} page functionality", f"{subject} use cases"])
        if intent == "page_functionality" and entities.get("pages"):
            candidates.append(f"{entities['pages'][0]} page usage")
        maximum = max(1, int(os.getenv("MAX_RETRIEVAL_QUERIES", os.getenv("NLP_MAX_RETRIEVAL_QUERIES", "5"))))
        return self._unique([item for item in candidates if item.strip()])[:maximum]

    @staticmethod
    def _message_text(item: object) -> str:
        return str(getattr(item, "content", item))

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        result = []
        seen = set()
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result
