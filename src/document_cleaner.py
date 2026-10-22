"""Conservative cleanup for common PDF/DOCX extraction artifacts."""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Iterable

from langchain_core.documents import Document


_TOC_LINE = re.compile(
    r"^(?P<heading>.+?)\s*(?:\.{3,}|(?:\.\s*){3,})\s*(?:\d{1,4}|[ivxlcdm]{1,8})\s*$",
    re.IGNORECASE,
)
_DOT_ONLY_PAGE = re.compile(r"^\s*\.{3,}\s*\d{1,4}\s*$")
_SECTION_HEADING = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*[.)]?\s+))?"
    r"(?:introduction|purpose|scope|requirements|history log|color coding|"
    r"objectives?|overview|background|conclusion|appendix)\s*:?\s*$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u200b", "")
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _is_known_heading(line: str) -> bool:
    return bool(_SECTION_HEADING.match(line))


def _page_number_for(document: Document) -> set[str]:
    page = document.metadata.get("page")
    if page is None:
        return set()
    try:
        number = int(page)
    except (TypeError, ValueError):
        return set()
    return {str(number), str(number + 1)}


def _clean_lines(document: Document, repeated_lines: set[str]) -> list[str]:
    page_numbers = _page_number_for(document)
    raw_lines = str(document.page_content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []
    for raw_line in raw_lines:
        line = _clean_line(raw_line)
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if _DOT_ONLY_PAGE.match(line):
            continue
        if _TOC_LINE.match(line):
            line = _TOC_LINE.sub(lambda match: match.group("heading").rstrip(), line)
        if line in page_numbers and not _BULLET.match(line):
            continue
        if _line_key(line) in repeated_lines and not _is_known_heading(line):
            continue
        cleaned.append(line)

    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    compact: list[str] = []
    for line in cleaned:
        if line == "" and compact and compact[-1] == "":
            continue
        compact.append(line)
    return compact


def clean_document(document: Document, repeated_lines: Iterable[str] = ()) -> Document:
    """Return a cleaned copy of one LangChain document, preserving metadata."""
    metadata = deepcopy(dict(document.metadata or {}))
    original = str(document.page_content or "")
    lines = _clean_lines(document, set(repeated_lines))
    text = "\n".join(lines).strip()

    # Keep the original only for source traceability; it is never sent to the LLM.
    if text != original:
        metadata["original_content"] = original

    existing_heading = metadata.get("heading") or metadata.get("section")
    if not existing_heading:
        for line in lines[:5]:
            match = _SECTION_HEADING.match(line)
            if match:
                heading = line.rstrip(":").strip()
                metadata["section"] = heading
                metadata["heading"] = heading
                break

    return Document(page_content=text, metadata=metadata)


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean documents and remove only confidently repeated page boundaries."""
    boundary_counts: Counter[str] = Counter()
    boundary_values: dict[str, str] = {}
    for document in documents:
        lines = [_clean_line(line) for line in str(document.page_content or "").splitlines()]
        non_empty = [line for line in lines if line]
        boundaries = set(non_empty[:3] + non_empty[-3:])
        for line in boundaries:
            if 3 <= len(line) <= 120 and not _BULLET.match(line) and "|" not in line:
                key = _line_key(line)
                boundary_counts[key] += 1
                boundary_values[key] = line

    required = max(2, (len(documents) + 1) // 2)
    repeated = {
        key for key, count in boundary_counts.items()
        if count >= required and not _is_known_heading(boundary_values[key])
    }
    return [clean_document(document, repeated) for document in documents]

