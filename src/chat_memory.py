"""Session-scoped chat history and follow-up query rewriting."""
from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

DEFAULT_MAX_HISTORY_TURNS = 6
_FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|former|latter|above)\b"
    r"|\b(the previous|the same|that application|this application|that project|this project)\b",
    re.IGNORECASE,
)
_FOLLOW_UP_OPENING_PATTERN = re.compile(
    r"^\s*(and then|also|what about|how about|why is that|where is that)\b",
    re.IGNORECASE,
)


def max_history_turns() -> int:
    """Return the configured number of recent user/assistant turns."""
    try:
        return max(0, int(os.getenv("MAX_HISTORY_TURNS", str(DEFAULT_MAX_HISTORY_TURNS))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_HISTORY_TURNS


def ensure_chat_session(state: MutableMapping[str, Any]) -> None:
    """Initialize per-browser chat state without touching the vector store."""
    if not isinstance(state.get("chat_messages"), list):
        state["chat_messages"] = []
    history = state.get("chat_history")
    if not isinstance(history, list) or any(
        not isinstance(message, BaseMessage) for message in history
    ):
        history = []
        for message in state["chat_messages"]:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "")).casefold()
            content = str(message.get("content", ""))
            if role in {"user", "human"}:
                history.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                history.append(AIMessage(content=content))
        state["chat_history"] = history
    if not state.get("chat_session_id"):
        state["chat_session_id"] = uuid.uuid4().hex


def reset_chat_session(state: MutableMapping[str, Any]) -> None:
    """Start a fresh in-memory conversation; leave documents/indexing untouched."""
    state["chat_messages"] = []
    state["chat_history"] = []
    state["chat_session_id"] = uuid.uuid4().hex
    state["active_page"] = "Chat"
    state["scroll_chat_to_latest"] = False
    for key in list(state.keys()):
        if str(key).startswith("view_source_"):
            try:
                del state[key]
            except Exception:
                pass


def as_langchain_history(
    history: Sequence[Any] | None,
    turns: int | None = None,
) -> list[BaseMessage]:
    """Convert session messages to LangChain messages, bounded by turn count."""
    if not history:
        return []
    try:
        limit = max_history_turns() if turns is None else max(0, int(turns))
        if limit == 0:
            return []
        converted: list[BaseMessage] = []
        for item in history:
            if isinstance(item, HumanMessage):
                converted.append(item)
            elif isinstance(item, AIMessage):
                converted.append(item)
            elif isinstance(item, BaseMessage):
                if item.type == "human":
                    converted.append(HumanMessage(content=str(item.content)))
                elif item.type == "ai":
                    converted.append(AIMessage(content=str(item.content)))
            elif isinstance(item, Mapping):
                role = str(item.get("role", "")).casefold()
                content = str(item.get("content", ""))
                if role in {"user", "human"}:
                    converted.append(HumanMessage(content=content))
                elif role in {"assistant", "ai"}:
                    converted.append(AIMessage(content=content))
        converted = converted[-(limit * 2):]
        # Histories are normally complete pairs. Drop a leading assistant message
        # if an incomplete pair is encountered at the truncation boundary.
        if converted and isinstance(converted[0], AIMessage):
            converted = converted[1:]
        return converted
    except Exception:
        return []


def needs_query_rewrite(query: str) -> bool:
    """Use a conservative lexical check so standalone questions stay untouched."""
    return bool(_FOLLOW_UP_PATTERN.search(query) or _FOLLOW_UP_OPENING_PATTERN.search(query))


def rewrite_standalone_query(
    llm: Any,
    query: str,
    history: Sequence[Any] | None,
    turns: int | None = None,
) -> str:
    """Resolve follow-up references for retrieval; fail back to the original query."""
    recent = as_langchain_history(history, turns=turns)
    if not recent or not needs_query_rewrite(query):
        return query
    try:
        messages: list[BaseMessage] = [
            SystemMessage(
                content=(
                    "Rewrite a follow-up question as a standalone search query using the "
                    "conversation only to resolve names and references. Do not answer it, "
                    "add facts, or assume anything not stated. Return only the standalone "
                    "question, with no label or explanation."
                )
            ),
            *recent,
            HumanMessage(content=f"Latest question: {query}"),
        ]
        response = llm.invoke(messages)
        rewritten = str(response.content).strip().strip('"`')
        rewritten = re.sub(r"^(standalone (search )?query|rewritten question)\s*:\s*", "", rewritten, flags=re.I)
        rewritten = rewritten.strip()
        return rewritten or query
    except Exception:
        return query


def history_for_prompt(history: Sequence[Any] | None, turns: int | None = None) -> str:
    """Render bounded history as disambiguation context, not factual evidence."""
    recent = as_langchain_history(history, turns=turns)
    lines = []
    for message in recent:
        speaker = "User" if message.type == "human" else "Assistant"
        lines.append(f"{speaker}: {message.content}")
    return "\n".join(lines)
