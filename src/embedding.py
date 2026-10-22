"""Structure-aware chunking and embedding helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200, model: SentenceTransformer | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model or SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    @staticmethod
    def _kind(document: Any) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        content_type = str(metadata.get("content_type", "")).casefold()
        source = str(metadata.get("source", "")).casefold()
        if content_type == "table":
            return "table"
        if any(term in source for term in ("faq", "frequently asked")):
            return "faq"
        if any(term in source for term in ("api", "endpoint", "swagger", "openapi")):
            return "api"
        if source.endswith((".pdf", ".docx")):
            return "manual"
        return "general"

    def _splitter(self, kind: str) -> RecursiveCharacterTextSplitter:
        settings = {
            "table": (2200, 100), "faq": (1400, 180), "api": (1200, 180),
            "manual": (1000, 200), "general": (self.chunk_size, self.chunk_overlap),
        }
        size, overlap = settings[kind]
        separators = {
            "table": ["\n\n", "\n", "|", " "],
            "faq": ["\n\nQ:", "\n\nQuestion:", "\n\n", "\n", " "],
            "api": ["\n## ", "\n### ", "\n```", "\n\n", "\n", " "],
            "manual": ["\n\n", "\n", ". ", " ", ""],
            "general": ["\n\n", "\n", " ", ""],
        }
        return RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=min(overlap, max(0, size // 4)),
            length_function=len, separators=separators[kind], add_start_index=True,
        )

    @staticmethod
    def _parent_id(document: Any, index: int) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        identity = "|".join(str(metadata.get(key, "")) for key in ("source", "page", "section"))
        return hashlib.sha1((identity or str(index)).encode("utf-8")).hexdigest()[:20]

    def _parent_documents(self, documents: list[Any]) -> list[Document]:
        grouped: dict[tuple[str, str, str], list[Any]] = {}
        order: list[tuple[str, str, str]] = []
        for index, document in enumerate(documents):
            metadata = getattr(document, "metadata", {}) or {}
            key = (str(metadata.get("source", "")), str(metadata.get("page", "")), str(metadata.get("section", "")) or f"document-{index}")
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(document)
        parents: list[Document] = []
        for index, key in enumerate(order):
            members = grouped[key]
            metadata = dict(getattr(members[0], "metadata", {}) or {})
            metadata["parent_id"] = self._parent_id(members[0], index)
            metadata["parent_index"] = index
            text = "\n\n".join(str(member.page_content).strip() for member in members if member.page_content.strip())
            original_parts = [
                str(member.metadata["original_content"])
                for member in members
                if member.metadata.get("original_content") is not None
                and str(member.page_content).strip()
            ]
            if original_parts:
                metadata["original_content"] = "\n\n".join(original_parts)
            parents.append(Document(page_content=text, metadata=metadata))
        return parents

    def chunk_documents(self, documents: list[Any]) -> list[Any]:
        chunks: list[Document] = []
        for parent in self._parent_documents(documents):
            kind = self._kind(parent)
            children = self._splitter(kind).split_documents([parent])
            for child_index, child in enumerate(children):
                heading = parent.metadata.get("heading") or parent.metadata.get("section")
                if heading and child_index > 0 and not str(child.page_content).lstrip().startswith(str(heading)):
                    child.page_content = f"{heading}\n\n{child.page_content}"
                child.metadata.update({
                    "parent_id": parent.metadata["parent_id"],
                    "parent_index": parent.metadata["parent_index"],
                    "child_index": child_index,
                    "chunk_kind": kind,
                })
                chunks.append(child)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} structure-aware chunks.")
        return chunks

    def embed_chunks(self, chunks: list[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    docs = load_all_documents("docs")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    embeddings = pipeline.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
