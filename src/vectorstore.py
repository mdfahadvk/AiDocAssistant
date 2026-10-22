import hashlib
import math
import os
from pathlib import Path
import re
from typing import Any, Callable
import uuid

import chromadb
from sentence_transformers import SentenceTransformer

from src.data_loader import calculate_file_hash
from src.document_cleaner import clean_documents
from src.embedding import EmbeddingPipeline
from src.project_config import DEFAULT_PROJECT, load_project_mapping, project_for_document


class ChromaVectorStore:
    """Persistent ChromaDB store with Sentence Transformers embeddings."""

    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = str(Path(persist_dir).expanduser())
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model: SentenceTransformer | None = None
        self._bm25_rows: list[dict[str, Any]] | None = None
        self._bm25_tokens: list[list[str]] | None = None
        chroma_mode = os.getenv("CHROMA_MODE", "local").lower()
        if chroma_mode == "cloud":
            try:
                self.client = chromadb.CloudClient()
            except Exception as exc:
                raise RuntimeError(
                    "Could not connect to Chroma Cloud. Set CHROMA_API_KEY, "
                    "CHROMA_TENANT, and CHROMA_DATABASE in .env."
                ) from exc
        elif chroma_mode == "local":
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            raise ValueError("CHROMA_MODE must be 'local' or 'cloud'.")
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _chunk_documents(self, documents: list[Any]) -> list[Any]:
        # Cleanup is local and happens before splitting/embedding for both
        # full rebuilds and individual uploads.
        documents = clean_documents(documents)
        pipeline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )
        return pipeline.chunk_documents(documents)

    @staticmethod
    def _metadata(
        chunk: Any,
        source_key: str,
        source_label: str,
        project: str | None = None,
        file_hash: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": source_label,
            "source_key": source_key,
            "source_label": source_label,
            "project": project or project_for_document(source_label),
        }
        if file_hash:
            metadata["file_hash"] = file_hash
        for key in ("section", "heading", "content_type", "ocr", "line", "parent_id", "parent_index", "child_index", "chunk_kind", "start_index"):
            if key in chunk.metadata:
                metadata[key] = chunk.metadata[key]
        original_content = chunk.metadata.get("original_content")
        if isinstance(original_content, str) and len(original_content) <= 12000:
            metadata["original_content"] = original_content
        page = chunk.metadata.get("page")
        if isinstance(page, (int, float, str)):
            metadata["page"] = page
        return metadata

    def _insert_chunks(
        self,
        chunks: list[Any],
        source_keys: list[str],
        source_labels: list[str],
        projects: list[str] | None = None,
        file_hashes: list[str | None] | None = None,
        batch_size: int = 128,
    ) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_keys = source_keys[start : start + batch_size]
            batch_labels = source_labels[start : start + batch_size]
            batch_projects = (projects or [None] * len(chunks))[start : start + batch_size]
            batch_hashes = (file_hashes or [None] * len(chunks))[start : start + batch_size]
            embeddings = self.model.encode(
                [chunk.page_content for chunk in batch],
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            ids = [
                hashlib.sha256(f"{key}\0{start + offset}".encode("utf-8")).hexdigest()
                for offset, key in enumerate(batch_keys)
            ]
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[chunk.page_content for chunk in batch],
                metadatas=[
                    self._metadata(chunk, key, label, project, file_hash)
                    for chunk, key, label, project, file_hash in zip(
                        batch, batch_keys, batch_labels, batch_projects, batch_hashes
                    )
                ],
            )

    def build_from_documents(self, documents: list[Any]) -> int:
        """Replace the collection contents with chunks from the given documents."""
        if not documents:
            raise ValueError("No documents to index. Add supported files under docs/.")
        chunks = self._chunk_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were created. Check that your documents contain text.")

        self.client.delete_collection("documents")
        self.collection = self.client.create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        source_keys: list[str] = []
        source_labels: list[str] = []
        projects: list[str] = []
        file_hashes: list[str | None] = []
        mapping = load_project_mapping()
        for chunk in chunks:
            source_path = Path(str(chunk.metadata.get("source", "unknown")))
            label = source_path.name or "unknown"
            if source_path.parent.name.casefold() == "uploads":
                key = f"upload:{label.casefold()}"
            else:
                key = f"file:{str(source_path).casefold()}"
            source_keys.append(key)
            source_labels.append(label)
            projects.append(project_for_document(label, mapping))
            try:
                file_hashes.append(calculate_file_hash(source_path))
            except (OSError, ValueError):
                file_hashes.append(None)
        self._insert_chunks(chunks, source_keys, source_labels, projects, file_hashes)
        self._invalidate_retrieval_indexes()
        print(f"[INFO] Indexed {len(chunks)} chunks in {self.persist_dir}")
        return len(chunks)

    def upsert_file(
        self,
        documents: list[Any],
        source_key: str,
        source_label: str,
        progress: Callable[[float], None] | None = None,
        project: str | None = None,
        file_hash: str | None = None,
    ) -> int:
        """Index one uploaded file, replacing any prior chunks for its stable key."""
        if not documents:
            raise ValueError("The uploaded file contains no readable text.")
        chunks = self._chunk_documents(documents)
        if not chunks:
            raise ValueError("No text chunks were created from this file.")

        # Encode before deleting the existing version, so an embedding failure leaves it intact.
        all_embeddings: list[list[float]] = []
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            encoded = self.model.encode(
                [chunk.page_content for chunk in batch],
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            all_embeddings.extend(encoded)
            if progress:
                progress(0.5 * min(1.0, (start + len(batch)) / len(chunks)))

        # Add the replacement set first so an insertion failure leaves the prior upload intact.
        previous_ids = self._previous_upload_ids(source_key, source_label)
        project = project or project_for_document(source_label)
        generation = uuid.uuid4().hex
        new_ids = [
            hashlib.sha256(f"{source_key}\0{generation}\0{index}".encode("utf-8")).hexdigest()
            for index in range(len(chunks))
        ]
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                batch_embeddings = all_embeddings[start : start + batch_size]
                batch_ids = new_ids[start : start + batch_size]
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=[chunk.page_content for chunk in batch],
                    metadatas=[
                        self._metadata(chunk, source_key, source_label, project, file_hash)
                        for chunk in batch
                    ],
                )
                if progress:
                    progress(0.5 + 0.5 * min(1.0, (start + len(batch)) / len(chunks)))
        except Exception:
            self.collection.delete(ids=new_ids)
            raise
        if previous_ids:
            self.collection.delete(ids=previous_ids)
        self._invalidate_retrieval_indexes()
        return len(chunks)

    def _invalidate_retrieval_indexes(self) -> None:
        self._bm25_rows = None
        self._bm25_tokens = None

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9_./:-]*", text.casefold())

    def _ensure_bm25_index(self) -> None:
        if getattr(self, "_bm25_rows", None) is not None and getattr(self, "_bm25_tokens", None) is not None:
            return
        records = self.collection.get(include=["documents", "metadatas"])
        self._bm25_rows = []
        self._bm25_tokens = []
        for text, metadata in zip(records.get("documents") or [], records.get("metadatas") or []):
            row = {"distance": 1.0, "metadata": {"text": text, **(metadata or {})}}
            self._bm25_rows.append(row)
            self._bm25_tokens.append(self._tokens(text + " " + str((metadata or {}).get("section", ""))))

    def _bm25_query(self, query_text: str, where: dict[str, Any] | None, limit: int) -> list[dict[str, Any]]:
        self._ensure_bm25_index()
        rows = self._bm25_rows or []
        tokenized = self._bm25_tokens or []
        query_tokens = self._tokens(query_text)
        if not query_tokens:
            return []
        allowed = [self._matches_where(row["metadata"], where) for row in rows]
        document_frequency: dict[str, int] = {}
        for tokens, is_allowed in zip(tokenized, allowed):
            if is_allowed:
                for token in set(tokens):
                    document_frequency[token] = document_frequency.get(token, 0) + 1
        corpus_size = max(1, sum(allowed))
        average_length = sum(len(tokens) for tokens, ok in zip(tokenized, allowed) if ok) / corpus_size
        scores: list[tuple[float, int]] = []
        for index, (tokens, is_allowed) in enumerate(zip(tokenized, allowed)):
            if not is_allowed:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (corpus_size - df + 0.5) / (df + 0.5))
                length_norm = 1 - 0.75 + 0.75 * len(tokens) / max(1.0, average_length)
                score += idf * frequency * 2.0 / (frequency + 1.2 * length_norm)
            if score > 0:
                scores.append((score, index))
        scores.sort(reverse=True)
        maximum = scores[0][0] if scores else 1.0
        results = []
        for score, index in scores[:limit]:
            result = dict(rows[index])
            result["bm25_score"] = score / maximum
            results.append(result)
        return results

    @staticmethod
    def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(ChromaVectorStore._matches_where(metadata, item) for item in where["$and"])
        if "$or" in where:
            return any(ChromaVectorStore._matches_where(metadata, item) for item in where["$or"])
        return all(metadata.get(key) == value for key, value in where.items())

    def query_hybrid(
        self, query_text: str, top_k: int = 5, where: dict[str, Any] | None = None, candidate_k: int = 20
    ) -> list[dict[str, Any]]:
        """Combine vector similarity and local BM25 keyword retrieval."""
        vector_rows = self.query(query_text, top_k=max(top_k, candidate_k), where=where)
        keyword_rows = self._bm25_query(query_text, where, max(top_k, candidate_k))
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in vector_rows:
            metadata = row.get("metadata") or {}
            key = (str(metadata.get("source_key", "")), str(metadata.get("parent_id", "")), str(metadata.get("text", "")))
            item = merged.setdefault(key, {**row, "vector_score": 0.0, "bm25_score": 0.0})
            distance = float(row.get("distance", 2.0))
            item["vector_score"] = max(item["vector_score"], max(0.0, 1.0 - distance / 2.0))
        for row in keyword_rows:
            metadata = row.get("metadata") or {}
            key = (str(metadata.get("source_key", "")), str(metadata.get("parent_id", "")), str(metadata.get("text", "")))
            item = merged.setdefault(key, {**row, "vector_score": 0.0, "bm25_score": 0.0})
            item["bm25_score"] = max(item["bm25_score"], float(row.get("bm25_score", 0.0)))
        vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.65"))
        keyword_weight = 1.0 - vector_weight
        for item in merged.values():
            item["hybrid_score"] = vector_weight * item["vector_score"] + keyword_weight * item["bm25_score"]
        return sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)[:candidate_k]

    def expand_parent_context(self, results: list[dict[str, Any]], max_chars: int = 6000) -> list[dict[str, Any]]:
        """Attach surrounding parent-section text to each retrieved child."""
        cache: dict[str, str] = {}
        for result in results:
            metadata = result.get("metadata") or {}
            parent_id = metadata.get("parent_id")
            if not parent_id:
                continue
            if parent_id not in cache:
                records = self.collection.get(where={"parent_id": parent_id}, include=["documents", "metadatas"])
                parts = sorted(
                    zip(records.get("documents") or [], records.get("metadatas") or []),
                    key=lambda item: int((item[1] or {}).get("child_index", 0)),
                )
                cache[parent_id] = "\n\n".join(text for text, _ in parts)[:max_chars]
            metadata["context_text"] = cache[parent_id]
        return results

    def find_duplicate_files(
        self, file_hash: str, exclude_source_key: str | None = None
    ) -> list[str]:
        """Return indexed filenames with the same content hash."""
        records = self.collection.get(include=["metadatas"])
        matches: set[str] = set()
        for metadata in records.get("metadatas") or []:
            metadata = metadata or {}
            if metadata.get("file_hash") != file_hash:
                continue
            if exclude_source_key and metadata.get("source_key") == exclude_source_key:
                continue
            matches.add(str(metadata.get("source_label") or metadata.get("source") or "Unknown"))
        return sorted(matches, key=str.casefold)

    def _previous_upload_ids(self, source_key: str, source_label: str) -> list[str]:
        records = self.collection.get(include=["metadatas"])
        matching_ids: list[str] = []
        for record_id, metadata in zip(records.get("ids", []), records.get("metadatas") or []):
            metadata = metadata or {}
            source = str(metadata.get("source", ""))
            source_path = Path(source.replace("\\", "/"))
            is_same_upload = (
                str(metadata.get("source_key", "")) == source_key
                or str(metadata.get("source_label", "")).casefold() == source_label.casefold()
                or (
                    source_path.parent.name.casefold() == "uploads"
                    and source_path.name.casefold() == source_label.casefold()
                )
            )
            if is_same_upload:
                matching_ids.append(record_id)
        return matching_ids

    def index_summary(self) -> tuple[int, list[str]]:
        """Return chunk count and distinct indexed document labels."""
        count = self.collection.count()
        if count == 0:
            return 0, []
        records = self.collection.get(include=["metadatas"])
        sources = sorted(
            {
                str(metadata.get("source_label") or metadata.get("source", "unknown"))
                for metadata in (records.get("metadatas") or [])
            }
        )
        sources = [
            "Legacy chunks (source metadata unavailable)" if source == "unknown" else source
            for source in sources
        ]
        return count, sources

    def project_summary(self) -> dict[str, list[str]]:
        """Return indexed documents grouped by project, treating legacy rows safely."""
        records = self.collection.get(include=["metadatas"])
        grouped: dict[str, set[str]] = {}
        for metadata in records.get("metadatas") or []:
            metadata = metadata or {}
            project = str(metadata.get("project") or "General / Unknown")
            document = metadata.get("source_label") or metadata.get("source") or "Unknown document"
            grouped.setdefault(project, set()).add(Path(str(document).replace("\\", "/")).name)
        return {project: sorted(documents, key=str.casefold) for project, documents in sorted(grouped.items())}

    def query(
        self, query_text: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            raise RuntimeError("ChromaDB is empty. Add documents and rebuild the index.")
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0].tolist()
        matches = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), count),
            include=["documents", "metadatas", "distances"],
            **({"where": where} if where else {}),
        )
        return [
            {
                "distance": distance,
                "metadata": {"text": text, **(metadata or {})},
            }
            for text, metadata, distance in zip(
                matches["documents"][0],
                matches["metadatas"][0],
                matches["distances"][0],
            )
        ]

    def query_many(
        self,
        query_texts: list[str],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve, deduplicate, and rank results for several query formulations."""
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for query_text in query_texts:
            for result in self.query(query_text, top_k=top_k, where=where):
                metadata = result.get("metadata") or {}
                key = (
                    str(metadata.get("source_key", "")),
                    str(metadata.get("page", "")),
                    str(metadata.get("text", "")),
                )
                previous = merged.get(key)
                if previous is None:
                    merged[key] = {**result, "_hits": 1, "_best_distance": result.get("distance", 2.0)}
                else:
                    previous["_hits"] += 1
                    previous["_best_distance"] = min(previous["_best_distance"], result.get("distance", 2.0))
        # Preserve Chroma's original best distance while rewarding agreement between variants.
        return sorted(merged.values(), key=lambda item: item["_best_distance"] - min(0.08, 0.02 * (item["_hits"] - 1)))[:top_k]
