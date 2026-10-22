import os
import re
from dotenv import load_dotenv
from src.vectorstore import ChromaVectorStore
from src.project_config import load_project_mapping
from langchain_groq import ChatGroq

from src.chat_memory import history_for_prompt, needs_query_rewrite, rewrite_standalone_query
from src.nlp.factory import QueryAnalyzerFactory
from src.evaluation.confidence import AnswerConfidenceService
from src.evaluation.models import unavailable_confidence

load_dotenv()


def normalize_query_spacing(query: str) -> str:
    """Make compact CamelCase/acronym terms searchable like spaced terms."""
    query = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", query)
    query = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", query)
    return re.sub(r"\s+", " ", query).strip()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "chroma_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-120b",
        vectorstore: ChromaVectorStore | None = None,
    ):
        self.vectorstore = vectorstore or ChromaVectorStore(persist_dir, embedding_model)
        self._reranker = None
        self.query_analyzer = QueryAnalyzerFactory.create()
        self.confidence_service = AnswerConfidenceService()
        if self.vectorstore.collection.count() == 0:
            raise RuntimeError("ChromaDB is empty. Add documents in the Documents tab first.")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        history: list | None = None,
        project: str | None = None,
        document: str | None = None,
        infer_project: bool = True,
    ) -> dict:
        try:
            analysis = self.query_analyzer.analyze(query, history)
        except Exception:
            analysis = None
        try:
            rewritten_query = normalize_query_spacing(
                rewrite_standalone_query(self.llm, query, history)
            )
        except Exception:
            rewritten_query = normalize_query_spacing(query)
        if analysis is not None and rewritten_query != normalize_query_spacing(query):
            try:
                rewritten_analysis = self.query_analyzer.analyze(rewritten_query, history)
                analysis.normalized_query = rewritten_analysis.normalized_query
                analysis.retrieval_queries = rewritten_analysis.retrieval_queries
                analysis.entities.update(rewritten_analysis.entities)
                analysis.project = analysis.project or rewritten_analysis.project
            except Exception:
                pass
        initial_query = normalize_query_spacing(
            analysis.normalized_query if analysis and analysis.normalized_query else rewritten_query
        )
        expansion_queries = [
            normalize_query_spacing(item)
            for item in (analysis.retrieval_queries if analysis else [])
            if normalize_query_spacing(item).casefold() != initial_query.casefold()
        ]
        try:
            conversation_context = (
                history_for_prompt(history) if needs_query_rewrite(query) else ""
            )
        except Exception:
            conversation_context = ""
        inferred_project = analysis.project if analysis else None
        effective_project = project or (inferred_project if infer_project else None)
        effective_document = document or (analysis.document if analysis else None)
        exhaustive_query = bool(
            re.search(r"\b(objective|objectives|goal|goals|all|each|every|list|enumerate|requirements)\b", query, re.I)
            or (analysis and any(item in {"list", "summary"} for item in analysis.query_types))
        )
        retrieval_top_k = max(top_k, 12 if exhaustive_query else top_k)
        where = None
        if effective_project:
            if effective_document:
                where = {"$and": [{"project": effective_project}, {"source_label": effective_document}]}
            else:
                # Include mapped filenames so indexes created before project
                # metadata was added remain searchable without exposing other docs.
                mapped_documents = load_project_mapping().get(effective_project, [])
                project_filters = [{"project": effective_project}]
                project_filters.extend({"source_label": name} for name in mapped_documents)
                where = project_filters[0] if len(project_filters) == 1 else {"$or": project_filters}
        # Keep the existing retrieval and relevance cutoff unchanged.
        max_distance = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.75"))
        parent_context_chars = max(1000, int(os.getenv("RAG_PARENT_CONTEXT_CHARS", "6000")))
        if exhaustive_query:
            parent_context_chars = max(parent_context_chars, 12000)
        initial_results = self._retrieve([initial_query], retrieval_top_k, where, parent_context_chars)
        initial_results = self._rank_results(initial_results, self._exact_terms(query, analysis), max_distance)
        initial_relevant = [r for r in initial_results if r.get("_rank_score", r.get("distance", 2.0)) <= max_distance]
        enable_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").casefold() not in {"0", "false", "no"}
        minimum = max(1, int(os.getenv("MIN_INITIAL_RELEVANT_RESULTS", "1")))
        broad = bool(analysis and (analysis.is_broad_query or analysis.expansion_required))
        use_expansion = enable_expansion and bool(expansion_queries) and (broad or len(initial_relevant) < minimum)
        if use_expansion:
            try:
                results = self._retrieve([initial_query, *expansion_queries], retrieval_top_k, where, parent_context_chars)
            except Exception:
                results = initial_results
                use_expansion = False
        else:
            results = initial_results
        results = self._rerank_results(results, initial_query)
        exact_terms = self._exact_terms(query, analysis)
        results = self._rank_results(results, exact_terms, max_distance)
        relevant = [r for r in results if r.get("_rank_score", r.get("distance", 2.0)) <= max_distance]
        if analysis:
            analysis.retrieval_queries = [initial_query, *expansion_queries] if use_expansion else [initial_query]
            analysis.expansion_required = use_expansion
        if not relevant:
            return {"answer": self._no_answer(effective_project, effective_document), "sources": [], "confidence": unavailable_confidence("Confidence unavailable because no relevant evidence was retrieved.").as_dict(), "analysis": analysis.as_dict() if analysis else {}}

        sources = []
        sources_by_location = {}
        context_parts = []
        max_context_chars = max(1000, int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000")))
        context_length = 0
        for result in relevant:
            metadata = result.get("metadata") or {}
            text = metadata.get("context_text") or metadata.get("text", "")
            if not text:
                continue
            document = metadata.get("source_label") or metadata.get("source") or "Unknown document"
            document = os.path.basename(str(document).replace("\\", "/"))
            page_index = metadata.get("page")
            page = page_index
            if document.casefold().endswith(".pdf") and isinstance(page_index, int):
                page = page_index + 1  # PyPDFLoader stores zero-based page indexes.
            location = (document.casefold(), str(page_index))
            if location not in sources_by_location:
                original_text = metadata.get("original_content") or text
                source = {
                    "id": len(sources) + 1,
                    "document": document,
                    "page": page,
                    "page_index": page_index,
                    "section": metadata.get("section") or metadata.get("heading"),
                    "heading": metadata.get("heading"),
                    "text": text,
                    "chunks": [original_text],
                    "project": metadata.get("project"),
                    "distance": result.get("distance"),
                }
                sources_by_location[location] = source
                sources.append(source)
            else:
                source = sources_by_location[location]
                original_text = metadata.get("original_content") or text
                if original_text in source["chunks"]:
                    continue
                source["chunks"].append(original_text)
            context_parts.append(f"[{source['id']}] {text}")
            context_length += len(text)
            if context_length >= max_context_chars:
                break

        if not context_parts:
            return {"answer": self._no_answer(effective_project, effective_document), "sources": [], "confidence": unavailable_confidence("Confidence unavailable because no evidence text was available.").as_dict(), "analysis": analysis.as_dict() if analysis else {}}

        excerpts = "\n\n".join(context_parts)
        completeness_instruction = """
For this question, provide an exhaustive list. If the evidence contains numbered or bulleted items, reproduce every item supported by the evidence; do not summarize, merge, or omit items. Preserve the source order and use a numbered list. Count the items in the evidence before answering and make sure the response contains the same complete set.
""" if exhaustive_query else ""
        prompt = f"""Answer the user's question using only the supplied document excerpts. The excerpts are the complete evidence available. Do not use outside knowledge, guess, or invent project details.

{completeness_instruction}

Citation rules:
- Cite claims with only the supplied citation numbers, formatted [1], [2], etc.
- Never invent a citation number or cite a source that was not supplied.
- Put citations next to the claim they support.
- Use citation numbers in the order they first appear in your answer, without gaps or duplicates in the source list.
- If the excerpts are relevant but only partially answer the question, give the supported part and clearly say what is missing. If no excerpt is relevant, reply exactly: '{self._no_answer()}' and do not include citations.

Answer formatting:
- Answer the question directly and stay concise but complete.
- Use a short Markdown heading only when it improves readability.
- Use bullets for unordered lists, numbered steps for procedures, and a table only when comparing structured values.
- Treat repeated headers/footers, standalone page numbers, dot leaders, and raw table-of-contents navigation lines as extraction noise. Do not reproduce them unless the user specifically asks about the document's structure.
- Do not repeat filenames, page numbers, or source metadata in the answer unless they directly answer the question.
- Preserve technical names, procedure names, URLs, meaningful numbers, headings, and table values.
- For an objectives question, return only the supported project objectives. Do not append requirements, benefits, implementation details, or unrelated sections unless the user asks for them.
- Do not use internal terms such as "excerpts", "retrieved chunks", or "supplied sources" in the answer. Refer to the available documentation naturally instead.

Conversation history for reference resolution only (not factual evidence):
{conversation_context or "(none)"}
Use prior user turns only to identify what pronouns such as "it" refer to. Prior assistant answers may be wrong and must never be used as evidence. Treat harmless spacing, capitalization, and abbreviations as possible variations of names when the retrieved excerpts make the match clear; do not require an exact text match. Answer from relevant related passages, while distinguishing different applications that have similar names, similar systems, and teams. Carefully associate each status or condition with the exact item named in the passage, and do not invent missing details. If the excerpts provide related information but do not fully answer the question, give the supported information and say what the documents do not specify. If there is no relevant support, use the safe no-answer response.

Current user question: {normalize_query_spacing(query)}

Supplied sources and excerpts:
{excerpts}

Answer:"""
        response = self.llm.invoke([prompt])
        answer = str(response.content).strip()
        answer = re.sub(
            r"\s*\(\s*based on the supplied excerpts\s*\)",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"\b(?:the\s+)?(?:supplied|retrieved)\s+excerpts\b",
            "the available documentation",
            answer,
            flags=re.IGNORECASE,
        )
        # Some Groq models emit alternate numeric citation brackets.
        # Normalize those citations before validating them against retrieved IDs.
        answer = re.sub(r"[\u3010\u3014](\d+)[\u3011\u3015]", r"[\1]", answer)
        allowed_ids = {str(source["id"]) for source in sources}
        if set(re.findall(r"\[(\d+)\]", answer)) - allowed_ids:
            answer = re.sub(
                r"\[(\d+)\]",
                lambda match: match.group(0) if match.group(1) in allowed_ids else "",
                answer,
            ).strip()
        if self._no_answer() in answer:
            return {"answer": self._no_answer(), "sources": [], "confidence": unavailable_confidence("Confidence unavailable because the answer contains no supported content.").as_dict(), "analysis": analysis.as_dict() if analysis else {}}
        answer, cited_sources, citation_mapping = self._normalize_citations(answer, sources)
        cited_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
        if not cited_ids:
            return {"answer": self._no_answer(), "sources": [], "confidence": unavailable_confidence("Confidence unavailable because the answer has no valid citations.").as_dict(), "analysis": analysis.as_dict() if analysis else {}}
        confidence_chunks = []
        for result in relevant:
            metadata = result.get("metadata") or {}
            source_label = os.path.basename(str(metadata.get("source_label") or metadata.get("source") or "Unknown document").replace("\\", "/"))
            page_index = metadata.get("page")
            location = (source_label.casefold(), str(page_index))
            source = sources_by_location.get(location)
            if source:
                confidence_chunks.append({"distance": result.get("distance"), "source_id": citation_mapping.get(source["id"]), "text": metadata.get("text", "")})
        confidence_service = getattr(self, "confidence_service", None)
        if confidence_service is None:
            confidence_service = AnswerConfidenceService()
        confidence = confidence_service.calculate_confidence(
            answer,
            confidence_chunks,
            cited_sources,
            project=effective_project,
            document=effective_document,
        )
        return {"answer": answer, "sources": cited_sources, "confidence": confidence.as_dict(), "analysis": analysis.as_dict() if analysis else {}}

    def _retrieve(
        self,
        queries: list[str],
        top_k: int,
        where: dict | None,
        parent_context_chars: int | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval and merge duplicate chunks across query variants."""
        merged = {}
        for retrieval_query in queries:
            kwargs = {"where": where} if where else {}
            hybrid_query = getattr(self.vectorstore, "query_hybrid", None)
            if callable(hybrid_query):
                candidate_k = max(top_k * 4, int(os.getenv("HYBRID_CANDIDATE_K", "20")))
                rows = hybrid_query(retrieval_query, top_k=top_k, candidate_k=candidate_k, **kwargs)
            else:
                rows = self.vectorstore.query(retrieval_query, top_k=top_k, **kwargs)
            for row in rows:
                metadata = row.get("metadata") or {}
                key = (str(metadata.get("source_key") or metadata.get("source_label") or metadata.get("source") or ""), str(metadata.get("page", "")), str(metadata.get("text", "")))
                if key not in merged:
                    merged[key] = {**row, "_hits": 1, "_best_distance": row.get("distance", 2.0)}
                else:
                    merged[key]["_hits"] += 1
                    merged[key]["_best_distance"] = min(merged[key]["_best_distance"], row.get("distance", 2.0))
        results = list(merged.values())
        expand_parent = getattr(self.vectorstore, "expand_parent_context", None)
        if callable(expand_parent):
            context_chars = parent_context_chars or max(1000, int(os.getenv("RAG_PARENT_CONTEXT_CHARS", "6000")))
            results = expand_parent(results, max_chars=context_chars)
        return results

    def _rerank_results(self, results: list[dict], query: str) -> list[dict]:
        """Rerank the hybrid candidate set with a cross-encoder when enabled."""
        if not results or os.getenv("ENABLE_RERANKING", "true").casefold() in {"0", "false", "no"}:
            return results
        if not hasattr(self.vectorstore, "query_hybrid"):
            return results
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                self._reranker = CrossEncoder(model_name)
            pairs = [
                (query, str((result.get("metadata") or {}).get("context_text") or (result.get("metadata") or {}).get("text", "")))
                for result in results
            ]
            scores = self._reranker.predict(pairs)
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            return sorted(results, key=lambda item: item.get("rerank_score", -float("inf")), reverse=True)
        except Exception:
            # Retrieval remains available if the optional model cannot be downloaded or loaded.
            return results

    @staticmethod
    def _normalize_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict], dict[int, int]]:
        """Renumber cited sources in first-use order and remove duplicates."""
        old_ids: list[int] = []
        for value in re.findall(r"\[(\d+)\]", answer):
            old_id = int(value)
            if old_id not in old_ids and any(source["id"] == old_id for source in sources):
                old_ids.append(old_id)
        mapping = {old_id: index for index, old_id in enumerate(old_ids, 1)}

        def replace(match: re.Match[str]) -> str:
            old_id = int(match.group(1))
            return f"[{mapping[old_id]}]" if old_id in mapping else ""

        normalized = re.sub(r"\[(\d+)\]", replace, answer)
        cited_sources: list[dict] = []
        for source in sources:
            old_id = source["id"]
            if old_id in mapping:
                cited_source = dict(source)
                cited_source["id"] = mapping[old_id]
                cited_sources.append(cited_source)
        return normalized, cited_sources, mapping

    @staticmethod
    def _exact_terms(query: str, analysis) -> list[str]:
        if not analysis or not (analysis.is_short_query or "section_lookup" in analysis.query_types):
            return []
        return [re.sub(r"[?!.]", "", query).strip()]

    @staticmethod
    def _rank_results(results: list[dict], exact_terms: list[str], max_distance: float) -> list[dict]:
        for result in results:
            metadata = result.get("metadata") or {}
            text = str(metadata.get("text", ""))
            hits = max(1, int(result.get("_hits", 1)))
            score = float(result.get("_best_distance", result.get("distance", 2.0))) - min(0.08, 0.02 * (hits - 1))
            # A keyword-only match can be valuable even when its vector score is weak.
            if result.get("bm25_score", 0.0) >= 0.35:
                score = min(score, max_distance * 0.9)
            for term in exact_terms:
                if term and re.search(rf"(?im)(?<!\w){re.escape(term)}(?!\w)", text):
                    # Heading matches are strong lexical evidence even when the
                    # semantic vector is weaker for a one-word section query.
                    score -= 0.15
            result["_rank_score"] = score
        return sorted(
            results,
            key=lambda item: (-item.get("rerank_score", -float("inf")), item.get("_rank_score", item.get("distance", 2.0))),
        )

    @staticmethod
    def _no_answer(project: str | None = None, document: str | None = None) -> str:
        if project or document:
            scope = project or document
            if project and document:
                scope = f"{project} / {document}"
            return (
                f"I couldn't find enough information to answer that confidently in {scope}. "
                "Try asking with more detail or select a different application or document."
            )
        return (
            "I couldn't find enough information to answer that confidently. "
            "Try asking with more detail or select a different application or document."
        )

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    result = rag_search.search_and_summarize(query, top_k=3)
    print("Answer:", result["answer"])
    for source in result["sources"]:
        print(f"[{source['id']}] {source['document']}")
