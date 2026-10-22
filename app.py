import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_loader import load_all_documents
from src.search import RAGSearch
from src.vectorstore import ChromaVectorStore


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions about your local documents.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the local index from the documents."
    )
    args = parser.parse_args()

    docs_dir = Path(os.getenv("DOCS_DIR", "docs"))
    index_dir = os.getenv("CHROMA_DIR", "chroma_store")
    if args.rebuild and not any(docs_dir.rglob("*")):
        raise SystemExit(f"No documents found in {docs_dir.resolve()}.")

    store = ChromaVectorStore(index_dir)
    if args.rebuild or store.collection.count() == 0:
        docs = load_all_documents(docs_dir)
        if not docs:
            raise SystemExit(f"No supported documents found in {docs_dir.resolve()}.")
        store.build_from_documents(docs)

    rag = RAGSearch(
        persist_dir=index_dir,
        llm_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        vectorstore=store,
    )
    print("MSI AI Assistant is ready. Ask a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                result = rag.search_and_summarize(question)
                print(f"\nAnswer: {result['answer']}")
                for source in result.get("sources", []):
                    page = f"\u2014 Page {source['page']}" if source.get("page") not in (None, "") else ""
                    print(f"[{source['id']}] {source['document']}{page}")
            except Exception as exc:
                print(f"Could not answer: {exc}")


if __name__ == "__main__":
    main()
