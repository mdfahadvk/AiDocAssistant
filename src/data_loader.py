import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


LOADERS: dict[str, Any] = {
    ".pdf": None,
    ".docx": None,
    ".txt": None,
    ".md": None,
    ".csv": None,
}
SUPPORTED_EXTENSIONS = frozenset((*LOADERS, ".json", ".jsonl"))


def calculate_file_hash(path: str | Path) -> str:
    """Return a stable SHA-256 hash without loading the whole file in memory."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pdf(path: Path) -> list[Document]:
    """Extract PDF text, tables, and OCR text for image-only pages."""
    import pymupdf

    documents: list[Document] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text("text").strip()
            ocr_used = False
            if len(text) < 50:
                try:
                    text_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                    text = page.get_text("text", textpage=text_page).strip()
                    ocr_used = bool(text)
                except Exception as exc:
                    logger.warning("OCR failed for %s page %s: %s", path.name, page_number + 1, exc)

            table_text: list[str] = []
            try:
                tables = page.find_tables()
                for table_number, table in enumerate(tables.tables, start=1):
                    rows = table.extract()
                    if rows:
                        table_text.append(
                            "Table %s\n%s" % (
                                table_number,
                                "\n".join(
                                    "| " + " | ".join(str(cell or "").strip() for cell in row) + " |"
                                    for row in rows
                                ),
                            )
                        )
            except Exception as exc:
                logger.warning("Table extraction failed for %s page %s: %s", path.name, page_number + 1, exc)

            combined = "\n\n".join(part for part in [text, *table_text] if part).strip()
            if combined:
                documents.append(
                    Document(
                        page_content=combined,
                        metadata={
                            "source": str(path),
                            "page": page_number,
                            "ocr": ocr_used,
                            "content_type": "pdf_page",
                        },
                    )
                )
    return documents


def _iter_docx_blocks(document: Any):
    """Yield DOCX paragraphs and tables in their original order."""
    from docx.document import Document as DocumentObject
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl

    parent = document.element.body if isinstance(document, DocumentObject) else document._tc
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _load_docx(path: Path) -> list[Document]:
    from docx import Document as WordDocument

    word_document = WordDocument(path)
    results: list[Document] = []
    heading = "Document"
    for block in _iter_docx_blocks(word_document):
        if hasattr(block, "text"):
            text = block.text.strip()
            if not text:
                continue
            style = str(getattr(getattr(block, "style", None), "name", ""))
            if style.casefold().startswith("heading"):
                heading = text
                continue
            results.append(
                Document(
                    page_content=f"{heading}\n\n{text}",
                    metadata={"source": str(path), "section": heading, "content_type": "paragraph"},
                )
            )
        else:
            rows = [
                "| " + " | ".join(cell.text.replace("\n", " ").strip() for cell in row.cells) + " |"
                for row in block.rows
            ]
            if rows:
                results.append(
                    Document(
                        page_content=f"{heading}\n\nTable\n" + "\n".join(rows),
                        metadata={"source": str(path), "section": heading, "content_type": "table"},
                    )
                )
    return results


def load_document_file(file_path: str | Path) -> list[Document]:
    """Load one supported document file into LangChain documents."""
    path = Path(file_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in LOADERS:
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(path)})]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(value, ensure_ascii=False, indent=2)
        return [Document(page_content=text, metadata={"source": str(path)})]

    documents = []
    failures = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"{path.name}, line {line_number}: {exc.msg}")
            continue
        documents.append(
            Document(
                page_content=json.dumps(value, ensure_ascii=False),
                metadata={"source": str(path), "line": line_number},
            )
        )
    for failure in failures:
        logger.warning("Skipped malformed JSONL: %s", failure)
    return documents


def load_all_documents(data_dir: str | Path) -> list[Document]:
    """Load supported documents recursively, reporting all parse failures."""
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"Document directory not found: {data_path}. Create it and add documents."
        )

    documents: list[Document] = []
    failures: list[str] = []
    for path in sorted(
        file_path
        for file_path in data_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        try:
            documents.extend(load_document_file(path))
        except Exception as exc:
            failures.append(f"{path}: {exc}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Could not load one or more documents:\n{details}")
    return documents
