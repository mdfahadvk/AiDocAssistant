<img width="1774" height="887" alt="AI Assistant RAG Application Flow" src="https://github.com/user-attachments/assets/4fd757ad-012f-4844-84ce-85d73334ed95" />
# MSI AI Assistant

A local Streamlit app for chatting with your documents and managing the indexed files. Documents are split into 1,000-character chunks with 200 characters of overlap, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.

## Quick start

Use Python 3.10 or newer and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the local spaCy English model once:

```powershell
python -m spacy download en_core_web_sm
```

Scanned PDF OCR requires Tesseract OCR to be installed separately and available on
`PATH`. On Windows, install Tesseract and restart the terminal before launching
Streamlit. Digitally-created PDFs use normal text extraction; OCR is used only for
pages where little or no text can be extracted.

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Then launch the browser UI:

```powershell
python -m streamlit run ui.py
```

The server binds to `127.0.0.1`, so the UI is available only from this computer by default.

The app provides **Chat** and **Documents** pages in the sidebar. In Documents, upload one or more supported files and select **Upload and index**. The app stores uploaded files under `docs/uploads/`, chunks and embeds them, and adds the vectors to ChromaDB. In Chat, ask questions about the indexed content. Chat history and follow-up context last only for the current browser session. Use **+ New Chat** to clear the conversation without changing indexed documents.

Supported file types: PDF, TXT, Markdown, CSV, DOCX, JSON, and JSONL. The embedding model is downloaded from Hugging Face the first time it is needed, so initial setup requires internet access. No Hugging Face token is required.

## Document cleanup and answer presentation

Before chunking and embedding, the app applies a conservative local cleanup
layer to LangChain `Document` objects. It removes obvious extraction noise such
as table-of-contents dot leaders, clearly identifiable page-number lines,
repeated page headers/footers, excess whitespace, and duplicate blank lines.
It preserves meaningful numbers, bullets, tables, URLs, technical identifiers,
procedure names, terminology, and section numbering. Recognized section
headings are retained in the document metadata as `section` and `heading` when
they are clear; headings are also carried into nearby chunks where practical.

The answer prompt asks Groq to present supported content cleanly: headings are
used when useful, lists use bullets or numbered steps, and tables are reserved
for structured comparisons. Raw TOC navigation text and extraction artifacts
are not repeated unless relevant to the question. This does not add a separate
formatting LLM call. Citation IDs are deduplicated and normalized in first-use
order (`[1]`, `[2]`, `[3]`). The answer uses cleaned evidence, while **View
Source** is labeled **Retrieved Evidence** and shows the original evidence when
it is available in the index.

After changing cleanup behavior, rebuild the index so existing ChromaDB chunks
are replaced with cleaned content:

```powershell
python app.py --rebuild
```

## Credentials and storage

The Groq API key is required for chat answers and belongs in `.env` as `GROQ_API_KEY`. Chroma indexing does not require the Groq key. The current question, up to six recent conversation turns when needed to resolve a follow-up, and retrieved text passages are sent to Groq. Conversation history stays in the current Streamlit browser session and is not stored in ChromaDB or a database.

By default, ChromaDB uses local persistence in `chroma_store/`. To use Chroma Cloud instead, set `CHROMA_MODE=cloud` and provide `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` in `.env`. Restart Streamlit after changing `.env` settings.

Optional `.env` settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model |
| `MAX_HISTORY_TURNS` | `6` | Recent user/assistant turns used only for follow-up question context |
| `DOCS_DIR` | `docs` | Root folder for uploaded and existing documents |
| `CHROMA_DIR` | `chroma_store` | Local ChromaDB directory |
| `CHROMA_MODE` | `local` | Use `cloud` for Chroma Cloud |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `RETRIEVAL_MAX_DISTANCE` | `0.65` | Maximum Chroma cosine distance accepted as relevant; lower values make the app more cautious about unrelated matches |
| `PROJECT_MAPPING_FILE` | `project_mapping.json` | JSON file mapping project/application names to document filenames |
| `NLP_PROVIDER` | `local` | Query analyzer provider; `local` uses spaCy, while `llm` is reserved for a future provider |
| `SPACY_MODEL` | `en_core_web_sm` | Local spaCy model |
| `NLP_TERMS_FILE` | `config/nlp_terms.json` | Domain vocabulary for project, page, API, database, and technology terms |
| `NLP_MAX_RETRIEVAL_QUERIES` | `5` | Maximum local query expansions per question |
| `ENABLE_QUERY_EXPANSION` | `true` | Enable selective fallback retrieval for short, broad, or weak queries |
| `MAX_RETRIEVAL_QUERIES` | `5` | Maximum expanded retrieval formulations |
| `MIN_INITIAL_RELEVANT_RESULTS` | `1` | Minimum results inside the existing distance cutoff before fallback expansion |
| `RAG_MAX_CONTEXT_CHARS` | `24000` | Maximum retrieved evidence characters sent to Groq |
| `HYBRID_VECTOR_WEIGHT` | `0.65` | Weight given to vector similarity; the remainder is BM25 keyword score |
| `HYBRID_CANDIDATE_K` | `20` | Candidate count retrieved before reranking |
| `ENABLE_RERANKING` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sentence Transformers cross-encoder model |
| `RAG_PARENT_CONTEXT_CHARS` | `6000` | Maximum surrounding parent-section text per retrieved result |

Chat uses a relevance cutoff before asking Groq to draft an answer. If no retrieved passage is close enough, or the passages do not contain the answer, the app asks you to upload the related document instead of answering from general model knowledge. The cutoff is a tuning setting: lower it to decline more often, or raise it if relevant questions are being declined. Follow-up questions can use the most recent `MAX_HISTORY_TURNS` conversational turns to resolve references before document retrieval. That session history stays in Streamlit memory; it is not stored in ChromaDB. Only retrieved document passages are factual evidence for answers.

The Chat page can restrict retrieval by project and, optionally, by document. Project assignments are configured in `project_mapping.json` (or the file named by `PROJECT_MAPPING_FILE`) and are stored as metadata on every newly indexed chunk. Unmapped documents use the `General` project. Rebuild the index after adding or changing mappings so existing chunks receive the new metadata; legacy chunks without project metadata are shown as `General / Unknown` and are never assigned to a configured project.

Query understanding is provided by a modular local spaCy layer. It detects lightweight intent, configured domain terms, entities, follow-up references, and a small set of retrieval query expansions before the existing Chroma search. Extend `config/nlp_terms.json` to add project/application names or technical terms without changing Python code. `QueryAnalyzer` is the provider boundary; `NLP_PROVIDER=local` is the current implementation and an LLM provider can be added later without changing the RAG or UI interfaces. If spaCy or its model is unavailable, the chatbot falls back to the original query and continues normally.

Short and broad questions such as `Introduction?`, `purpose`, and `What does it do?` are normalized and receive a small set of local retrieval formulations. Normal questions use one Chroma retrieval. Expansion runs only for a broad/short query or when the initial retrieval has too few passages inside the existing distance cutoff; it does not add an LLM call. Duplicate chunks are merged, repeated matches receive a small ranking boost, and exact section-heading matches receive an additional boost. The selected project and document filters apply to every formulation. If expansion fails, the initial results and the existing safe no-answer behavior are retained.

When uploading from the Documents page, choose an existing project or select `New project...` and enter a name. The uploaded filename is added to the mapping automatically and is indexed with that project. Re-uploading a filename under another project moves its mapping assignment.

## Adding and replacing documents

New uploads are added without reprocessing other indexed documents. Uploading a file with the same name replaces that upload's previous chunks. Uploaded files and the Chroma index are ignored by Git.

For a full rebuild from every supported file in `DOCS_DIR`, run:

```powershell
python app.py --rebuild
```

This replaces the app's ChromaDB `documents` collection. Keep a backup if you need to preserve an existing index.
If the Documents tab reports legacy chunks without filenames, rebuild once so ChromaDB can record the current source metadata.
