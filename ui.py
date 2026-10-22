import os
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.chat_memory import ensure_chat_session, reset_chat_session
from src.data_loader import SUPPORTED_EXTENSIONS, calculate_file_hash, load_document_file
from src.project_config import assign_document_to_project, canonical_project_name, load_project_mapping
from src.search import RAGSearch
from src.vectorstore import ChromaVectorStore


st.set_page_config(page_title="MSI AI Assistant", page_icon="Ã°Å¸â€œÅ¡", layout="wide")
load_dotenv()

WORKSPACE_USERS = {"mohammed.fahad", "vinayg", "rakesh"}


def authentication_required() -> bool:
    return os.getenv("AUTH_REQUIRED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enforce_browser_auth() -> None:
    """Require the ERP browser session cookies before displaying the app UI."""
    if not authentication_required():
        return

    cookies = getattr(st.context, "cookies", {})
    if cookies.get("hashkey") and cookies.get("teammsiuid"):
        return

    components.html(
        """
        <script>
        (() => {
            const redirectUrl = "https://erp.msisurfaces.com/nouser.html";
            const parentDocument = window.parent.document;
            try {
                window.parent.location.replace(redirectUrl);
            } catch (error) {
                const link = parentDocument.createElement("a");
                link.href = redirectUrl;
                link.target = "_top";
                link.rel = "noopener";
                parentDocument.body.appendChild(link);
                link.click();
            }
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )
    st.stop()


def has_workspace_access() -> bool:
    """Allow workspace navigation for approved users when auth is enabled."""
    if not authentication_required():
        return True
    cookies = getattr(st.context, "cookies", {})
    user_id = str(cookies.get("teammsiuid", "")).strip().casefold()
    return user_id in WORKSPACE_USERS


enforce_browser_auth()
ensure_chat_session(st.session_state)
workspace_access = has_workspace_access()
if not workspace_access:
    st.session_state.active_page = "Chat"


@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=os.getenv("CHROMA_DIR", "chroma_store"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )


@st.cache_resource(show_spinner=False)
def get_rag_search() -> RAGSearch:
    return RAGSearch(
        llm_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        vectorstore=get_vector_store(),
    )


def save_and_index_upload(
    uploaded_file,
    store: ChromaVectorStore,
    docs_dir: Path,
    progress=None,
    project: str | None = None,
) -> int:
    # Uploaded names are untrusted. Keep only the basename and save into an app-owned folder.
    filename = Path(uploaded_file.name.replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")

    upload_dir = docs_dir / "uploads"
    staging_dir = docs_dir / ".ui_upload_staging"
    upload_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staging_dir / f"{uuid.uuid4().hex}{suffix}"
    target_path = upload_dir / filename

    try:
        staged_path.write_bytes(uploaded_file.getvalue())
        file_hash = calculate_file_hash(staged_path)
        source_key = f"upload:{filename.casefold()}"
        duplicates = store.find_duplicate_files(file_hash, exclude_source_key=source_key)
        if duplicates:
            raise ValueError(
                "This file has already been indexed as: " + ", ".join(duplicates)
            )
        documents = load_document_file(staged_path)
        if not documents:
            raise ValueError("The file contains no readable text.")

        chunk_count = store.upsert_file(
            documents,
            source_key=source_key,
            source_label=filename,
            file_hash=file_hash,
            project=project,
            progress=progress,
        )
        staged_path.replace(target_path)
        return chunk_count
    finally:
        staged_path.unlink(missing_ok=True)


st.title("MSI AI Assistant")
st.caption("Intelligent application support")

vector_store = None
vector_store_error = None
try:
    vector_store = get_vector_store()
except Exception as exc:
    vector_store_error = str(exc)

st.markdown(
    """<style>
    /* Keep the assistant surface clean by hiding Streamlit's runtime toolbar. */
    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        --rag-sidebar-bg: #f8fafc;
        --rag-sidebar-text: #1f2937;
        --rag-sidebar-muted: #64748b;
        --rag-sidebar-border: #e2e8f0;
        --rag-sidebar-accent: #2563eb;
        --rag-sidebar-active: #eff6ff;
        --rag-sidebar-hover: #f1f5f9;
        background: var(--rag-sidebar-bg);
        color: var(--rag-sidebar-text);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.9rem 0.75rem 0.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
        min-height: 2.45rem;
        justify-content: flex-start;
        border-radius: 0.6rem;
        text-align: left;
        font-size: 0.82rem;
        font-weight: 600;
        transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        border: 1px solid var(--rag-sidebar-border) !important;
        background: #ffffff !important;
        color: var(--rag-sidebar-text) !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        border-color: #bfdbfe !important;
        background: var(--rag-sidebar-active) !important;
        color: var(--rag-sidebar-accent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        border: 1px solid #bfdbfe !important;
        background: var(--rag-sidebar-active) !important;
        color: #1d4ed8 !important;
        box-shadow: inset 3px 0 var(--rag-sidebar-accent);
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] {
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--rag-sidebar-muted) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {
        border-color: var(--rag-sidebar-border) !important;
        background: var(--rag-sidebar-hover) !important;
        color: var(--rag-sidebar-text) !important;
    }
    [data-testid="stSidebar"] .rag-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.15rem 0.35rem 0.8rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid var(--rag-sidebar-border);
    }
    [data-testid="stSidebar"] .rag-brand-mark {
        display: grid;
        place-items: center;
        width: 2.25rem;
        height: 2.25rem;
        flex: 0 0 2.25rem;
        border: 1px solid #bfdbfe;
        border-radius: 0.65rem;
        background: #eff6ff;
        color: #2563eb;
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] .rag-brand-title {
        font-size: 0.98rem;
        font-weight: 720;
        line-height: 1.2;
    }
    [data-testid="stSidebar"] .rag-brand-caption {
        margin-top: 0.22rem;
        color: var(--rag-sidebar-muted);
        font-size: 0.74rem;
    }
    [data-testid="stSidebar"] .rag-section-label {
        margin: 0.85rem 0 0.35rem;
        color: #475569;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    [data-testid="stSidebar"] .rag-sidebar-stats {
        display: grid;
        gap: 0.05rem;
        padding: 0.15rem 0 0.25rem;
    }
    [data-testid="stSidebar"] .rag-sidebar-stat {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 1.9rem;
        color: #475569;
        font-size: 0.81rem;
    }
    [data-testid="stSidebar"] .rag-sidebar-stat strong {
        color: var(--rag-sidebar-text);
        font-size: 0.84rem;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stSidebar"] .rag-sidebar-stat-icon {
        display: none;
        width: 1.25rem;
        text-align: center;
    }
    [data-testid="stSidebar"] .rag-kb-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.65rem 0.7rem;
        border: 1px solid var(--rag-sidebar-border);
        border-radius: 0.65rem;
        background: #ffffff;
        color: var(--rag-sidebar-text);
        font-size: 0.75rem;
    }
    [data-testid="stSidebar"] .rag-status-icon {
        flex: 0 0 0.8rem;
        font-size: 0.82rem;
        line-height: 1;
        text-align: center;
    }
    [data-testid="stSidebar"] .rag-user-card {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.7rem 0.2rem 0.1rem;
        margin-top: 0.6rem;
        border-top: 1px solid var(--rag-sidebar-border);
    }
    [data-testid="stSidebar"] .rag-user-avatar {
        display: grid;
        place-items: center;
        width: 2rem;
        height: 2rem;
        flex: 0 0 2rem;
        border: 1px solid #cbd5e1;
        border-radius: 50%;
        background: #ffffff;
        color: var(--rag-sidebar-text);
        font-size: 0.8rem;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .rag-user-label {
        color: var(--rag-sidebar-muted);
        font-size: 0.68rem;
    }
    [data-testid="stSidebar"] .rag-user-name {
        margin-top: 0.1rem;
        font-size: 0.82rem;
        color: var(--rag-sidebar-text);
        font-weight: 650;
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            --rag-sidebar-bg: #111827;
            --rag-sidebar-text: #f8fafc;
            --rag-sidebar-muted: #a7b2c4;
            --rag-sidebar-border: #334155;
            --rag-sidebar-accent: #60a5fa;
            --rag-sidebar-active: #1e3a5f;
            --rag-sidebar-hover: #1f2937;
        }
        [data-testid="stSidebar"] .rag-brand-mark,
        [data-testid="stSidebar"] .rag-kb-status,
        [data-testid="stSidebar"] .rag-user-avatar,
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            background: #1f2937 !important;
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] .rag-brand-mark {
            border-color: #334155;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background: var(--rag-sidebar-active) !important;
            border-color: #3b82f6 !important;
            color: #bfdbfe !important;
        }
        [data-testid="stSidebar"] .rag-section-label {
            color: #a7b2c4;
        }
        [data-testid="stSidebar"] .rag-sidebar-stat {
            color: #cbd5e1;
        }
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] {
        width: 5.25rem !important;
        min-width: 5.25rem !important;
        max-width: 5.25rem !important;
        flex-basis: 5.25rem !important;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-brand-copy,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-section-label,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-sidebar-stat-label,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-sidebar-stat-count,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-status-label,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-user-copy {
        display: none !important;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-brand,
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-user-card {
        justify-content: center;
        padding-left: 0;
        padding-right: 0;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] [data-testid^="stBaseButton"] {
        justify-content: center;
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-sidebar-stat {
        justify-content: center;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-sidebar-stat-icon {
        display: inline-block;
    }
    body:has(.rag-sidebar-compact) [data-testid="stSidebar"] .rag-kb-status {
        justify-content: center;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
    }
    </style>""",
    unsafe_allow_html=True,
)
with st.sidebar:
    compact_sidebar = st.session_state.get("compact_sidebar", False)
    if compact_sidebar:
        st.markdown('<span class="rag-sidebar-compact"></span>', unsafe_allow_html=True)
    st.markdown(
        """<div class="rag-brand">
            <div class="rag-brand-mark" title="MSI AI Assistant">&#128218;</div>
            <div class="rag-brand-copy">
                <div class="rag-brand-title">MSI AI Assistant</div>
                <div class="rag-brand-caption">Intelligent application support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "\u2795" if compact_sidebar else "+ New Chat",
        key="new_chat",
        type="secondary",
        use_container_width=True,
        help="Start a fresh conversation. Indexed documents stay available.",
    ):
        reset_chat_session(st.session_state)
        st.rerun()
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Chat"
    if workspace_access:
        st.markdown('<div class="rag-section-label">ASSISTANT</div>', unsafe_allow_html=True)
        if st.button(
            "\U0001F4AC" if compact_sidebar else "\U0001F4AC   Chat",
            key="navigate_chat",
            type="primary" if st.session_state.active_page == "Chat" else "tertiary",
            use_container_width=True,
            help="Chat with your indexed documents",
        ):
            st.session_state.active_page = "Chat"
            st.rerun()
        if st.button(
            "\U0001F4C4" if compact_sidebar else "\U0001F4C4   Applications",
            key="navigate_documents",
            type="primary" if st.session_state.active_page == "Documents" else "tertiary",
            use_container_width=True,
            help="Manage application knowledge",
        ):
            st.session_state.active_page = "Documents"
            st.rerun()

    st.markdown('<div class="rag-section-label">KNOWLEDGE</div>', unsafe_allow_html=True)
    stats_slot = st.empty()
    status_slot = st.empty()
    user_slot = st.empty()

    if st.button(
        "\u203a" if compact_sidebar else "\u2039   Collapse",
        key="toggle_compact_sidebar",
        help="Expand sidebar" if compact_sidebar else "Collapse to icons",
        use_container_width=True,
    ):
        st.session_state.compact_sidebar = not compact_sidebar
        st.rerun()

selected_page = st.session_state.active_page

if vector_store is None:
    st.error(f"The assistant is temporarily unavailable: {vector_store_error}")
elif selected_page == "Chat":
    st.subheader("How can I help you?")
    chunk_count = vector_store.collection.count()
    if chunk_count == 0:
        st.info("Application knowledge is not available yet. Add knowledge from the Applications page to get started.")
    indexed_projects = vector_store.project_summary() if chunk_count else {}
    configured_projects = load_project_mapping()
    all_applications_label = "All Applications"
    if st.session_state.get("selected_project") == "All Projects":
        st.session_state.selected_project = all_applications_label
    project_options = [
        all_applications_label,
        *sorted(set(configured_projects) | set(indexed_projects), key=str.casefold),
    ]
    selected_project = st.selectbox(
        "Application",
        project_options,
        key="selected_project",
        help="Restrict retrieval to one indexed project.",
    )
    selected_document = None
    if workspace_access and selected_project != all_applications_label:
        if selected_project not in indexed_projects:
            st.info(
                f"{selected_project} is configured but has no indexed documents yet. "
                "Update project_mapping.json to match the document filenames, then rebuild the index."
            )
        document_options = [
            "All Documents",
            *indexed_projects.get(selected_project, []),
        ]
        selected_document_option = st.selectbox(
            "Document",
            document_options,
            key="selected_document",
        )
        if not selected_document_option.startswith("All "):
            selected_document = selected_document_option

    # Declaring chat_input outside tabs keeps it pinned at the bottom of the page.
    input_placeholder = (
        "Ask me anything about this application..."
        if selected_project != all_applications_label
        else "Ask me about MSI applications..."
    )
    question = st.chat_input(input_placeholder, disabled=chunk_count == 0)
    if question:
        try:
            prior_history = list(st.session_state.chat_history)
        except Exception:
            prior_history = []
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.scroll_chat_to_latest = True
        try:
            with st.spinner("Reviewing application knowledge and preparing an answer..."):
                result = get_rag_search().search_and_summarize(
                    question,
                    history=prior_history,
                    project=None if selected_project == all_applications_label else selected_project,
                    document=selected_document,
                    infer_project=selected_project != all_applications_label,
                )
            if isinstance(result, str):
                result = {"answer": result, "sources": []}
        except Exception as exc:
            result = {"answer": f"I couldn't answer that question: {exc}", "sources": []}
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "confidence": result.get("confidence"),
                "analysis": result.get("analysis", {}),
            }
        )
        try:
            st.session_state.chat_history.extend(
                [HumanMessage(content=question), AIMessage(content=result["answer"])]
            )
        except Exception:
            # A session-memory failure must not discard the response or break chat.
            pass

    for message_index, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources", []) if message["role"] == "assistant" else []
            confidence = message.get("confidence") if message["role"] == "assistant" else None
            if sources:
                with st.expander("References", expanded=False):
                    for source in sources:
                        citation_id = source["id"]
                        page = source.get("page")
                        section = source.get("section") or source.get("heading")
                        location = []
                        if page not in (None, ""):
                            location.append(f"Page {page}")
                        if section:
                            location.append(f"Section: {section}")
                        location_text = f" — {' · '.join(location)}" if location else ""
                        st.markdown(f"**[{citation_id}] {source['document']}**{location_text}")
                        source_key = f"view_source_{message_index}_{citation_id}"
                        if st.button("View Source", key=source_key):
                            st.session_state[f"{source_key}_open"] = True
                        if st.session_state.get(f"{source_key}_open", False):
                            st.markdown(f"**Document:** {source['document']}")
                            if page not in (None, ""):
                                st.markdown(f"**Page:** {page}")
                            if section:
                                st.markdown(f"**Section:** {section}")
                            st.markdown("**Retrieved Evidence:**")
                            for chunk_text in source.get("chunks", [source["text"]]):
                                st.code(chunk_text, language=None)
                if confidence and confidence.get("score") is not None:
                    label = f"{confidence['score']}% Confidence"
                    with st.expander(label, expanded=False):
                        st.markdown(f"**Answer Confidence**  \n{confidence['score']}% — {confidence['level']}")
                        for factor, value in (
                            ("Retrieval relevance", confidence.get("retrieval_relevance")),
                            ("Groundedness", confidence.get("groundedness")),
                            ("Citation coverage", confidence.get("citation_coverage")),
                            ("Evidence consistency", confidence.get("evidence_consistency")),
                            ("Scope match", confidence.get("scope_match")),
                        ):
                            st.write(f"{factor}: {value * 100:.0f}%" if value is not None else f"{factor}: Unknown")
                        st.caption(confidence.get("explanation", ""))
                else:
                    st.caption("Confidence unavailable")
    # Scroll only after a new answer is added, keeping the newest turn visible.
    st.markdown(
        '<div id="chat-latest-marker" style="height: 1px; margin-bottom: 7rem;"></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.pop("scroll_chat_to_latest", False):
        components.html(
            """<script>
            (() => {
                const scrollToLatest = () => {
                    const marker = window.parent.document.getElementById("chat-latest-marker");
                    if (marker) marker.scrollIntoView({behavior: "smooth", block: "end"});
                };
                requestAnimationFrame(() => requestAnimationFrame(scrollToLatest));
                setTimeout(scrollToLatest, 250);
            })();
            </script>""",
            height=0,
            scrolling=False,
        )

elif vector_store is not None:
    st.subheader("Manage Application Knowledge")
    st.write("Upload application guidance and supporting material to keep the assistant’s knowledge current.")
    docs_dir = Path(os.getenv("DOCS_DIR", "docs")).expanduser()
    accepted_types = sorted(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS)
    configured_projects = load_project_mapping()
    indexed_projects_for_upload = vector_store.project_summary()
    upload_project_options = [
        "General",
        *sorted(
            (set(configured_projects) | set(indexed_projects_for_upload)) - {"General"},
            key=str.casefold,
        ),
        "New project...",
    ]

    upload_project_choice = st.selectbox(
        "Application",
        upload_project_options,
        key="upload_project_choice",
        help="Assign all uploaded files in this batch to an existing or new project.",
    )
    new_project_name = ""
    if upload_project_choice == "New project...":
        new_project_name = st.text_input(
            "New project name",
            key="new_project_name",
            help="Enter a name for the new project.",
        )

    with st.form("document_upload_form", clear_on_submit=True):
        uploads = st.file_uploader(
        "Application knowledge files",
            type=accepted_types,
            accept_multiple_files=True,
            help="PDF, TXT, Markdown, CSV, DOCX, JSON, and JSONL are supported.",
        )
        submitted = st.form_submit_button("Upload and index", type="primary")

    if submitted:
        if not uploads:
            st.warning("Choose at least one document to upload.")
        else:
            upload_project = new_project_name.strip() or (
                "" if upload_project_choice == "New project..." else upload_project_choice
            )
            if not upload_project:
                st.warning("Enter a project name before uploading.")
                st.stop()
            upload_project = canonical_project_name(upload_project, configured_projects)
            for uploaded_file in uploads:
                progress = st.progress(0, text=f"Processing {Path(uploaded_file.name).name}Ã¢â‚¬Â¦")
                try:
                    chunks_added = save_and_index_upload(
                        uploaded_file,
                        vector_store,
                        docs_dir,
                        project=upload_project,
                        progress=lambda value: progress.progress(
                            int(value * 100), text=f"Chunking, embedding, and indexing {Path(uploaded_file.name).name}Ã¢â‚¬Â¦"
                        ),
                    )
                    assign_document_to_project(uploaded_file.name, upload_project)
                    progress.progress(100, text=f"Indexed {Path(uploaded_file.name).name}")
                    st.success(
                        f"{Path(uploaded_file.name).name}: indexed {chunks_added} chunks. "
                        "Uploading this filename again replaces its indexed chunks."
                    )
                except Exception as exc:
                    progress.empty()
                    st.error(f"Could not index {Path(uploaded_file.name).name}: {exc}")

    try:
        indexed_chunks, indexed_sources = vector_store.index_summary()
        indexed_projects = vector_store.project_summary()
        st.metric("Indexed chunks", indexed_chunks)
        st.markdown("**Indexed documents**")
        if indexed_projects:
            for project, documents in indexed_projects.items():
                st.write(f"**{project}**")
                for source in documents:
                    st.write(f"- {source}")
            if "Legacy chunks (source metadata unavailable)" in indexed_sources:
                st.warning(
                    "Some existing chunks have no stored filename metadata. Run "
                    "`python app.py --rebuild` once to rebuild them from the files in DOCS_DIR."
                )
        else:
            st.caption("No documents are indexed yet.")
    except Exception as exc:
        st.error(f"Could not read the application knowledge index: {exc}")


# Fill sidebar status and counts after page work so uploads are reflected immediately.
indexed_chunks = 0
indexed_sources = []
index_summary_error = None
if vector_store is not None:
    try:
        indexed_chunks, indexed_sources = vector_store.index_summary()
    except Exception as exc:
        index_summary_error = str(exc)

stats_available = vector_store is not None and index_summary_error is None
indexed_document_count = len(indexed_sources) if stats_available else "N/A"
indexed_chunk_count = indexed_chunks if stats_available else "N/A"
with stats_slot.container():
    st.markdown(
        f"""<div class="rag-sidebar-stats">
            <div class="rag-sidebar-stat" title="Documents: {indexed_document_count}">
                <span class="rag-sidebar-stat-icon">&#128196;</span>
                <span class="rag-sidebar-stat-label">Documents</span>
                <strong class="rag-sidebar-stat-count">{indexed_document_count}</strong>
            </div>
            <div class="rag-sidebar-stat" title="Knowledge Base: {indexed_chunk_count}">
                <span class="rag-sidebar-stat-icon">&#128203;</span>
                <span class="rag-sidebar-stat-label">Knowledge Base</span>
                <strong class="rag-sidebar-stat-count">{indexed_chunk_count}</strong>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

if vector_store_error or index_summary_error:
    status_text = "Assistant Unavailable"
    status_color = "#f87171"
    status_icon = "\u26a0"
elif indexed_chunks:
    status_text = "Assistant Ready"
    status_color = "#4ade80"
    status_icon = "\u25cf"
else:
    status_text = "Assistant Ready"
    status_color = "#fbbf24"
    status_icon = "\u25cb"
with status_slot.container():
    st.markdown(
        f"""<div class="rag-kb-status" title="{status_text}">
            <span class="rag-status-icon" style="color:{status_color}">{status_icon}</span>
            <span class="rag-status-label">{status_text}</span>
        </div>""",
        unsafe_allow_html=True,
    )
with user_slot.container():
    st.markdown(
        """<div class="rag-user-card" title="User: Mohammed">
            <div class="rag-user-avatar">M</div>
            <div class="rag-user-copy">
                <div class="rag-user-label">User</div>
                <div class="rag-user-name">Mohammed</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


