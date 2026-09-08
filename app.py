"""
app.py

The DocuRAG Streamlit application. This is the entry point that wires
together every module built in src/ into a working UI:

  upload PDF -> extract + clean + chunk -> embed -> build FAISS index
             -> generate structured summary
             -> answer questions using the PDF and/or the web, with
                page-level and web sources clearly kept separate

Layout (per the project spec): a left panel with the document's structured
summary, and a right panel with the chat interface for asking questions.

Phase 15 update: chat calls rag_pipeline.answer() (the Phase 14 hybrid
entry point — query routing + PDF and/or web retrieval + fusion) instead of
the PDF-only answer_question() used through Phase 11, and renders
"Sources:" / "External Sources:" as clearly separate blocks.

Phase 16 update: chat history is now passed into rag_pipeline.answer(), so
a follow-up question like "How large was it?" gets resolved against recent
turns before retrieval — see rag_pipeline.py's module docstring for why
this only affects the QUERY, never substitutes for grounding the ANSWER.

Run with: `streamlit run app.py` (from the project root, so the `src/`
import path below resolves correctly).
"""

import hashlib
import sys
from pathlib import Path

import streamlit as st

# src/ modules use plain (non-package) imports internally (see e.g. each
# module's own `except ImportError: from x import y` fallback), so we add
# src/ to sys.path here rather than requiring a src/__init__.py package.
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

import chunker
import embeddings
import llm
import metadata_extractor
import pdf_loader
import rag_pipeline
import summarizer
import vector_store
import web_retriever


st.set_page_config(page_title="DocuRAG", page_icon="📄", layout="wide")

_SESSION_DEFAULTS = {
    "file_hash": None,
    "vector_store": None,
    "metadata": None,
    "summary": None,
    "summary_error": None,
    "processing_error": None,
    "chat_history": [],  # list of {"role": "user"|"assistant", "content": str, "sources": str | None}
}
for _key, _default in _SESSION_DEFAULTS.items():
    st.session_state.setdefault(_key, _default)


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedding_model():
    """
    Cached across the whole app session (not just one document) via
    st.cache_resource, so the model is loaded from disk once per server
    process rather than on every rerun or every question — reloading a
    model per question would be needlessly slow (see project spec Section 30).
    """
    return embeddings.load_embedding_model()


def process_document(filename: str, file_bytes: bytes) -> None:
    """Run the full pipeline for a newly uploaded PDF and store results in session_state."""
    st.session_state.processing_error = None

    try:
        pages = pdf_loader.load_pdf_pages(file_bytes)
    except pdf_loader.PDFLoadError as e:
        st.session_state.processing_error = f"Could not read this PDF: {e}"
        return
    except pdf_loader.NoExtractableTextError as e:
        st.session_state.processing_error = f"{e}"
        return

    try:
        metadata = metadata_extractor.extract_metadata(file_bytes)
    except metadata_extractor.PDFLoadError:
        # Extraction already succeeded above, so this is a soft failure in
        # the metadata step specifically — fall back to minimal metadata
        # rather than blocking the whole app over it.
        metadata = metadata_extractor.DocumentMetadata(
            num_pages=len(pages), title=None, title_source="not found",
            authors=[], authors_source="not found",
            document_type=summarizer.NOT_IDENTIFIED, document_type_signals=[],
            headings=[], boilerplate_lines=[],
        )

    document_id = chunker.make_document_id(filename)
    chunks = chunker.chunk_pages(pages, document_id=document_id, boilerplate_lines=set(metadata.boilerplate_lines))

    if not chunks:
        st.session_state.processing_error = "No usable text was found to index in this document."
        return

    try:
        model = get_embedding_model()
    except embeddings.EmbeddingModelError as e:
        st.session_state.processing_error = str(e)
        return

    chunks, chunk_embeddings = embeddings.embed_chunks(chunks, model=model)

    try:
        store = vector_store.VectorStore.build(chunks, chunk_embeddings)
    except vector_store.VectorStoreError as e:
        st.session_state.processing_error = str(e)
        return

    summary = None
    summary_error = None
    if llm.is_configured():
        try:
            summary = summarizer.generate_summary(chunks, metadata)
        except (llm.MissingAPIKeyError, llm.LLMGenerationError, summarizer.SummarizationError) as e:
            summary_error = f"Could not generate the AI summary: {e}"
    else:
        summary_error = (
            "No GEMINI_API_KEY configured — showing basic document info only. "
            "Add a key to a .env file to enable the full AI-generated summary "
            "(see .env.example)."
        )

    st.session_state.vector_store = store
    st.session_state.metadata = metadata
    st.session_state.summary = summary
    st.session_state.summary_error = summary_error
    st.session_state.chat_history = []  # a new document starts a fresh conversation


def render_summary_panel() -> None:
    st.subheader("📄 Document Summary")

    metadata = st.session_state.metadata
    if metadata is None:
        st.info("Upload a PDF to see its summary here.")
        return

    st.markdown(f"**Title:** {metadata.title or summarizer.NOT_IDENTIFIED}")
    authors_text = ", ".join(metadata.authors) if metadata.authors else summarizer.NOT_IDENTIFIED
    st.markdown(f"**Authors:** {authors_text}")
    st.markdown(f"**Pages:** {metadata.num_pages}")
    st.markdown(f"**Document type:** {metadata.document_type}")

    summary = st.session_state.summary
    if summary is None:
        if st.session_state.summary_error:
            st.warning(st.session_state.summary_error)
        return

    st.divider()
    st.markdown(f"**Objective:** {summary.objective}")
    st.markdown(f"**Problem Statement:** {summary.problem_statement}")
    st.markdown("**Detailed Summary:**")
    st.write(summary.detailed_summary)
    st.markdown(f"**Methodology:** {summary.methodology}")

    if summary.important_concepts:
        st.markdown("**Important Concepts:** " + ", ".join(summary.important_concepts))

    st.markdown(f"**Datasets / Data Sources:** {summary.datasets}")
    st.markdown(f"**Key Findings:** {summary.key_findings}")
    st.markdown(f"**Limitations:** {summary.limitations}")
    st.markdown(f"**Conclusion:** {summary.conclusion}")

    if summary.keywords:
        st.markdown("**Keywords:** " + ", ".join(summary.keywords))


def render_chat_panel() -> None:
    st.subheader("💬 Ask the Document")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("pdf_sources"), message.get("web_sources"))

    question = st.chat_input("Ask a question about the uploaded PDF...")
    if not question:
        return

    st.session_state.chat_history.append(
        {"role": "user", "content": question, "pdf_sources": None, "web_sources": None}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            answer_text = "Please upload a PDF first — I don't have a document to search yet."
            pdf_sources = web_sources = None
            st.markdown(answer_text)
        else:
            with st.spinner("Searching the document and the web, then thinking..."):
                answer_text, pdf_sources, web_sources = _answer_safely(question)
            st.markdown(answer_text)
            _render_sources(pdf_sources, web_sources)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer_text, "pdf_sources": pdf_sources, "web_sources": web_sources}
    )


def _render_sources(pdf_sources: str | None, web_sources: str | None) -> None:
    """
    Renders PDF and web sources as two visually distinct blocks, per the
    project spec's requirement to never mix the two. Either, both, or
    neither may be present depending on which route answered the question.
    """
    if pdf_sources:
        st.caption(f"**Sources:**\n\n{pdf_sources}")
    if web_sources:
        st.caption(f"**External Sources:**\n\n{web_sources}")


def _build_conversation_history() -> list:
    """
    Pairs up consecutive user/assistant messages from session_state into
    rag_pipeline.ConversationTurn objects. If the current (unanswered)
    question has already been appended to chat_history by the time this
    runs, it's naturally excluded — its "assistant" reply doesn't exist yet
    to pair with, so the pairing loop just leaves it dangling and moves on.
    """
    turns = []
    pending_question = None
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question is not None:
            turns.append(rag_pipeline.ConversationTurn(question=pending_question, answer=message["content"]))
            pending_question = None
    return turns


def _answer_safely(question: str) -> tuple[str, str | None, str | None]:
    """
    Runs rag_pipeline.answer() (query routing + PDF and/or web retrieval +
    fusion, with conversation-aware follow-up resolution) and translates
    its typed exceptions into plain chat messages, rather than letting
    Streamlit show a raw traceback for an expected, recoverable situation
    (missing key, API failure, web search unavailable, etc. — see project
    spec Section 28).

    Returns (answer_text, pdf_sources_text, web_sources_text). The latter
    two are None when that source wasn't used for this answer, or on
    failure.
    """
    try:
        model = get_embedding_model()
        history = _build_conversation_history()
        result = rag_pipeline.answer(question, st.session_state.vector_store, model, history=history)
        return result.answer, result.pdf_sources_text, result.web_sources_text
    except llm.MissingAPIKeyError:
        return (
            "Answering this needs the Gemini API, but no GEMINI_API_KEY is "
            "configured. Add one to a .env file to enable this (see .env.example).",
            None, None,
        )
    except llm.LLMGenerationError as e:
        return f"Sorry, I couldn't generate an answer right now: {e}", None, None
    except embeddings.EmbeddingModelError as e:
        return f"Sorry, the embedding model isn't available: {e}", None, None
    except web_retriever.MissingWebSearchAPIKeyError:
        return (
            "This question needs a web search, but no TAVILY_API_KEY is "
            "configured. Add one to a .env file to enable web-augmented "
            "answers (see .env.example), or try asking something the PDF "
            "alone can answer.",
            None, None,
        )
    except web_retriever.WebSearchError as e:
        return f"Sorry, the web search failed: {e}", None, None


def main() -> None:
    st.title("📚 DocuRAG — Intelligent PDF Analysis & Hybrid RAG")
    st.caption("Upload a PDF to get a structured summary and ask grounded questions about it.")

    if llm.is_configured():
        st.sidebar.success("Gemini API key detected.")
    else:
        st.sidebar.warning(
            "No GEMINI_API_KEY found. Summaries and answers that need the "
            "LLM won't work until one is added to a .env file."
        )

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        # Streamlit reruns this whole script on every interaction (e.g. every
        # chat message) — only reprocess when the uploaded file has actually
        # changed, instead of re-embedding the same document every time.
        if file_hash != st.session_state.file_hash:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                process_document(uploaded_file.name, file_bytes)
            st.session_state.file_hash = file_hash

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)

    left_col, right_col = st.columns(2)
    with left_col:
        render_summary_panel()
    with right_col:
        render_chat_panel()


main()
