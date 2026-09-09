"""
rag_pipeline.py

Stages 8, 10, 11 & 12 of the DocuRAG pipeline: question answering, covering
PDF-only, web-only, combined PDF+web questions, and now conversation-aware
follow-up questions.

This file has grown across phases, as planned from the start:
answer_question() (Phase 9) remains the PDF-only building block, unchanged.
Phase 13's query_router.py decides which of three routes a question takes.
Phase 14 added web_retriever.py integration plus the PDF+web fusion prompt.
Phase 16 (this update) adds conversation history support to answer(): a
follow-up question like "How large was it?" gets rewritten into a
standalone question ("How large was the CUFS dataset?") before routing and
retrieval happen, using recent conversation turns to resolve the reference.

IMPORTANT — conversation history informs the QUERY, not the ANSWER:
per the project spec, "conversational history must not replace document
retrieval." History is used ONLY to rewrite an ambiguous follow-up into a
standalone, retrievable question. The actual answer is still generated
from freshly retrieved PDF/web evidence for that standalone question, the
same as every other question — history is never substituted for grounding.

Design choice — no LLM call when nothing was retrieved: if retrieval finds
no relevant chunks at all, this module returns the standard "not found"
message directly, WITHOUT calling Gemini. Two reasons: (1) there is no
context to ground an answer in, so a real LLM call could only guess or
hallucinate — better to just say so; (2) it means a genuinely unanswerable
question doesn't cost an API call or require a configured API key at all.
The same principle now applies to the WEB_ONLY and PDF_AND_WEB routes below.

Design choice — different failure handling for WEB_ONLY vs PDF_AND_WEB when
web search itself fails or isn't configured:
- WEB_ONLY: the router already decided this question has no PDF signal at
  all, so a PDF-only fallback probably can't help. The web search error is
  allowed to propagate, the same way a missing Gemini key propagates from
  answer_question() — surfacing the real problem beats guessing.
- PDF_AND_WEB: the router detected the PDF IS at least partially relevant.
  If web search fails here, this module falls back to answering from PDF
  evidence alone (clearly, not silently) rather than failing a question
  the PDF could partly answer just because the web half didn't work.

Design choice — heuristic pre-check before rewriting a question: rewriting
costs one extra LLM call, so it's only attempted when both (a) conversation
history exists and (b) the question looks like it might be a follow-up
(contains a pronoun like "it"/"this", or is very short). Fully self-
contained questions ("What dataset was used?") skip this step entirely —
consistent with this project's general approach of avoiding LLM calls that
aren't actually needed (see also query_router.py's heuristic-first design).

Depends on:
- src/retriever.py (Phase 7) — retrieve(), build_context(), format_pdf_sources().
- src/web_retriever.py (Phase 12) — search_web(), build_web_context(), format_web_sources().
- src/query_router.py (Phase 13) — classify_query(), QueryRoute.
- src/llm.py (Phase 8) — generate(), and its exception types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sentence_transformers import SentenceTransformer

try:
    from .retriever import (  # when imported as part of the src package
        DEFAULT_TOP_K,
        RetrievalResult,
        build_context,
        format_pdf_sources,
        retrieve,
    )
    from .vector_store import VectorStore
    from .query_router import QueryRoute, classify_query
    from . import llm
    from . import web_retriever
except ImportError:
    from retriever import (  # when run as a standalone script
        DEFAULT_TOP_K,
        RetrievalResult,
        build_context,
        format_pdf_sources,
        retrieve,
    )
    from vector_store import VectorStore
    from query_router import QueryRoute, classify_query
    import llm
    import web_retriever


# Shown whenever the document simply doesn't contain the answer — either
# because nothing relevant was retrieved at all, or because the LLM itself
# decides the retrieved context doesn't actually support an answer. Kept as
# one shared constant so both cases produce identical, predictable text.
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded document."

QA_SYSTEM_PROMPT = f"""You are DocuRAG, a document question-answering assistant.
Answer the user's question using ONLY the information in the PDF CONTEXT below.
Each context passage is labeled with the page number it came from.

Rules:
- If the answer is not supported by the given context, say exactly:
  "{NOT_FOUND_MESSAGE}"
- Do not use any outside knowledge.
- Do not invent page numbers, authors, numbers, or facts that are not in the context.
- When you state a fact, mention which page it came from, e.g. (Page 3).
"""

QA_USER_PROMPT_TEMPLATE = """PDF CONTEXT:
{context}

QUESTION:
{question}

Answer using only the PDF CONTEXT above, and cite page numbers for every claim.
"""


def build_qa_prompt(context: str, question: str) -> str:
    return QA_USER_PROMPT_TEMPLATE.format(context=context, question=question)


@dataclass
class AnswerResult:
    question: str
    answer: str
    retrieval: RetrievalResult
    used_llm: bool  # False when we short-circuited before calling Gemini (nothing relevant was retrieved)

    @property
    def sources_text(self) -> str:
        """The "Sources:" block for this answer, e.g. for display in a UI or CLI."""
        return format_pdf_sources(self.retrieval)


def answer_question(
    question: str,
    vector_store: VectorStore,
    embedding_model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
    api_key: str | None = None,
    gemini_model: str = llm.DEFAULT_GEMINI_MODEL,
) -> AnswerResult:
    """
    Answer a question about the currently loaded PDF, grounded in retrieved
    chunks, with page-level source attribution.

    Args:
        question: the user's question.
        vector_store: VectorStore already built for the current document.
        embedding_model: the SAME model used to build vector_store.
        top_k / min_similarity: passed through to retriever.retrieve().
        api_key / gemini_model: passed through to llm.generate(). If
            api_key is omitted, llm.py loads GEMINI_API_KEY from .env.

    Returns:
        An AnswerResult. If nothing relevant was retrieved, `answer` is
        NOT_FOUND_MESSAGE and `used_llm` is False — no LLM call is made in
        that case (see module docstring).

    Raises:
        ValueError: empty question (from retriever.retrieve()).
        llm.MissingAPIKeyError: relevant context WAS found, but no Gemini
            API key is configured to generate an answer from it.
        llm.LLMGenerationError: the Gemini call failed or returned an
            unusable response.

    Note on error handling: this function lets llm.py's exceptions
    propagate rather than catching them, so the caller (the Streamlit app,
    Phase 11) can show an appropriate on-screen message. Catching and
    silently downgrading them here would hide a real problem (e.g. a
    missing API key) behind a generic-looking answer.
    """
    retrieval = retrieve(question, vector_store, embedding_model, top_k=top_k, min_similarity=min_similarity)

    if retrieval.is_empty:
        return AnswerResult(question=question, answer=NOT_FOUND_MESSAGE, retrieval=retrieval, used_llm=False)

    context = build_context(retrieval)
    user_prompt = build_qa_prompt(context, question)
    answer = llm.generate(QA_SYSTEM_PROMPT, user_prompt, api_key=api_key, model=gemini_model)

    return AnswerResult(question=question, answer=answer, retrieval=retrieval, used_llm=True)


# --------------------------------------------------------------------------
# Phase 14: web-only and hybrid (PDF+web) answering
# --------------------------------------------------------------------------

NOT_FOUND_MESSAGE_WEB = "I could not find this information from the web search results."

WEB_ONLY_SYSTEM_PROMPT = f"""You are DocuRAG, a document question-answering assistant.
Answer the user's question using ONLY the information in the WEB CONTEXT below.
Each passage is labeled with the source title and URL it came from.

Rules:
- If the answer is not supported by the given context, say exactly:
  "{NOT_FOUND_MESSAGE_WEB}"
- Do not use any outside knowledge beyond the given context.
- Do not invent URLs, titles, or facts not present in the given context.
- When you state a fact, reference which source it came from.
"""

WEB_ONLY_USER_PROMPT_TEMPLATE = """WEB CONTEXT:
{context}

QUESTION:
{question}

Answer using only the WEB CONTEXT above, and reference sources for every claim.
"""

# The example format in the project spec (PDF Evidence: / External
# Information:) is encoded directly into this prompt so the LLM structures
# its own output that way, rather than us trying to split a single answer
# into two sections after the fact — the model has the context to know
# which sentence belongs in which section; string-splitting a finished
# answer would not.
HYBRID_SYSTEM_PROMPT = """You are DocuRAG, a document question-answering assistant that can combine
information from the uploaded PDF with supporting information from the web.

You are given PDF CONTEXT (passages from the uploaded document, tagged with
page numbers) and WEB CONTEXT (excerpts from web pages, tagged with source
titles and URLs). Either one may have no relevant content for this question.

Rules:
- Structure your answer using exactly these two labeled sections, in order:

  PDF Evidence:
  <answer using ONLY the PDF CONTEXT, citing page numbers, e.g. (Page 3).
  If the PDF context has nothing relevant, write exactly:
  "Not found in the uploaded document.">

  External Information:
  <answer using ONLY the WEB CONTEXT, referencing source titles. If the web
  context has nothing relevant, write exactly:
  "No relevant web results were found.">

- Never present information from the web as if it came from the PDF, or
  vice versa. Do not mix the two sections together.
- Do not use any knowledge beyond what is given in PDF CONTEXT and WEB CONTEXT.
- Do not invent page numbers, URLs, titles, or facts not present in the
  given context.
"""

HYBRID_USER_PROMPT_TEMPLATE = """PDF CONTEXT:
{pdf_context}

WEB CONTEXT:
{web_context}

QUESTION:
{question}

Answer following the PDF Evidence / External Information structure described in your instructions.
"""


# --------------------------------------------------------------------------
# Phase 16: conversation history -> standalone question rewriting
# --------------------------------------------------------------------------

DEFAULT_MAX_HISTORY_TURNS = 3

# Words that commonly signal a question is referring back to something
# said earlier, rather than being fully self-contained.
_REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those",
    "they", "them", "their", "theirs",
    "he", "him", "his", "she", "her", "hers",
}

CONDENSE_QUESTION_SYSTEM_PROMPT = """You rewrite a user's follow-up question into a standalone
question, using the conversation history to resolve any pronouns or
implicit references (e.g. "it", "that", "this approach").

Rules:
- Preserve the user's intent exactly. Do not add information, assumptions,
  or opinions that aren't implied by the conversation.
- If the question is already standalone and self-contained, return it
  completely unchanged.
- Output ONLY the rewritten question itself — no quotes, no explanation,
  no preamble.
"""

CONDENSE_QUESTION_USER_TEMPLATE = """Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


@dataclass
class ConversationTurn:
    """One past exchange, used only to help rewrite a new follow-up question."""
    question: str
    answer: str


def _looks_like_followup(question: str) -> bool:
    """
    Cheap heuristic: does this question look like it might depend on
    conversation context? Used to skip the extra LLM call for questions
    that are obviously already self-contained.
    """
    words = re.findall(r"[a-zA-Z']+", question.lower())
    if not words:
        return False
    return len(words) <= 4 or any(w in _REFERENCE_WORDS for w in words)


def _format_history(history: list[ConversationTurn], max_turns: int) -> str:
    recent = history[-max_turns:] if max_turns > 0 else history
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in recent)


def resolve_standalone_question(
    question: str,
    history: list[ConversationTurn] | None,
    api_key: str | None = None,
    gemini_model: str = llm.DEFAULT_GEMINI_MODEL,
    max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
) -> str:
    """
    Rewrite `question` into a standalone form if it looks like a follow-up
    referencing prior conversation. Returns `question` unchanged if there's
    no history, the question doesn't look like a follow-up (see
    _looks_like_followup), or the rewrite itself fails — this is a helpful
    enhancement layered on top of retrieval, never something retrieval
    depends on (see module docstring).
    """
    if not history or not _looks_like_followup(question):
        return question

    prompt = CONDENSE_QUESTION_USER_TEMPLATE.format(
        history=_format_history(history, max_history_turns), question=question
    )
    try:
        rewritten = llm.generate(
            CONDENSE_QUESTION_SYSTEM_PROMPT, prompt, api_key=api_key, model=gemini_model, temperature=0.0
        )
    except (llm.MissingAPIKeyError, llm.LLMGenerationError):
        # Fall back to the original question rather than blocking the
        # whole answer over a reformulation failure.
        return question

    return rewritten.strip() or question


@dataclass
class HybridAnswerResult:
    """
    The result of answer() — covers all three routes with one type, so
    callers (app.py) don't need to branch on route to know how to read the
    result. pdf_retrieval is None for a pure WEB_ONLY answer; web_results
    is empty for a pure PDF_ONLY answer.
    """
    question: str            # the original question, exactly as the user typed it
    resolved_question: str   # the standalone form actually used for retrieval/search
    route: QueryRoute
    answer: str
    pdf_retrieval: RetrievalResult | None
    web_results: list = field(default_factory=list)  # list[web_retriever.WebResult]
    used_llm: bool = False

    @property
    def pdf_sources_text(self) -> str | None:
        """The PDF "Sources:" block, or None if this answer used no PDF retrieval at all."""
        if self.pdf_retrieval is None:
            return None
        return format_pdf_sources(self.pdf_retrieval)

    @property
    def web_sources_text(self) -> str | None:
        """The "External Sources:" block, or None if this answer used no web results at all."""
        if not self.web_results:
            return None
        return web_retriever.format_web_sources(self.web_results)


def _answer_web_only(
    original_question: str,
    resolved_question: str,
    api_key: str | None,
    gemini_model: str,
    web_max_results: int,
    web_api_key: str | None,
) -> HybridAnswerResult:
    # Intentionally NOT caught here — see the module docstring's note on
    # why WEB_ONLY lets web search failures propagate rather than falling
    # back to the PDF.
    web_results = web_retriever.search_web(resolved_question, api_key=web_api_key, max_results=web_max_results)

    if not web_results:
        return HybridAnswerResult(
            question=original_question, resolved_question=resolved_question, route=QueryRoute.WEB_ONLY,
            answer=NOT_FOUND_MESSAGE_WEB, pdf_retrieval=None, web_results=[], used_llm=False,
        )

    context = web_retriever.build_web_context(web_results)
    prompt = WEB_ONLY_USER_PROMPT_TEMPLATE.format(context=context, question=resolved_question)
    answer_text = llm.generate(WEB_ONLY_SYSTEM_PROMPT, prompt, api_key=api_key, model=gemini_model)

    return HybridAnswerResult(
        question=original_question, resolved_question=resolved_question, route=QueryRoute.WEB_ONLY,
        answer=answer_text, pdf_retrieval=None, web_results=web_results, used_llm=True,
    )


def _answer_hybrid(
    original_question: str,
    resolved_question: str,
    vector_store: VectorStore,
    embedding_model: SentenceTransformer,
    top_k: int,
    min_similarity: float | None,
    api_key: str | None,
    gemini_model: str,
    web_max_results: int,
    web_api_key: str | None,
) -> HybridAnswerResult:
    pdf_retrieval = retrieve(resolved_question, vector_store, embedding_model, top_k=top_k, min_similarity=min_similarity)

    try:
        web_results = web_retriever.search_web(resolved_question, api_key=web_api_key, max_results=web_max_results)
    except (web_retriever.MissingWebSearchAPIKeyError, web_retriever.WebSearchError):
        # See module docstring: fall back to PDF-only evidence rather than
        # failing a question the PDF can at least partly answer.
        web_results = []

    pdf_context = build_context(pdf_retrieval) if not pdf_retrieval.is_empty else ""
    web_context = web_retriever.build_web_context(web_results)

    if not pdf_context and not web_context:
        return HybridAnswerResult(
            question=original_question, resolved_question=resolved_question, route=QueryRoute.PDF_AND_WEB,
            answer=NOT_FOUND_MESSAGE, pdf_retrieval=pdf_retrieval, web_results=web_results, used_llm=False,
        )

    prompt = HYBRID_USER_PROMPT_TEMPLATE.format(
        pdf_context=pdf_context or "(no relevant PDF content was retrieved)",
        web_context=web_context or "(no web search results were available)",
        question=resolved_question,
    )
    answer_text = llm.generate(HYBRID_SYSTEM_PROMPT, prompt, api_key=api_key, model=gemini_model)

    return HybridAnswerResult(
        question=original_question, resolved_question=resolved_question, route=QueryRoute.PDF_AND_WEB,
        answer=answer_text, pdf_retrieval=pdf_retrieval, web_results=web_results, used_llm=True,
    )


def answer(
    question: str,
    vector_store: VectorStore,
    embedding_model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
    api_key: str | None = None,
    gemini_model: str = llm.DEFAULT_GEMINI_MODEL,
    web_max_results: int = web_retriever.DEFAULT_MAX_RESULTS,
    web_api_key: str | None = None,
    history: list[ConversationTurn] | None = None,
) -> HybridAnswerResult:
    """
    The full hybrid entry point: optionally resolve a follow-up question
    into standalone form using conversation history (Phase 16), classify
    the resulting question (query_router.py), then answer it from the PDF,
    the web, or both.

    Args:
        question / vector_store / embedding_model / top_k / min_similarity:
            same meaning as answer_question().
        api_key / gemini_model: passed through to llm.generate().
        web_max_results / web_api_key: passed through to web_retriever.search_web().
        history: recent conversation turns, oldest first. Used only to
            resolve references in `question` (e.g. "it") into a standalone
            question BEFORE retrieval — retrieval and generation are still
            always grounded fresh for that standalone question, never
            answered from history alone. Pass None (default) or [] to
            disable this entirely.

    Returns:
        A HybridAnswerResult. `question` holds the original text the user
        typed; `resolved_question` holds what was actually searched for
        (identical to `question` if no rewrite happened).

    Raises:
        ValueError: empty question.
        llm.MissingAPIKeyError / llm.LLMGenerationError: an LLM call was
            needed and failed (see answer_question() for when this happens
            on the PDF_ONLY route; the same applies here whenever any
            route reaches an LLM call).
        web_retriever.MissingWebSearchAPIKeyError / web_retriever.WebSearchError:
            only for the WEB_ONLY route — see the module docstring for why
            PDF_AND_WEB instead degrades gracefully to a PDF-only answer.
    """
    resolved_question = resolve_standalone_question(question, history, api_key=api_key, gemini_model=gemini_model)
    routing = classify_query(resolved_question)

    if routing.route == QueryRoute.PDF_ONLY:
        pdf_result = answer_question(resolved_question, vector_store, embedding_model, top_k, min_similarity, api_key, gemini_model)
        return HybridAnswerResult(
            question=question, resolved_question=resolved_question, route=routing.route,
            answer=pdf_result.answer, pdf_retrieval=pdf_result.retrieval, web_results=[], used_llm=pdf_result.used_llm,
        )

    if routing.route == QueryRoute.WEB_ONLY:
        return _answer_web_only(question, resolved_question, api_key, gemini_model, web_max_results, web_api_key)

    return _answer_hybrid(
        question, resolved_question, vector_store, embedding_model, top_k, min_similarity,
        api_key, gemini_model, web_max_results, web_api_key,
    )


if __name__ == "__main__":
    # Quick manual test: `python src/rag_pipeline.py path/to/file.pdf "a question"`
    import sys

    try:
        from .pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from .chunker import chunk_pages, make_document_id
        from .metadata_extractor import extract_metadata
        from .embeddings import load_embedding_model, embed_chunks, EmbeddingModelError
    except ImportError:
        from pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from chunker import chunk_pages, make_document_id
        from metadata_extractor import extract_metadata
        from embeddings import load_embedding_model, embed_chunks, EmbeddingModelError

    if len(sys.argv) < 3:
        print('Usage: python rag_pipeline.py <path_to_pdf> "<question>" ["<earlier_question>" "<earlier_answer>"]')
        sys.exit(1)

    pdf_path, question_arg = sys.argv[1], sys.argv[2]
    history_arg = [ConversationTurn(question=sys.argv[3], answer=sys.argv[4])] if len(sys.argv) >= 5 else None

    try:
        pages = load_pdf_pages(pdf_path)
    except (PDFLoadError, NoExtractableTextError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    meta = extract_metadata(pdf_path)
    doc_id = make_document_id(pdf_path)
    chunks = chunk_pages(pages, document_id=doc_id, boilerplate_lines=set(meta.boilerplate_lines))

    try:
        model = load_embedding_model()
    except EmbeddingModelError as e:
        print(e)
        sys.exit(1)

    chunks, chunk_embeddings = embed_chunks(chunks, model=model)
    store = VectorStore.build(chunks, chunk_embeddings)

    try:
        result = answer(question_arg, store, model, history=history_arg)
    except (llm.MissingAPIKeyError, llm.LLMGenerationError,
             web_retriever.MissingWebSearchAPIKeyError, web_retriever.WebSearchError) as e:
        print(f"Could not generate an answer: {e}")
        sys.exit(1)

    print(f"Question: {result.question}")
    if result.resolved_question != result.question:
        print(f"Resolved to: {result.resolved_question}")
    print(f"Route: {result.route.value}\n")
    print(f"Answer:\n{result.answer}\n")
    if result.pdf_sources_text:
        print(f"Sources:\n{result.pdf_sources_text}")
    if result.web_sources_text:
        print(f"\nExternal Sources:\n{result.web_sources_text}")
    print(f"\n(used_llm={result.used_llm})")
