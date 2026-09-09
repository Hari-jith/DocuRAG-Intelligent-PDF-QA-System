"""
retriever.py

Stage 6 of the DocuRAG pipeline: given a user's question and an already-
built VectorStore for the current document, retrieve the most relevant
chunks and assemble them into ready-to-use RAG context with page
citations.

This module deliberately stops right before the LLM call:
- llm.py (Phase 8) is responsible for actually sending a prompt to Gemini.
- rag_pipeline.py (later) wires the full flow together, including web
  retrieval and query routing for hybrid questions.

Pipeline implemented here:
  question -> query embedding -> FAISS search -> (optional relevance
  filter) -> ranked chunks -> formatted context + page citations

Why retrieval comes before generation: an LLM only knows what's in its
training data (which won't include this specific PDF) plus whatever text
we hand it in the prompt. Retrieval is how we find the small, relevant
slice of the document to hand it — without this step the model would have
nothing document-specific to ground its answer in, and would have to guess.

Depends on:
- src/embeddings.py (Phase 5) — for embed_query().
- src/vector_store.py (Phase 6) — for VectorStore and SearchResult.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

try:
    from .embeddings import embed_query  # when imported as part of the src package
    from .vector_store import SearchResult, VectorStore
except ImportError:
    from embeddings import embed_query  # when run as a standalone script
    from vector_store import SearchResult, VectorStore


# Per the project's retrieval guidance: a reasonable starting TOP_K, not a
# universally "correct" value — see notebooks/document_rag_experiments.ipynb
# Section 15 for how changing it trades recall against precision. Once
# config.py exists this default should live there instead.
DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """
    The outcome of retrieving for one question: the ranked (and possibly
    relevance-filtered) chunks, kept alongside the original query text so
    downstream code (llm.py, rag_pipeline.py) doesn't have to pass the
    question around separately.
    """
    query: str
    results: list[SearchResult]

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    @property
    def pages(self) -> list[int]:
        """Distinct page numbers actually retrieved, in ascending order."""
        return sorted({r.chunk.page for r in self.results})

    def to_debug_rows(self) -> list[dict]:
        """
        Plain dicts (chunk_id, page, similarity_score, text) for quick
        inspection — e.g. `pd.DataFrame(retrieval.to_debug_rows())` in a
        notebook. Kept dependency-free here (no pandas import) since this
        module is used by the production app, not just for exploration.
        """
        return [
            {
                "chunk_id": r.chunk.chunk_id,
                "page": r.chunk.page,
                "similarity_score": round(r.similarity_score, 4),
                "text": r.chunk.text,
            }
            for r in self.results
        ]


def filter_by_relevance(results: list[SearchResult], min_similarity: float) -> list[SearchResult]:
    """
    Drop results below a similarity threshold. Kept as a separate function
    (rather than baked into retrieve()) so it can also be reused later,
    e.g. in rag_pipeline.py when deciding whether PDF evidence is strong
    enough to answer from, or whether to fall back to web retrieval.

    No single min_similarity value is "correct" for every document/query —
    cosine similarity scores are relative to a given embedding model and
    document, not an absolute confidence percentage. Treat this as a coarse
    filter, not a calibrated probability.
    """
    return [r for r in results if r.similarity_score >= min_similarity]


def retrieve(
    query: str,
    vector_store: VectorStore,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
) -> RetrievalResult:
    """
    Run the PDF retrieval step for one question.

    Args:
        query: the user's question.
        vector_store: a VectorStore already built for the current document
            (vector_store.py, Phase 6).
        model: the SAME embedding model used to build `vector_store` — a
            mismatched model would silently produce meaningless scores, so
            this is required rather than defaulted/reloaded here.
        top_k: how many chunks to retrieve before any filtering.
        min_similarity: if given, results scoring below this are dropped
            (see filter_by_relevance). Left as None by default — filtering
            too aggressively risks silently discarding chunks that were
            actually useful, especially since scores aren't perfectly
            comparable across different questions.

    Returns:
        A RetrievalResult. If nothing sufficiently relevant is found (an
        empty vector_store, or filtering removes every result), `results`
        is simply an empty list — this is a normal, expected outcome, not
        an error. Callers (llm.py / rag_pipeline.py) are expected to tell
        the user "this wasn't found in the document" in that case, rather
        than letting the LLM guess.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    query_embedding = embed_query(query, model=model)
    results = vector_store.search(query_embedding, top_k=top_k)

    if min_similarity is not None:
        results = filter_by_relevance(results, min_similarity)

    return RetrievalResult(query=query, results=results)


def build_context(retrieval: RetrievalResult) -> str:
    """
    Join retrieved chunks into a single block of LLM-ready context, each
    passage tagged with the page it came from. This inline page-tagging is
    what lets both the LLM (to cite pages in its answer) and our own code
    (format_pdf_sources, below) know exactly which page each piece of
    evidence came from — the mechanism that makes source attribution
    possible at all.

    Returns an empty string if nothing was retrieved; callers should treat
    an empty context as "no PDF evidence available" rather than sending it
    to the LLM as if it were valid (empty) context.
    """
    if retrieval.is_empty:
        return ""
    return "\n\n".join(f"[Page {r.chunk.page}]\n{r.chunk.text}" for r in retrieval.results)


def format_pdf_sources(retrieval: RetrievalResult) -> str:
    """
    A "Sources:" bullet list of the pages actually retrieved, in the format
    the project spec calls for:

        Sources:
        - PDF — Page 4
        - PDF — Page 7

    Always derived from the real RetrievalResult that was used to answer —
    never invented — which is central to avoiding fabricated citations.
    """
    if retrieval.is_empty:
        return "No relevant pages were found in the document."
    return "\n".join(f"- PDF — Page {p}" for p in retrieval.pages)


if __name__ == "__main__":
    # Quick manual test:
    # `python src/retriever.py path/to/file.pdf "a question" [top_k] [min_similarity]`
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
        print('Usage: python retriever.py <path_to_pdf> "<question>" [top_k] [min_similarity]')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]
    cli_top_k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TOP_K
    cli_min_similarity = float(sys.argv[4]) if len(sys.argv) > 4 else None

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

    retrieval = retrieve(question, store, model, top_k=cli_top_k, min_similarity=cli_min_similarity)

    print(f"Question: {question}\n")
    if retrieval.is_empty:
        print("No relevant chunks found.")
    else:
        for row in retrieval.to_debug_rows():
            snippet = row["text"][:90].replace("\n", " ")
            print(f"  page={row['page']}  score={row['similarity_score']}  \"{snippet}...\"")

        print("\n--- Context that would be sent to the LLM ---")
        print(build_context(retrieval))

        print("\n--- Sources ---")
        print(format_pdf_sources(retrieval))
