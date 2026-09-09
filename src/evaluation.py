"""
evaluation.py

Phase 17 of the DocuRAG project: a small, honest RAG evaluation harness.

Running the app without errors is not the same as the app answering well —
this module exists specifically so that claim is never made by omission.
It runs a small, hand-built evaluation set (data/evaluation/eval_set.json)
through the REAL pipeline — the same pdf_loader -> chunker -> embeddings ->
vector_store -> retriever -> rag_pipeline code path the Streamlit app
uses — and reports concrete numbers instead of "it seems to work".

Metrics computed:

- Hit@K (retrieval-only, no LLM needed): did at least one of the top-K
  retrieved chunks come from a page the question is actually answerable
  from? A coarse but cheap and meaningful retrieval-quality signal.

- Recall@K (retrieval-only): of ALL the pages that could support an
  answer, what fraction were retrieved within the top-K? Distinguishes
  "found the answer" from "found every relevant page" — a question with
  evidence spread across two pages could Hit@K on just one of them.

- Trap-question handling: a few questions in the eval set have NO
  relevant page at all (relevant_pages: []) — things this document simply
  doesn't discuss. For these, Hit@K/Recall@K don't apply; instead this
  checks whether the system correctly declines to answer rather than
  guessing. This is really a hallucination-control check, using the
  evaluation harness's own machinery rather than a separate one.

- An optional LLM-answer check (skipped automatically if no
  GEMINI_API_KEY is configured): does the generated answer cite pages
  that actually overlap with the expected ones, and does it correctly
  decline on trap questions? This is a coarse proxy for "faithfulness" —
  it confirms the answer is grounded in the right part of the document,
  NOT that its wording is factually flawless. True faithfulness/
  correctness checking would need either human review of every answer, or
  a second LLM call to judge the first one's answer (an "LLM-as-judge"
  setup). Both are more rigor than a portfolio-scale project needs, but
  are worth naming explicitly rather than silently skipping — see
  discuss_evaluation_limitations() below.

IMPORTANT caveat about the retrieval-only trap-question check: FAISS's
IndexFlatIP always returns the top-K nearest vectors, no matter how
dissimilar they actually are — there's no built-in "nothing matched"
outcome unless a min_similarity threshold is supplied. So a trap
question's "correctly_declined" signal is only meaningful in one of two
cases: (a) a min_similarity threshold was passed to evaluate(), or (b) the
LLM-answer check ran and its answer is compared against the exact
NOT_FOUND_MESSAGE text. Without either, expect this signal to under-report
correct behavior — this is exactly the kind of caveat Section 24 asks
projects like this to be upfront about, not paper over.

Depends on: the whole pipeline it evaluates —
src/vector_store.py, src/retriever.py, src/rag_pipeline.py, src/llm.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from .retriever import DEFAULT_TOP_K, retrieve  # when imported as part of the src package
    from .vector_store import VectorStore
    from . import llm
    from . import rag_pipeline
except ImportError:
    from retriever import DEFAULT_TOP_K, retrieve  # when run as a standalone script
    from vector_store import VectorStore
    import llm
    import rag_pipeline


DEFAULT_EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "eval_set.json"


@dataclass
class EvalItem:
    question: str
    expected_info: str          # human-readable reference for manual review — not machine-checked
    relevant_pages: list[int]   # empty list = a "trap" question with no answer in the document


@dataclass
class EvalItemResult:
    question: str
    expected_pages: list[int]
    retrieved_pages: list[int]
    hit: bool | None                        # None for trap questions
    recall: float | None                    # None for trap questions
    correctly_declined: bool | None         # only meaningful for trap questions — see module docstring
    answer: str | None = None               # only populated when run_llm_check=True
    answer_cites_expected_page: bool | None = None  # only populated when run_llm_check=True, non-trap question


def load_eval_set(path: str | Path = DEFAULT_EVAL_SET_PATH) -> list[EvalItem]:
    """Load an evaluation set in the {question, expected_info, relevant_pages} JSON format."""
    with open(path) as f:
        raw_items = json.load(f)
    return [
        EvalItem(
            question=item["question"],
            expected_info=item.get("expected_info", ""),
            relevant_pages=item.get("relevant_pages", []),
        )
        for item in raw_items
    ]


def hit_at_k(retrieved_pages: set[int], expected_pages: list[int]) -> bool:
    """True if at least one retrieved page is among the expected pages."""
    return bool(retrieved_pages.intersection(expected_pages))


def recall_at_k(retrieved_pages: set[int], expected_pages: list[int]) -> float:
    """Fraction of the expected pages that were actually retrieved."""
    if not expected_pages:
        return 0.0
    return len(retrieved_pages.intersection(expected_pages)) / len(expected_pages)


def evaluate(
    eval_items: list[EvalItem],
    vector_store: VectorStore,
    embedding_model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float | None = None,
    run_llm_check: bool = False,
    api_key: str | None = None,
    gemini_model: str = llm.DEFAULT_GEMINI_MODEL,
) -> list[EvalItemResult]:
    """
    Run every item in eval_items through retrieval (and, if requested,
    full answer generation) and compute per-item metrics.

    Args:
        eval_items: e.g. from load_eval_set().
        vector_store / embedding_model: the document's already-built
            retrieval pipeline (see this module's __main__ for how to
            build these from a PDF).
        top_k / min_similarity: passed through to retriever.retrieve().
        run_llm_check: if True, also runs rag_pipeline.answer_question()
            for every item and checks whether the generated answer cites
            the right pages / correctly declines on trap questions. Costs
            one LLM call per item, so it's opt-in.
        api_key / gemini_model: passed through to llm.generate(), only
            used when run_llm_check=True.
    """
    results = []
    for item in eval_items:
        retrieval = retrieve(item.question, vector_store, embedding_model, top_k=top_k, min_similarity=min_similarity)
        retrieved_pages = set(retrieval.pages)

        if item.relevant_pages:
            hit = hit_at_k(retrieved_pages, item.relevant_pages)
            recall = recall_at_k(retrieved_pages, item.relevant_pages)
            correctly_declined = None
        else:
            hit = recall = None
            # Only meaningful if min_similarity was set — see module docstring.
            correctly_declined = retrieval.is_empty

        answer_text = None
        cites_expected = None
        if run_llm_check:
            try:
                qa_result = rag_pipeline.answer_question(
                    item.question, vector_store, embedding_model,
                    top_k=top_k, min_similarity=min_similarity, api_key=api_key, gemini_model=gemini_model,
                )
                answer_text = qa_result.answer
                if item.relevant_pages:
                    cited_pages = set(qa_result.retrieval.pages)
                    cites_expected = bool(cited_pages.intersection(item.relevant_pages))
                else:
                    # The real test for a trap question: did the LLM's own
                    # answer correctly decline? This supersedes the raw
                    # retrieval-based signal above when available.
                    correctly_declined = answer_text.strip() == rag_pipeline.NOT_FOUND_MESSAGE
            except (llm.MissingAPIKeyError, llm.LLMGenerationError) as e:
                answer_text = f"[LLM check skipped: {e}]"

        results.append(EvalItemResult(
            question=item.question,
            expected_pages=item.relevant_pages,
            retrieved_pages=sorted(retrieved_pages),
            hit=hit,
            recall=recall,
            correctly_declined=correctly_declined,
            answer=answer_text,
            answer_cites_expected_page=cites_expected,
        ))
    return results


def summarize(results: list[EvalItemResult]) -> dict:
    """Aggregate per-item results into overall scores."""
    answerable = [r for r in results if r.expected_pages]
    traps = [r for r in results if not r.expected_pages]

    hit_rate = sum(1 for r in answerable if r.hit) / len(answerable) if answerable else None
    avg_recall = sum(r.recall for r in answerable) / len(answerable) if answerable else None
    trap_decline_rate = (
        sum(1 for r in traps if r.correctly_declined) / len(traps) if traps else None
    )

    answered_with_llm = [r for r in answerable if r.answer_cites_expected_page is not None]
    faithfulness_rate = (
        sum(1 for r in answered_with_llm if r.answer_cites_expected_page) / len(answered_with_llm)
        if answered_with_llm else None
    )

    return {
        "num_questions": len(results),
        "num_answerable_questions": len(answerable),
        "num_trap_questions": len(traps),
        "hit_rate": hit_rate,
        "avg_recall": avg_recall,
        "trap_correctly_declined_rate": trap_decline_rate,
        "answer_cites_expected_page_rate": faithfulness_rate,
    }


def discuss_evaluation_limitations() -> str:
    """
    Returns a short, honest discussion of what this evaluation does and
    does not show — printed at the end of the __main__ report so the
    numbers are never presented without their caveats attached.
    """
    return (
        "What this evaluation does NOT show:\n"
        "- Hit@K/Recall@K measure retrieval only. A 'hit' means a relevant\n"
        "  chunk was retrieved, not that the final generated answer was\n"
        "  correct, well-written, or fully faithful to it.\n"
        "- 'answer_cites_expected_page_rate' is a coarse faithfulness proxy:\n"
        "  it checks whether the answer's cited pages overlap with the\n"
        "  expected ones, not whether every sentence in the answer is\n"
        "  factually accurate. True faithfulness/correctness checking needs\n"
        "  human review or a second LLM-as-judge call — out of scope here.\n"
        "- The eval set is small and hand-built for one sample document.\n"
        "  These numbers describe this document and this eval set, not RAG\n"
        "  quality in general.\n"
        "- Trap-question 'correctly_declined' from retrieval alone is only\n"
        "  meaningful if a min_similarity threshold was used — FAISS always\n"
        "  returns its top-K nearest vectors regardless of how irrelevant\n"
        "  they actually are."
    )


if __name__ == "__main__":
    # Quick manual run: `python src/evaluation.py [path_to_pdf] [path_to_eval_set.json]`
    # Defaults to this project's own sample PDF and eval set if omitted.
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

    default_pdf = Path(__file__).parent.parent / "data" / "documents" / "sample_paper.pdf"
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pdf)
    eval_set_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_EVAL_SET_PATH

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

    eval_items = load_eval_set(eval_set_path)
    run_llm = llm.is_configured()
    if not run_llm:
        print("No GEMINI_API_KEY found — running retrieval-only metrics (Hit@K, Recall@K).\n")

    results = evaluate(eval_items, store, model, run_llm_check=run_llm)

    for r in results:
        label = "TRAP" if not r.expected_pages else "Q"
        print(f"[{label}] {r.question}")
        print(f"    expected_pages={r.expected_pages}  retrieved_pages={r.retrieved_pages}")
        if r.hit is not None:
            print(f"    hit={r.hit}  recall={r.recall:.2f}")
        if r.correctly_declined is not None:
            print(f"    correctly_declined={r.correctly_declined}")
        if r.answer is not None:
            print(f"    answer: {r.answer[:100]}")
        print()

    print("=== Summary ===")
    for key, value in summarize(results).items():
        print(f"{key}: {value}")

    print()
    print(discuss_evaluation_limitations())
