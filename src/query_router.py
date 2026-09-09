"""
query_router.py

Stage 11 of the DocuRAG pipeline: decide whether a question should be
answered from the PDF, from the web, or from both, before any retrieval or
generation happens. This decision feeds rag_pipeline.py (Phase 14), which
will call retriever.py and/or web_retriever.py based on the route chosen
here.

Design choice — a heuristic phrase-matching router, not an LLM-based one:
The project spec explicitly asks for "a simple and understandable routing
mechanism", not "a complicated agent framework". A deterministic, phrase-
based classifier is:
  - free and instant — no extra LLM call (and therefore no extra API cost
    or latency) just to decide how to answer a question,
  - fully explainable — the exact phrases that triggered a route are
    available for debugging or display, unlike an LLM's internal reasoning,
  - predictable — the same question always routes the same way.
The trade-off is that it's less flexible than an LLM classifier for oddly-
phrased questions. If evaluation (Phase 17) shows this heuristic routing
too many questions incorrectly, swapping in an LLM-based classifier later
is a contained change — rag_pipeline.py only depends on getting a
QueryRoute back, not on how it was produced.

Routing logic: look for two independent signals in the question text —
phrases suggesting the question is about the document itself ("this
paper", "the authors", ...) and phrases suggesting a need for information
beyond it ("current", "latest", "compared to other", ...). Both present ->
PDF_AND_WEB. Only web signals -> WEB_ONLY. Otherwise -> PDF_ONLY, since the
PDF is the primary knowledge source by default (per the project spec) and
most questions about an uploaded document won't contain any special
signal words at all.

Depends on: nothing else in src/ — kept standalone so it's trivial to test
and swap out independently of the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryRoute(str, Enum):
    PDF_ONLY = "PDF_ONLY"
    WEB_ONLY = "WEB_ONLY"
    PDF_AND_WEB = "PDF_AND_WEB"


@dataclass
class RoutingResult:
    route: QueryRoute
    matched_pdf_phrases: list[str]
    matched_web_phrases: list[str]


# Phrases suggesting the question is asking about the uploaded document
# itself. Not exhaustive — this is a coarse, deliberately simple signal,
# not an attempt to cover every possible phrasing.
PDF_SIGNAL_PHRASES = [
    "this paper", "this document", "this pdf", "this report", "this study",
    "the document", "the paper", "the study", "the report",
    "the author", "the authors", "according to",
    "the dataset used", "the dataset", "mentioned in", "stated in",
    "the methodology", "the findings", "the results section",
    "in the pdf", "in the document", "in this pdf",
]

# Phrases suggesting the question needs information beyond the document —
# something current, external, or comparative against the outside world.
WEB_SIGNAL_PHRASES = [
    "current", "currently", "latest", "recent", "recently", "today", "now",
    "nowadays", "this year", "up to date", "up-to-date", "state of the art",
    "compare with existing", "compare to existing", "compared to other",
    "compare it to", "how does this compare", "other papers", "other research",
    "in the industry", "real-world", "real world", "elsewhere",
    "outside this document", "outside the document", "versus other",
]


def _find_matches(text: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if re.search(rf"\b{re.escape(p)}\b", text)]


def classify_query(question: str) -> RoutingResult:
    """
    Decide how a question should be answered.

    Args:
        question: the user's question.

    Returns:
        A RoutingResult with the chosen route and the specific phrases that
        matched in each category (useful for debugging or for showing the
        user why a question triggered a web search).

    Raises:
        ValueError: empty question.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    text = question.lower()
    matched_pdf = _find_matches(text, PDF_SIGNAL_PHRASES)
    matched_web = _find_matches(text, WEB_SIGNAL_PHRASES)

    if matched_web and matched_pdf:
        route = QueryRoute.PDF_AND_WEB
    elif matched_web:
        route = QueryRoute.WEB_ONLY
    else:
        route = QueryRoute.PDF_ONLY  # default: the PDF is the primary knowledge source

    return RoutingResult(route=route, matched_pdf_phrases=matched_pdf, matched_web_phrases=matched_web)


if __name__ == "__main__":
    # Quick manual test: `python src/query_router.py "a question"`
    import sys

    if len(sys.argv) != 2:
        print('Usage: python query_router.py "<question>"')
        sys.exit(1)

    result = classify_query(sys.argv[1])
    print(f"Route: {result.route.value}")
    if result.matched_pdf_phrases:
        print(f"  matched PDF signals: {result.matched_pdf_phrases}")
    if result.matched_web_phrases:
        print(f"  matched web signals: {result.matched_web_phrases}")
