"""
summarizer.py

Stage 9 of the DocuRAG pipeline: produce the structured document summary
shown immediately after upload (title, objective, methodology, key
findings, limitations, etc.).

Two different sources feed the final summary:
- Title, authors, page count, and document type come from
  metadata_extractor.py (Phase 3) — these are heuristic-but-direct facts
  about the PDF's structure, and asking an LLM to re-guess them would just
  add a second, less reliable source for the same information.
- Objective, problem statement, methodology, key findings, limitations,
  conclusion, and keywords come from the LLM, since these require actually
  understanding and condensing the document's content — heuristics can't
  produce them.

Map-reduce strategy for long documents (Section 14 / 23 of the project
spec): sending an entire long PDF's text in one LLM call risks exceeding
context limits and tends to produce vaguer summaries that skim everything
shallowly. Instead:
  1. MAP: split the document's chunks into word-budgeted groups and
     summarize each group independently -> several intermediate summaries.
  2. REDUCE: combine those intermediate summaries into one final
     structured summary.

For documents short enough to comfortably fit in a single call, the map
step is skipped entirely and the reduce step runs directly on the full
document text — no reason to spend two LLM calls summarizing a two-page
PDF. Both paths end by calling the exact same structured-summary function,
so there's only one JSON prompt to maintain, not two.

Depends on:
- src/chunker.py (Phase 4) — for the Chunk type.
- src/metadata_extractor.py (Phase 3) — for DocumentMetadata (title,
  authors, num_pages, document_type).
- src/llm.py (Phase 8) — generate(), and its exception types.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

try:
    from .chunker import Chunk  # when imported as part of the src package
    from .metadata_extractor import DocumentMetadata
    from . import llm
except ImportError:
    from chunker import Chunk  # when run as a standalone script
    from metadata_extractor import DocumentMetadata
    import llm


NOT_IDENTIFIED = "Not clearly identified in the document."

# If the document's total word count is at or below this, skip the map
# step and summarize directly from the full text in one call. Above this,
# map first to condense. Not a universally "correct" cutoff — a rough
# balance between "small enough to fit comfortably in one call" and
# "avoid an unnecessary extra call for short documents". Once config.py
# exists these thresholds should move there.
DEFAULT_DIRECT_SUMMARY_MAX_WORDS = 3000

# Per-call word budget for each MAP-step intermediate summary.
DEFAULT_MAP_GROUP_MAX_WORDS = 2000


class SummarizationError(Exception):
    """Raised when the LLM's summary response can't be parsed into a usable structured summary."""


@dataclass
class DocumentSummary:
    # From metadata_extractor.py — not re-derived by the LLM.
    title: str
    authors: list[str]
    num_pages: int
    document_type: str

    # From the LLM.
    objective: str
    problem_statement: str
    detailed_summary: str
    methodology: str
    important_concepts: list[str]
    datasets: str
    key_findings: str
    limitations: str
    conclusion: str
    keywords: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

CHUNK_SUMMARY_SYSTEM_PROMPT = (
    "You are a precise, factual summarizer. Only use information explicitly "
    "present in the given excerpt. Do not add outside knowledge or guess at "
    "things the excerpt doesn't state."
)

CHUNK_SUMMARY_USER_TEMPLATE = """Summarize the following excerpt from a document in 2-4 sentences,
capturing any concrete facts, numbers, or claims it contains.

Excerpt (from {page_label}):
{text}
"""

FINAL_SUMMARY_SYSTEM_PROMPT = f"""You are a precise, factual document summarizer.
Base your summary ONLY on the text given to you. Do not use outside knowledge
and do not guess at anything the text doesn't support.

Respond with ONLY a single valid JSON object — no markdown code fences, no
commentary before or after it. The JSON object must have exactly these keys,
all as strings except where noted:

- "objective": the document's main goal or purpose
- "problem_statement": the problem or need the document addresses
- "detailed_summary": a thorough paragraph summarizing the document's content
- "methodology": the approach or method used, if applicable
- "important_concepts": an array of short strings naming key concepts
- "datasets": datasets or data sources mentioned, if any
- "key_findings": the main results or findings
- "limitations": limitations acknowledged in the document, if any
- "conclusion": the document's conclusion or closing takeaway
- "keywords": an array of short strings — important keywords/topics

If a field cannot be determined from the given text, use exactly this
string as its value: "{NOT_IDENTIFIED}" (or an empty array for the two
array fields). Never invent information to fill a field.
"""

FINAL_SUMMARY_USER_TEMPLATE = """DOCUMENT TEXT:
{text}

Produce the structured JSON summary described in your instructions.
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _page_label(chunks: list[Chunk]) -> str:
    pages = sorted({c.page for c in chunks})
    if len(pages) == 1:
        return f"page {pages[0]}"
    return f"pages {pages[0]}-{pages[-1]}"


def _chunks_to_text(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[Page {c.page}]\n{c.text}" for c in chunks)


def _group_chunks_by_word_budget(chunks: list[Chunk], max_words: int) -> list[list[Chunk]]:
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_words = 0
    for c in chunks:
        if current and current_words + c.word_count > max_words:
            groups.append(current)
            current, current_words = [], 0
        current.append(c)
        current_words += c.word_count
    if current:
        groups.append(current)
    return groups


def _parse_summary_json(raw_text: str) -> dict:
    # Some models wrap JSON in markdown code fences despite instructions
    # not to — strip that defensively before parsing, rather than failing
    # on an otherwise-perfectly-good response.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise SummarizationError(f"Could not parse the LLM's summary response as JSON: {e}") from e


def _as_str(value, default: str = NOT_IDENTIFIED) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _as_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        # Defensive fallback: the model returned a comma-separated string
        # instead of a JSON array.
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


# --------------------------------------------------------------------------
# Core summarization calls
# --------------------------------------------------------------------------

def _summarize_group(chunks: list[Chunk], api_key: str | None, gemini_model: str) -> str:
    """MAP step: one intermediate summary for one group of chunks."""
    prompt = CHUNK_SUMMARY_USER_TEMPLATE.format(page_label=_page_label(chunks), text=_chunks_to_text(chunks))
    summary = llm.generate(CHUNK_SUMMARY_SYSTEM_PROMPT, prompt, api_key=api_key, model=gemini_model)
    return f"({_page_label(chunks)}) {summary}"


def _generate_structured_fields(text: str, api_key: str | None, gemini_model: str) -> dict:
    """REDUCE step (or the direct single-call path): text -> structured JSON fields."""
    prompt = FINAL_SUMMARY_USER_TEMPLATE.format(text=text)
    raw = llm.generate(FINAL_SUMMARY_SYSTEM_PROMPT, prompt, api_key=api_key, model=gemini_model)
    return _parse_summary_json(raw)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def generate_summary(
    chunks: list[Chunk],
    metadata: DocumentMetadata,
    api_key: str | None = None,
    gemini_model: str = llm.DEFAULT_GEMINI_MODEL,
    direct_summary_max_words: int = DEFAULT_DIRECT_SUMMARY_MAX_WORDS,
    map_group_max_words: int = DEFAULT_MAP_GROUP_MAX_WORDS,
) -> DocumentSummary:
    """
    Produce the full structured DocumentSummary for a processed PDF.

    Args:
        chunks: the document's chunks, e.g. from chunker.chunk_pages().
        metadata: the document's metadata, from metadata_extractor.extract_metadata().
        api_key / gemini_model: passed through to llm.generate().
        direct_summary_max_words / map_group_max_words: tune the map-reduce
            behavior — see the module docstring.

    Raises:
        ValueError: no chunks given (nothing to summarize).
        llm.MissingAPIKeyError / llm.LLMGenerationError: an LLM call failed.
        SummarizationError: the LLM's response couldn't be parsed as the
            expected structured JSON.
    """
    if not chunks:
        raise ValueError("Cannot summarize a document with no chunks.")

    total_words = sum(c.word_count for c in chunks)

    if total_words <= direct_summary_max_words:
        fields = _generate_structured_fields(_chunks_to_text(chunks), api_key, gemini_model)
    else:
        groups = _group_chunks_by_word_budget(chunks, map_group_max_words)
        intermediate_summaries = [_summarize_group(g, api_key, gemini_model) for g in groups]
        fields = _generate_structured_fields("\n".join(intermediate_summaries), api_key, gemini_model)

    return DocumentSummary(
        title=metadata.title or NOT_IDENTIFIED,
        authors=metadata.authors,
        num_pages=metadata.num_pages,
        document_type=metadata.document_type,
        objective=_as_str(fields.get("objective")),
        problem_statement=_as_str(fields.get("problem_statement")),
        detailed_summary=_as_str(fields.get("detailed_summary")),
        methodology=_as_str(fields.get("methodology")),
        important_concepts=_as_str_list(fields.get("important_concepts")),
        datasets=_as_str(fields.get("datasets")),
        key_findings=_as_str(fields.get("key_findings")),
        limitations=_as_str(fields.get("limitations")),
        conclusion=_as_str(fields.get("conclusion")),
        keywords=_as_str_list(fields.get("keywords")),
    )


if __name__ == "__main__":
    # Quick manual test: `python src/summarizer.py path/to/file.pdf`
    import sys

    try:
        from .pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from .chunker import chunk_pages, make_document_id
        from .metadata_extractor import extract_metadata
    except ImportError:
        from pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from chunker import chunk_pages, make_document_id
        from metadata_extractor import extract_metadata

    if len(sys.argv) != 2:
        print("Usage: python summarizer.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        pages = load_pdf_pages(pdf_path)
    except (PDFLoadError, NoExtractableTextError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    meta = extract_metadata(pdf_path)
    doc_id = make_document_id(pdf_path)
    chunks = chunk_pages(pages, document_id=doc_id, boilerplate_lines=set(meta.boilerplate_lines))

    if not llm.is_configured():
        print("No GEMINI_API_KEY found — cannot generate a real summary. Add one to a .env file and try again.")
        sys.exit(1)

    try:
        summary = generate_summary(chunks, meta)
    except (llm.MissingAPIKeyError, llm.LLMGenerationError, SummarizationError) as e:
        print(f"Summarization failed: {e}")
        sys.exit(1)

    for field_name, value in summary.__dict__.items():
        print(f"{field_name}: {value}")
