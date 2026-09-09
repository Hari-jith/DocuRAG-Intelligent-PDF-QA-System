"""
metadata_extractor.py

Stage 2 of the DocuRAG pipeline: pull lightweight, structural metadata out
of a PDF — title, authors, page count, a best-guess document type, and
section headings — using PyMuPDF's font/layout information plus a handful
of text heuristics.

Important distinction from summarizer.py (Phase 10): this module does NOT
call an LLM and does NOT produce the long structured summary (objective,
methodology, key findings, etc.). It only extracts things that can be
determined from the PDF's own text and layout. This keeps metadata
available immediately after upload, before any LLM call happens.

Depends on: src/pdf_loader.py (already created in Phase 2) — reused here
only for the PDFSource type alias and the PDFLoadError exception, so both
modules fail the same way on a missing/corrupted file.

Design principle (matches the "do not invent metadata" project rule): every
guess this module makes is heuristic, not certain. Rather than silently
returning a possibly-wrong value, each guessed field is paired with a
"*_source" field explaining how it was derived, so the UI layer (app.py,
built later) can show the user how confident to be — e.g. an explicit
"Authors:" label found in the text is trustworthy; "first line on the
page" is a fallback guess. If nothing could be found at all, the field is
left empty/None rather than fabricated, and the UI is expected to display
"Not clearly identified in the document." in that case.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

try:
    from .pdf_loader import PDFSource, PDFLoadError  # when imported as part of the src package
except ImportError:
    from pdf_loader import PDFSource, PDFLoadError  # when run as a standalone script


@dataclass
class Heading:
    """A detected section heading."""
    text: str
    page: int


@dataclass
class DocumentMetadata:
    num_pages: int

    title: str | None
    title_source: str  # "font-size heuristic" | "first-line fallback" | "not found"

    authors: list[str]
    authors_source: str  # "explicit label" | "heuristic (line after title)" | "not found"

    document_type: str  # a guessed label, or "Not clearly identified in the document."
    document_type_signals: list[str]  # keywords that triggered the guess, for transparency

    headings: list[Heading]
    boilerplate_lines: list[str]  # repeated headers/footers detected and excluded


@dataclass
class _Line:
    """Internal helper: one line of text on a page, with its largest font size."""
    text: str
    page: int
    max_font_size: float
    y0: float  # top y-coordinate, used to keep reading order


# --------------------------------------------------------------------------
# PDF opening (mirrors pdf_loader._open_pdf so this module can be used on
# its own, without requiring pdf_loader.load_pdf_pages to run first)
# --------------------------------------------------------------------------

def _open_pdf(source: PDFSource) -> pymupdf.Document:
    try:
        if isinstance(source, bytes):
            if len(source) == 0:
                raise PDFLoadError("Received an empty file (0 bytes).")
            return pymupdf.open(stream=source, filetype="pdf")
        path = str(source)
        return pymupdf.open(path)
    except PDFLoadError:
        raise
    except Exception as e:
        raise PDFLoadError(f"Could not open PDF — it may be corrupted or invalid: {e}") from e


# --------------------------------------------------------------------------
# Layout extraction: turn PyMuPDF's raw "dict" output into a flat list of
# lines, each carrying the largest font size used within that line. Font
# size is the main signal we use to guess what's a title vs. a heading vs.
# ordinary body text.
# --------------------------------------------------------------------------

def _extract_lines(doc: pymupdf.Document) -> list[_Line]:
    lines: list[_Line] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                max_size = max((s.get("size", 0.0) for s in spans), default=0.0)
                y0 = line.get("bbox", [0, 0, 0, 0])[1]
                lines.append(_Line(text=text, page=page_index + 1, max_font_size=max_size, y0=y0))
    return lines


# --------------------------------------------------------------------------
# Boilerplate (repeated header/footer) detection. A line that appears,
# near-identically, on at least half of the document's pages is almost
# certainly a running header or footer rather than real content — we
# exclude these before guessing the title, authors, or headings so a
# repeated header like "Company Confidential" doesn't get mistaken for
# either.
# --------------------------------------------------------------------------

def _detect_boilerplate(lines: list[_Line], num_pages: int) -> set[str]:
    if num_pages <= 1:
        return set()

    normalized_to_pages: dict[str, set[int]] = {}
    for ln in lines:
        key = re.sub(r"\d+", "#", ln.text.strip().lower())  # ignore page numbers like "Page 3 of 12"
        normalized_to_pages.setdefault(key, set()).add(ln.page)

    threshold = max(2, (num_pages // 2) + 1)
    return {key for key, pages in normalized_to_pages.items() if len(pages) >= threshold}


def _is_boilerplate(text: str, boilerplate_keys: set[str]) -> bool:
    key = re.sub(r"\d+", "#", text.strip().lower())
    return key in boilerplate_keys


# --------------------------------------------------------------------------
# Title detection
# --------------------------------------------------------------------------

def _guess_title(page1_lines: list[_Line], body_font_size: float) -> tuple[str | None, str]:
    if not page1_lines:
        return None, "not found"

    largest_size = max(ln.max_font_size for ln in page1_lines)

    # A real title is usually noticeably larger than ordinary body text. If
    # the largest text on page 1 isn't meaningfully bigger than body text,
    # font size alone isn't a reliable signal (common for plain-text PDFs
    # with little formatting), so we fall back below.
    if largest_size >= body_font_size * 1.05:
        title_lines = [ln.text for ln in page1_lines if ln.max_font_size >= largest_size * 0.95]
        title = " ".join(title_lines).strip()
        if title:
            return title, "font-size heuristic"

    # Fallback: best-effort guess using the first line of real content.
    # This is still literal text taken from the document, not a fabricated
    # value — but it's a weaker signal, hence the different *_source label.
    fallback = page1_lines[0].text.strip()
    return (fallback or None), ("first-line fallback" if fallback else "not found")


# --------------------------------------------------------------------------
# Author detection
# --------------------------------------------------------------------------

_AUTHOR_LABEL_RE = re.compile(r"^\s*(?:authors?|by)\s*:\s*(.+)$", re.IGNORECASE)
_AFFILIATION_HINTS = ("university", "department", "institute", "college", "@", "school of")


def _split_author_list(raw: str) -> list[str]:
    parts = re.split(r",| and | & ", raw)
    return [p.strip() for p in parts if p.strip()]


def _guess_authors(page1_lines: list[_Line], title: str | None) -> tuple[list[str], str]:
    # Strategy 1: an explicit "Authors:" or "By:" label anywhere on page 1.
    for ln in page1_lines:
        match = _AUTHOR_LABEL_RE.match(ln.text)
        if match:
            authors = _split_author_list(match.group(1))
            if authors:
                return authors, "explicit label"

    # Strategy 2: the line(s) immediately following the title, if they look
    # like a name list rather than an affiliation/address line.
    if title:
        try:
            title_index = next(i for i, ln in enumerate(page1_lines) if ln.text in title)
        except StopIteration:
            title_index = -1

        for ln in page1_lines[title_index + 1: title_index + 3]:
            lower = ln.text.lower()
            looks_like_affiliation = any(hint in lower for hint in _AFFILIATION_HINTS)
            looks_like_names = ("," in ln.text or " and " in lower) and not looks_like_affiliation
            if looks_like_names:
                return _split_author_list(ln.text), "heuristic (line after title)"

    return [], "not found"


# --------------------------------------------------------------------------
# Document type detection
# --------------------------------------------------------------------------

_DOCUMENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Research paper": ["abstract", "references", "et al.", "keywords:", "doi:", "methodology"],
    "Resume / CV": ["curriculum vitae", "work experience", "professional experience", "skills", "objective:"],
    "Invoice": ["invoice", "total due", "amount due", "invoice number", "bill to"],
    "Report": ["executive summary", "table of contents", "recommendations"],
}
_MIN_SIGNALS_REQUIRED = 2


def _guess_document_type(full_text_lower: str) -> tuple[str, list[str]]:
    best_type = "Not clearly identified in the document."
    best_signals: list[str] = []
    for doc_type, keywords in _DOCUMENT_TYPE_KEYWORDS.items():
        found = [kw for kw in keywords if kw in full_text_lower]
        if len(found) >= _MIN_SIGNALS_REQUIRED and len(found) > len(best_signals):
            best_type, best_signals = doc_type, found
    return best_type, best_signals


# --------------------------------------------------------------------------
# Heading detection
# --------------------------------------------------------------------------

_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.)\s+\S")


def _guess_headings(content_lines: list[_Line], body_font_size: float, title: str | None) -> list[Heading]:
    size_threshold = body_font_size * 1.15
    headings: list[Heading] = []
    for ln in content_lines:
        text = ln.text.strip()
        if len(text) == 0 or len(text) > 80:
            continue
        if title and text in title:
            continue  # the title itself shouldn't also be listed as a heading
        looks_bigger = ln.max_font_size >= size_threshold
        looks_numbered = bool(_NUMBERED_HEADING_RE.match(text))
        if looks_bigger or looks_numbered:
            headings.append(Heading(text=text, page=ln.page))
    return headings


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def extract_metadata(source: PDFSource) -> DocumentMetadata:
    """
    Extract structural metadata from a PDF: page count, a best-guess title
    and author list, a best-guess document type, and detected section
    headings. Every guessed field is paired with a "*_source" explanation
    (see DocumentMetadata) so callers never have to treat a heuristic guess
    as a certain fact.

    Args:
        source: path to a PDF file (str/Path), or raw PDF bytes.

    Raises:
        PDFLoadError: file missing, not a PDF, or corrupted/unreadable.
    """
    doc = _open_pdf(source)
    try:
        num_pages = doc.page_count
        if num_pages == 0:
            raise PDFLoadError("The PDF has 0 pages.")

        all_lines = _extract_lines(doc)
    finally:
        doc.close()

    boilerplate_keys = _detect_boilerplate(all_lines, num_pages)
    boilerplate_examples = sorted({ln.text for ln in all_lines if _is_boilerplate(ln.text, boilerplate_keys)})

    content_lines = [ln for ln in all_lines if not _is_boilerplate(ln.text, boilerplate_keys)]
    content_lines.sort(key=lambda ln: (ln.page, ln.y0))  # preserve top-to-bottom reading order

    if not content_lines:
        # Every line looked like boilerplate (extremely unusual) — fall back
        # to the unfiltered lines so we still return something sensible.
        content_lines = sorted(all_lines, key=lambda ln: (ln.page, ln.y0))

    font_sizes = [ln.max_font_size for ln in content_lines if ln.max_font_size > 0]
    body_font_size = _median(font_sizes) if font_sizes else 0.0

    page1_lines = [ln for ln in content_lines if ln.page == 1]
    title, title_source = _guess_title(page1_lines, body_font_size)
    authors, authors_source = _guess_authors(page1_lines, title)

    full_text_lower = " ".join(ln.text for ln in content_lines).lower()
    document_type, type_signals = _guess_document_type(full_text_lower)

    headings = _guess_headings(content_lines, body_font_size, title)

    return DocumentMetadata(
        num_pages=num_pages,
        title=title,
        title_source=title_source,
        authors=authors,
        authors_source=authors_source,
        document_type=document_type,
        document_type_signals=type_signals,
        headings=headings,
        boilerplate_lines=boilerplate_examples,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[mid]


if __name__ == "__main__":
    # Quick manual test: `python src/metadata_extractor.py path/to/file.pdf`
    import sys

    if len(sys.argv) != 2:
        print("Usage: python metadata_extractor.py <path_to_pdf>")
        sys.exit(1)

    try:
        meta = extract_metadata(sys.argv[1])
    except PDFLoadError as e:
        print(f"Failed to extract metadata: {e}")
        sys.exit(1)

    print(f"Pages: {meta.num_pages}")
    print(f"Title ({meta.title_source}): {meta.title or 'Not clearly identified in the document.'}")
    print(f"Authors ({meta.authors_source}): {meta.authors or 'Not clearly identified in the document.'}")
    print(f"Document type: {meta.document_type}  (signals: {meta.document_type_signals})")
    print(f"Detected {len(meta.headings)} heading(s):")
    for h in meta.headings:
        print(f"  [Page {h.page}] {h.text}")
    if meta.boilerplate_lines:
        print(f"Excluded repeated header/footer lines: {meta.boilerplate_lines}")
