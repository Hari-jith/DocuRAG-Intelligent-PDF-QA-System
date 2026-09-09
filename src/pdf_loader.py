"""
pdf_loader.py

Stage 1 of the DocuRAG pipeline: open a PDF and extract raw text page by
page using PyMuPDF, preserving page numbers as metadata.

This module intentionally does ONLY extraction. It does not clean text,
chunk text, or pull out structured metadata like title/authors — those are
separate concerns handled by later modules (chunker.py, metadata_extractor.py)
so each file in src/ stays focused on one job.

Why page-by-page matters: every downstream step (chunking, retrieval,
answer generation) needs to know which page a piece of text came from, so
the application can eventually show the user "Sources: PDF - Page 4"
instead of an answer with no way to verify it against the document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pymupdf  # PyMuPDF. Note: the older `import fitz` alias is deprecated.


class PDFLoadError(Exception):
    """Raised when a PDF file cannot be found, opened, or read at all."""


class NoExtractableTextError(Exception):
    """
    Raised when a PDF opens successfully but contains no extractable text on
    any page. This almost always means the PDF is scanned / image-based and
    would require OCR — which is explicitly out of scope for this version.
    We raise rather than silently returning empty results, so the caller
    (eventually the Streamlit app) can show the user a clear explanation
    instead of a confusing blank summary.
    """


@dataclass
class PageContent:
    """Raw text extracted from a single PDF page, with its page number."""

    page: int        # 1-indexed page number — required for source attribution
    text: str         # raw extracted text, not yet cleaned
    char_count: int   # length of the stripped text; used to flag near-empty pages


# A PDF can come from a file path (typical during development/notebooks) or
# from raw bytes (typical for a Streamlit file uploader, which hands back an
# in-memory byte stream rather than a path on disk). Supporting both here
# means the Streamlit integration in Phase 11 won't need to save the upload
# to disk first just to read it.
PDFSource = Union[str, Path, bytes]


def _open_pdf(source: PDFSource) -> pymupdf.Document:
    """Open a PDF from a file path or raw bytes, raising PDFLoadError on failure."""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise PDFLoadError(f"PDF file not found: {path}")
            if path.suffix.lower() != ".pdf":
                raise PDFLoadError(f"Expected a .pdf file, got: {path.suffix or '(no extension)'}")
            return pymupdf.open(str(path))

        if isinstance(source, bytes):
            if len(source) == 0:
                raise PDFLoadError("Received an empty file (0 bytes).")
            return pymupdf.open(stream=source, filetype="pdf")

        raise PDFLoadError(
            f"Unsupported PDF source type: {type(source).__name__}. "
            "Expected a file path (str/Path) or raw bytes."
        )
    except PDFLoadError:
        raise
    except Exception as e:
        # PyMuPDF raises its own errors (e.g. pymupdf.FileDataError) for
        # corrupted or non-PDF files. We wrap them in our own exception type
        # so callers only need to catch PDFLoadError, not a PyMuPDF-specific one.
        raise PDFLoadError(f"Could not open PDF — it may be corrupted or invalid: {e}") from e


def load_pdf_pages(source: PDFSource) -> list[PageContent]:
    """
    Extract raw text from every page of a PDF, preserving page numbers.

    Args:
        source: path to a PDF file (str/Path), or raw PDF bytes.

    Returns:
        A list of PageContent, one entry per page, in page order (page 1 first).

    Raises:
        PDFLoadError: file missing, not a PDF, or corrupted/unreadable.
        NoExtractableTextError: the PDF opened fine but no page has any
            extractable text (likely a scanned document — needs OCR).
    """
    doc = _open_pdf(source)
    try:
        if doc.page_count == 0:
            raise PDFLoadError("The PDF has 0 pages.")

        pages: list[PageContent] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            pages.append(PageContent(page=i + 1, text=text, char_count=len(text.strip())))
    finally:
        doc.close()

    total_extractable_chars = sum(p.char_count for p in pages)
    if total_extractable_chars == 0:
        raise NoExtractableTextError(
            "No extractable text was found in any page of this PDF. "
            "This usually means the document is scanned/image-based and "
            "would require OCR, which is not supported in this version."
        )

    return pages


def get_empty_pages(pages: list[PageContent], min_chars: int = 20) -> list[int]:
    """
    Return page numbers with little/no extractable text (blank pages, or
    pages that are mostly a full-page image). The document as a whole can
    still be usable even if a handful of pages like this exist — this is
    just useful information to surface to the user, not a reason to fail.
    """
    return [p.page for p in pages if p.char_count < min_chars]


def get_full_text(pages: list[PageContent]) -> str:
    """
    Join all page texts into one string with page markers, e.g. for quick
    manual inspection or debugging. The actual RAG pipeline chunks page by
    page (see chunker.py) rather than working off this joined string.
    """
    return "\n\n".join(f"[Page {p.page}]\n{p.text}" for p in pages)


if __name__ == "__main__":
    # Quick manual test: `python src/pdf_loader.py path/to/file.pdf`
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    try:
        loaded_pages = load_pdf_pages(sys.argv[1])
    except (PDFLoadError, NoExtractableTextError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    print(f"Loaded {len(loaded_pages)} pages")
    for p in loaded_pages:
        print(f"  Page {p.page}: {p.char_count} characters")

    empty = get_empty_pages(loaded_pages)
    if empty:
        print(f"Pages with little/no text: {empty}")
