"""
chunker.py

Stage 3 of the DocuRAG pipeline: clean the raw page text produced by
pdf_loader.py, then split it into overlapping chunks ready for embedding.

There is no dedicated "cleaner" module in this project, so light text
cleaning lives here rather than in chunking's own file — cleaning only
exists to prepare text for chunking, so keeping them together avoids an
extra file for a handful of regex substitutions.

Why chunk instead of embedding whole pages: an embedding model produces one
fixed-size vector per input. A whole page usually covers several different
ideas, so one vector has to "average" all of them, which blurs similarity
search. Smaller, focused chunks give the retriever something more precise
to match a question against.

Why overlap between chunks: hard, non-overlapping boundaries can split a
fact right down the middle — half of a sentence ends up in one chunk, half
in the next, and neither chunk alone contains the whole fact. A small
overlap (10-20% here) lets neighboring chunks share some text so boundary
information isn't lost.

Depends on:
- src/pdf_loader.py (Phase 2) — for the PageContent type these functions
  consume.
- src/metadata_extractor.py (Phase 3) — optional. If you already extracted
  metadata for this document, its `boilerplate_lines` (repeated headers/
  footers detected there) can be passed in here so the exact same lines get
  stripped out before chunking, instead of duplicating that detection logic.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    from .pdf_loader import PageContent  # when imported as part of the src package
except ImportError:
    from pdf_loader import PageContent  # when run as a standalone script


# Baseline defaults per the project's chunking guidance: ~500-800 words per
# chunk with ~10-20% overlap is a reasonable starting point for real
# documents. These are not universally "optimal" values — see Section 15 of
# notebooks/document_rag_experiments.ipynb for a hands-on look at how
# changing them affects retrieval.
DEFAULT_CHUNK_SIZE = 600      # words per chunk
DEFAULT_CHUNK_OVERLAP_RATIO = 0.15  # 15% overlap between consecutive chunks


@dataclass
class Chunk:
    text: str
    page: int            # which page this chunk came from — required for source attribution
    chunk_id: str
    document_id: str
    word_count: int


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

# A conservative set of patterns for artifacts that show up across most
# PDFs regardless of document type. Document-specific repeated headers/
# footers are better handled via `boilerplate_lines` (see clean_text),
# since those vary per document and metadata_extractor.py already detects
# them from font/layout information that plain text alone doesn't have.
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)


def clean_text(text: str, boilerplate_lines: set[str] | None = None) -> str:
    """
    Light, conservative cleaning of one page's raw extracted text.

    Args:
        text: raw text from PageContent.text (pdf_loader.py).
        boilerplate_lines: exact repeated header/footer lines to strip,
            typically DocumentMetadata.boilerplate_lines from
            metadata_extractor.py. Matching is case-insensitive and ignores
            surrounding whitespace.

    We deliberately avoid aggressive cleaning (e.g. stripping all newlines,
    "fixing" hyphenation, or removing punctuation) — over-cleaning can
    destroy meaning just as easily as leaving noise in. Only patterns we
    can recognize with confidence are removed.
    """
    boilerplate_lower = {b.strip().lower() for b in (boilerplate_lines or set())}

    # Normalize unicode (e.g. combines "ﬁ" ligature forms into normal
    # letters) and drop stray control characters that occasionally appear
    # in poorly-encoded PDFs.
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())

    kept_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in boilerplate_lower:
            continue
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _chunk_page_text(text: str, page: int, chunk_size: int, overlap_ratio: float, document_id: str) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks: list[Chunk] = []
    start = 0
    chunk_num = 0
    while start < len(words):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(
            text=" ".join(window),
            page=page,
            chunk_id=f"{document_id}_p{page}_c{chunk_num}",
            document_id=document_id,
            word_count=len(window),
        ))
        chunk_num += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_pages(
    pages: list[PageContent],
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
    boilerplate_lines: set[str] | None = None,
) -> list[Chunk]:
    """
    Clean and chunk every page of a document.

    Chunking is done independently per page (rather than on one big joined
    string) so every chunk can carry an exact, unambiguous page number —
    the foundation of source attribution later in the pipeline. The
    trade-off is that a sentence split across a page boundary won't be
    fully captured in any single chunk; this is a known, accepted
    limitation (see notebooks/document_rag_experiments.ipynb, Section 22)
    rather than something this module tries to solve.

    Args:
        pages: output of pdf_loader.load_pdf_pages().
        document_id: identifier for this document, used as a prefix for
            every chunk_id (e.g. a slug derived from the filename, or a
            uuid — see make_document_id()).
        chunk_size: target words per chunk.
        overlap_ratio: fraction of each chunk repeated in the next
            (0.0-0.9).
        boilerplate_lines: repeated header/footer lines to strip before
            chunking, typically DocumentMetadata.boilerplate_lines from
            metadata_extractor.py.

    Raises:
        ValueError: if chunk_size or overlap_ratio are out of range.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not (0.0 <= overlap_ratio < 0.9):
        raise ValueError(f"overlap_ratio must be in [0.0, 0.9), got {overlap_ratio}")

    all_chunks: list[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text, boilerplate_lines)
        all_chunks.extend(_chunk_page_text(cleaned, page.page, chunk_size, overlap_ratio, document_id))
    return all_chunks


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_document_id(filename: str | None) -> str:
    """
    Derive a short, filesystem/ID-safe document identifier from a filename,
    e.g. "My Paper (v2).pdf" -> "my_paper_v2". Falls back to a random id if
    no usable filename is available (e.g. an in-memory upload with no name).
    """
    if not filename:
        import uuid
        return uuid.uuid4().hex[:12]

    import os
    basename = os.path.basename(filename)
    stem = re.sub(r"\.pdf$", "", basename, flags=re.IGNORECASE)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug or make_document_id(None)


def get_chunk_stats(chunks: list[Chunk]) -> dict:
    """Quick summary stats for inspecting a chunking result (e.g. in a notebook or logs)."""
    if not chunks:
        return {"count": 0}

    word_counts = [c.word_count for c in chunks]
    pages_covered = sorted({c.page for c in chunks})
    return {
        "count": len(chunks),
        "pages_covered": pages_covered,
        "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
    }


if __name__ == "__main__":
    # Quick manual test: `python src/chunker.py path/to/file.pdf`
    import sys

    try:
        from .pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from .metadata_extractor import extract_metadata
    except ImportError:
        from pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from metadata_extractor import extract_metadata

    if len(sys.argv) != 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    try:
        loaded_pages = load_pdf_pages(sys.argv[1])
    except (PDFLoadError, NoExtractableTextError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    # Reuse metadata_extractor's boilerplate detection so repeated headers/
    # footers get stripped before chunking, rather than duplicating that
    # detection logic here.
    meta = extract_metadata(sys.argv[1])
    boilerplate = set(meta.boilerplate_lines)

    doc_id = make_document_id(sys.argv[1])
    result_chunks = chunk_pages(loaded_pages, document_id=doc_id, boilerplate_lines=boilerplate)

    print(f"document_id: {doc_id}")
    print(f"boilerplate lines stripped: {boilerplate}")
    print(get_chunk_stats(result_chunks))
    print()
    for c in result_chunks[:3]:
        print(f"[{c.chunk_id}] page={c.page} words={c.word_count}")
        print(f"  {c.text[:100]}...")
