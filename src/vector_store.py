"""
vector_store.py

Stage 5 of the DocuRAG pipeline: build a local FAISS index over chunk
embeddings, and maintain the FAISS vector ID -> chunk metadata mapping
required for source attribution. Also performs the similarity search
itself: given a query vector, return the top-k most similar chunks.

Why IndexFlatIP + L2-normalization, instead of the more common IndexFlatL2:
Cosine similarity measures the *angle* between two vectors rather than raw
Euclidean distance, which suits sentence embeddings well — the direction of
the vector carries the semantic meaning more reliably than its magnitude.
If every vector is first L2-normalized to unit length, the inner product
between two vectors becomes mathematically identical to their cosine
similarity. So IndexFlatIP (Inner Product) over normalized vectors gives us
cosine similarity search without needing a dedicated "cosine index" type.

Why "Flat" (exact search) rather than an approximate index (IVF, HNSW,
etc.): Flat brute-force-compares the query against every stored vector,
which is exact but O(n). For a single document's worth of chunks (hundreds,
not millions), this is fast enough that trading accuracy for speed with an
approximate index isn't worth the added complexity — a case of not
over-engineering for a scale this project doesn't operate at.

FAISS itself is not a trained model — it's a data structure for fast vector
search. No training happens here.

Depends on:
- src/chunker.py (Phase 4) — for the Chunk type, stored as this index's
  metadata.
- src/embeddings.py (Phase 5) — not called directly, but this module
  consumes its output: an embeddings array whose rows are aligned with a
  list of Chunk objects (exactly what embed_chunks() returns).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

try:
    from .chunker import Chunk  # when imported as part of the src package
except ImportError:
    from chunker import Chunk  # when run as a standalone script


# Where a VectorStore is saved/loaded from by default, if no directory is
# given explicitly. Matches the project's top-level vectorstore/ folder.
# Once config.py exists, this should move there alongside the other
# configurable paths/constants rather than staying a local default.
DEFAULT_VECTORSTORE_DIR = Path("vectorstore")


class VectorStoreError(Exception):
    """Raised for invalid vector store construction, save, or load operations."""


@dataclass
class SearchResult:
    chunk: Chunk
    similarity_score: float  # cosine similarity, in [-1.0, 1.0] (typically 0-1 for real text)


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize a copy of the given vectors so inner product == cosine similarity."""
    normalized = np.ascontiguousarray(embeddings, dtype="float32").copy()
    faiss.normalize_L2(normalized)
    return normalized


class VectorStore:
    """
    A local FAISS index paired with the chunk metadata needed to make its
    search results meaningful. FAISS vector ID `i` always corresponds to
    `self._chunks[i]` — this is the "FAISS vector ID -> chunk ID -> page
    number -> text" mapping the project's architecture requires. Without
    this pairing, a search would only return "vector #7 matched" with no
    way to recover what that vector represents.
    """

    def __init__(self, chunks: list[Chunk], embedding_dim: int, document_id: str | None = None):
        self._chunks = chunks
        self._embedding_dim = embedding_dim
        self.document_id = document_id or (chunks[0].document_id if chunks else None)
        self._index = faiss.IndexFlatIP(embedding_dim)

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: np.ndarray) -> "VectorStore":
        """
        Build a new VectorStore from chunks and their embeddings.

        Args:
            chunks: e.g. the output of chunker.chunk_pages().
            embeddings: a (len(chunks), embedding_dim) float array, e.g. the
                output of embeddings.embed_chunks() — row i must correspond
                to chunks[i].

        Raises:
            VectorStoreError: no chunks given, embeddings aren't 2D, or the
                number of embeddings doesn't match the number of chunks.
        """
        if len(chunks) == 0:
            raise VectorStoreError(
                "Cannot build a vector store with zero chunks. This usually "
                "means the document produced no usable text after chunking."
            )
        if embeddings.ndim != 2:
            raise VectorStoreError(f"Expected a 2D embeddings array, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Number of embeddings ({embeddings.shape[0]}) does not match "
                f"number of chunks ({len(chunks)})."
            )

        store = cls(chunks, embedding_dim=embeddings.shape[1])
        store._index.add(_normalize(embeddings))
        return store

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        """
        Find the top_k chunks most similar to a query embedding.

        Args:
            query_embedding: shape (1, embedding_dim), e.g. from
                embeddings.embed_query(). Must come from the SAME embedding
                model used to build this index — comparing vectors from two
                different models produces meaningless scores.
            top_k: how many results to return. Not assumed to be universally
                "correct" at any particular value — see the notebook's
                Section 15 for how this trades recall against precision.

        Returns:
            SearchResults ordered from most to least similar. Fewer than
            top_k results are returned if the index has fewer vectors than
            top_k.
        """
        if len(self) == 0:
            return []
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}")

        query = _normalize(query_embedding)
        k = min(top_k, len(self))
        scores, ids = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], similarity_score=float(score)))
        return results

    def __len__(self) -> int:
        return self._index.ntotal

    # ----------------------------------------------------------------
    # Persistence. Optional — Version 1 typically keeps a single
    # document's store in memory for the current session (see Streamlit
    # session_state usage in app.py, Phase 11). Saving/loading is useful
    # for the notebook, for tests, or for skipping re-embedding a document
    # you've already processed before. Per the project's .gitignore rules,
    # saved vector stores should not be committed to source control.
    # ----------------------------------------------------------------

    def save(self, directory: str | Path | None = None) -> Path:
        """Save this index and its chunk metadata to `directory` (default: vectorstore/<document_id>/)."""
        target_dir = Path(directory) if directory else DEFAULT_VECTORSTORE_DIR / (self.document_id or "unnamed_document")
        target_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(target_dir / "index.faiss"))

        metadata = {
            "document_id": self.document_id,
            "embedding_dim": self._embedding_dim,
            "chunks": [asdict(c) for c in self._chunks],
        }
        with open(target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return target_dir

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a VectorStore previously saved with .save()."""
        target_dir = Path(directory)
        index_path = target_dir / "index.faiss"
        metadata_path = target_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(f"No saved vector store found at: {target_dir}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        chunks = [Chunk(**c) for c in metadata["chunks"]]
        store = cls(chunks, embedding_dim=metadata["embedding_dim"], document_id=metadata["document_id"])
        store._index = faiss.read_index(str(index_path))
        return store


if __name__ == "__main__":
    # Quick manual test: `python src/vector_store.py path/to/file.pdf "a question"`
    import sys

    try:
        from .pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from .chunker import chunk_pages, make_document_id
        from .metadata_extractor import extract_metadata
        from .embeddings import load_embedding_model, embed_chunks, embed_query, EmbeddingModelError
    except ImportError:
        from pdf_loader import load_pdf_pages, PDFLoadError, NoExtractableTextError
        from chunker import chunk_pages, make_document_id
        from metadata_extractor import extract_metadata
        from embeddings import load_embedding_model, embed_chunks, embed_query, EmbeddingModelError

    if len(sys.argv) != 3:
        print('Usage: python vector_store.py <path_to_pdf> "<question>"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]

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
    print(f"Built vector store with {len(store)} vectors (document_id={store.document_id})")

    query_vec = embed_query(question, model=model)
    results = store.search(query_vec, top_k=3)

    print(f"\nQuestion: {question}\n")
    for r in results:
        snippet = r.chunk.text[:100].replace("\n", " ")
        print(f"  page={r.chunk.page}  score={r.similarity_score:.4f}  \"{snippet}...\"")
