"""
embeddings.py

Stage 4 of the DocuRAG pipeline: convert chunk text (and later, user
questions) into dense numerical vectors using a local Sentence Transformers
model. This module only produces embeddings — it does not build or query
the FAISS index (that's vector_store.py, Phase 6) and does not decide how
vectors are compared for similarity (also vector_store.py's concern, since
that depends on the specific FAISS index type chosen).

Model: sentence-transformers/all-MiniLM-L6-v2
- Small (~80MB) and fast enough to run on CPU, which matters for a
  free/locally-runnable project — no GPU or paid embedding API required.
- Produces a 384-dimensional vector for any input text.
- Trained so that semantically similar sentences end up close together in
  vector space, even when they share few or no exact words — this is what
  makes semantic search possible in the first place, as opposed to plain
  keyword matching.

No training happens anywhere in this module — the model is used purely for
inference (turning text into vectors), exactly as pretrained.

Note on configuration: DEFAULT_EMBEDDING_MODEL is defined locally here for
now. Once config.py exists it will become the single source of truth for
this value (and others like CHUNK_SIZE, TOP_K); at that point the default
here would be replaced with an import from config.py rather than kept as a
second copy of the same constant.
"""

from __future__ import annotations

import functools

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .chunker import Chunk  # when imported as part of the src package
except ImportError:
    from chunker import Chunk  # when run as a standalone script


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load (e.g. no internet on
    first run, an invalid model name, or a corrupted local cache)."""


@functools.lru_cache(maxsize=4)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load a Sentence Transformers model, caching it in memory per process so
    repeated calls with the same model_name don't reload from disk.

    In the Streamlit app (Phase 11) this will typically be wrapped a second
    time with @st.cache_resource, which caches across Streamlit reruns —
    the two caches serve different purposes and are both worth keeping:
    lru_cache here helps any caller (scripts, notebooks, tests), while
    st.cache_resource is specifically about surviving Streamlit's rerun
    model.

    The first call downloads the model from Hugging Face and caches it to
    disk (usually under ~/.cache/huggingface); every call after that, in
    any process, loads instantly from that local cache — no re-download,
    no training involved.

    Raises:
        EmbeddingModelError: if the model can't be loaded — most commonly
            no internet access on a first-ever run, or an invalid model
            name.
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise EmbeddingModelError(
            f"Failed to load embedding model '{model_name}'. If this is the "
            f"first time running DocuRAG, this usually means the model "
            f"couldn't be downloaded (check your internet connection). "
            f"Original error: {e}"
        ) from e


def get_embedding_dimension(model: SentenceTransformer) -> int:
    """The fixed size of every vector this model produces (384 for MiniLM-L6-v2)."""
    return model.get_sentence_embedding_dimension()


def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of texts (e.g. chunk texts) into embedding vectors.

    Args:
        texts: the strings to embed. An empty list returns an empty array
            with the correct embedding dimension, rather than erroring.
        model: an already-loaded model (from load_embedding_model()). If
            omitted, one is loaded using model_name.
        model_name: used only when `model` isn't provided.
        batch_size: how many texts to encode per batch — larger batches
            are faster but use more memory.
        show_progress: show a progress bar (useful for large documents,
            noisy for a Streamlit app or short scripts).

    Returns:
        A float32 numpy array of shape (len(texts), embedding_dim).
    """
    model = model or load_embedding_model(model_name)

    if not texts:
        return np.empty((0, get_embedding_dimension(model)), dtype="float32")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_query(
    query: str,
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Encode a single user question into an embedding vector, shaped (1, dim)
    so it can be passed directly to a FAISS index's .search().

    Using the exact same model for both chunks and queries is essential —
    the two vectors are only comparable because they were produced by the
    same model into the same vector space. Mixing embedding models between
    indexing and querying would silently produce meaningless similarity
    scores.
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")
    return embed_texts([query], model=model, model_name=model_name)


def embed_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = False,
) -> tuple[list[Chunk], np.ndarray]:
    """
    Convenience wrapper around embed_texts() for chunker.Chunk objects.
    Returns the same chunk list back alongside its embeddings, purely so
    callers have both in one place — row i of the returned array always
    corresponds to chunks[i]. This pairing is exactly what vector_store.py
    needs to build the FAISS-ID -> chunk metadata mapping.
    """
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model=model, model_name=model_name, batch_size=batch_size, show_progress=show_progress)
    return chunks, embeddings


if __name__ == "__main__":
    # Quick manual test: `python src/embeddings.py path/to/file.pdf`
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
        print("Usage: python embeddings.py <path_to_pdf>")
        sys.exit(1)

    try:
        pages = load_pdf_pages(sys.argv[1])
    except (PDFLoadError, NoExtractableTextError) as e:
        print(f"Failed to load PDF: {e}")
        sys.exit(1)

    meta = extract_metadata(sys.argv[1])
    doc_id = make_document_id(sys.argv[1])
    chunks = chunk_pages(pages, document_id=doc_id, boilerplate_lines=set(meta.boilerplate_lines))

    try:
        model = load_embedding_model()
    except EmbeddingModelError as e:
        print(e)
        sys.exit(1)

    print(f"Loaded model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Embedding dimension: {get_embedding_dimension(model)}")

    chunks, chunk_embeddings = embed_chunks(chunks, model=model, show_progress=True)
    print(f"Embedded {len(chunks)} chunks -> shape {chunk_embeddings.shape}")

    q_vec = embed_query("What is this document about?", model=model)
    print(f"Query embedding shape: {q_vec.shape}")
