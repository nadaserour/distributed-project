# rag/retriever.py
# Loads the pre-built FAISS index from disk and answers semantic queries.
# Build the index first by running:
#     python -m rag.build_index

import pickle
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & settings
# ---------------------------------------------------------------------------
INDEX_FOLDER = Path(__file__).parent / "index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 2

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[dict] | None = None

# ---------------------------------------------------------------------------
# Load index + embedding model
# ---------------------------------------------------------------------------
def load_index() -> None:
    """
    Load FAISS index and embedding model into memory once.
    Shared by all requests inside the worker process.
    """
    global _model, _index, _chunks

    # Already loaded
    if _index is not None and _model is not None and _chunks is not None:
        return

    index_path = INDEX_FOLDER / "faiss.index"
    chunks_path = INDEX_FOLDER / "chunks.pkl"

    # Validate files exist
    if not index_path.exists():
        raise FileNotFoundError(
            f"[RAG] Missing FAISS index file: {index_path}\n"
            "Run: python -m rag.build_index"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"[RAG] Missing chunks metadata file: {chunks_path}\n"
            "Run: python -m rag.build_index"
        )

    logger.info("[RAG] Loading FAISS index...")
    _index = faiss.read_index(str(index_path))

    logger.info("[RAG] Loading chunk metadata...")
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    logger.info(
        f"[RAG] Loaded {_index.ntotal} vectors from "
        f"{len(_unique_sources())} documents."
    )

    logger.info(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")

    # This downloads only first time, then cached
    _model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("[RAG] Retriever ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_sources() -> list[str]:
    """Return unique source filenames."""
    if _chunks is None:
        return []

    return sorted({
        c.get("source", "unknown")
        for c in _chunks
        if isinstance(c, dict)
    })


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------
def retrieve_context(query: str | None, k: int = TOP_K) -> str:
    """
    Retrieve top-k relevant chunks from FAISS index.

    Args:
        query : user question
        k     : number of chunks to return

    Returns:
        Formatted context string for LLM prompt.
    """

    try:
        load_index()

        # -------------------------------------------------------------------
        # Defensive query handling
        # -------------------------------------------------------------------
        if query is None:
            logger.warning("[RAG] Received None query. Using empty string.")
            query = ""

        if not isinstance(query, str):
            logger.warning(
                f"[RAG] Query was not string ({type(query)}). Converting."
            )
            query = str(query)

        query = query.strip()

        if not query:
            return (
                "GENERAL_KNOWLEDGE_MODE: "
                "No query text provided."
            )

        # -------------------------------------------------------------------
        # Safety checks
        # -------------------------------------------------------------------
        if _model is None:
            raise RuntimeError("[RAG] Embedding model not initialized.")

        if _index is None:
            raise RuntimeError("[RAG] FAISS index not initialized.")

        if _chunks is None:
            raise RuntimeError("[RAG] Chunk metadata not initialized.")

        if _index.ntotal == 0:
            return (
                "GENERAL_KNOWLEDGE_MODE: "
                "The FAISS index is empty."
            )

        # -------------------------------------------------------------------
        # Embed query
        # -------------------------------------------------------------------
        query_vec = _model.encode([query]).astype("float32")

        # -------------------------------------------------------------------
        # Search
        # -------------------------------------------------------------------
        k_actual = min(max(1, k), _index.ntotal)

        distances, indices = _index.search(query_vec, k_actual)

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            if idx < 0:
                continue

            if idx >= len(_chunks):
                continue

            chunk = _chunks[idx]

            source = chunk.get("source", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            text = chunk.get("text", "")

            results.append(
                f"[Source: {source} | "
                f"Chunk {chunk_id} | "
                f"relevance: {dist:.3f}]\n"
                f"{text}"
            )

        # -------------------------------------------------------------------
        # Fallback
        # -------------------------------------------------------------------
        if not results:
            return (
                "GENERAL_KNOWLEDGE_MODE: "
                "No relevant research chunks found."
            )

        return "\n\n".join(results)

    except Exception as exc:
        logger.error(f"[RAG] Retrieval failure: {exc}")

        return (
            "GENERAL_KNOWLEDGE_MODE: "
            f"Retriever failure ({exc})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def list_indexed_sources() -> list[str]:
    """Return all indexed source filenames."""
    load_index()
    return _unique_sources()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    print("\n=== RAG Retriever Self-Test ===\n")

    try:
        sources = list_indexed_sources()

        print(f"Indexed sources ({len(sources)}):")

        for s in sources:
            print(f" - {s}")

        print()

        test_queries = [
            "What is classifier-free guidance?",
            "How do latent diffusion models work?",
            "Explain DDIM sampling.",
            None,
            "",
        ]

        for q in test_queries:

            print("=" * 80)
            print(f"QUESTION: {q}")
            print("-" * 80)

            result = retrieve_context(q)

            print(result)
            print()

    except Exception as exc:
        print(f"\nSelf-test failed:\n{exc}")