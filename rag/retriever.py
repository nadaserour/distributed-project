# rag/retriever.py
# Loads the pre-built FAISS index from disk and answers semantic queries.
# Build the index first by running:  python -m rag.build_index

import pickle
import logging
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & settings  (must match build_index.py)
# ---------------------------------------------------------------------------
INDEX_FOLDER    = Path(__file__).parent / "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 3     # number of chunks returned per query

# ---------------------------------------------------------------------------
# Lazy-loaded globals  (loaded once on first query, shared by all workers)
# ---------------------------------------------------------------------------
_model:  SentenceTransformer | None = None
_index:  faiss.Index               | None = None
_chunks: list[dict]                | None = None


def load_index() -> None:
    """
    Load the FAISS index and chunk metadata from disk into memory.
    Called automatically on the first retrieve_context() call.
    Only loads once — subsequent calls are instant (globals already set).
    """
    global _model, _index, _chunks

    if _index is not None:
        return   # already loaded

    index_path  = INDEX_FOLDER / "faiss.index"
    chunks_path = INDEX_FOLDER / "chunks.pkl"

    # ── Friendly error if build_index.py hasn't been run yet ─────────────
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "\n[RAG] Index files not found!\n"
            f"  Expected: {index_path}\n"
            f"  Expected: {chunks_path}\n\n"
            "  Please build the index first:\n"
            "    python -m rag.build_index\n"
        )

    logger.info("[RAG] Loading FAISS index from disk…")
    _index  = faiss.read_index(str(index_path))

    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    logger.info(f"[RAG] Loaded {len(_chunks)} chunks from {len(_unique_sources())} documents.")

    logger.info(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
    _model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("[RAG] Retriever ready.")


def _unique_sources() -> list[str]:
    """Return unique source document names currently in _chunks."""
    if _chunks is None:
        return []
    return sorted({c["source"] for c in _chunks})


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------
def retrieve_context(query: str, k: int = TOP_K) -> str:
    """
    Main public function — called by gpu_worker.py.

    1. Embeds the query into a vector
    2. Searches the FAISS index for the k nearest chunk vectors
    3. Returns those chunks formatted as a context string for the LLM

    Args:
        query : the student's question / request
        k     : how many chunks to retrieve (default TOP_K)

    Returns:
        A formatted string with source labels and chunk text,
        ready to be injected into the LLM prompt.
    """
    load_index()   # no-op if already loaded

    # Embed the query
    query_vec = _model.encode([query]).astype("float32")

    # Search — k can't exceed index size
    k_actual = min(k, _index.ntotal)
    distances, indices = _index.search(query_vec, k_actual)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(_chunks):
            chunk = _chunks[idx]
            results.append(
                f"[Source: {chunk['source']} | Chunk {chunk['chunk_id']} "
                f"| relevance: {dist:.3f}]\n"
                f"{chunk['text']}"
            )

    if not results:
        return "No relevant information found in the college bylaws for this query."

    return "\n\n".join(results)


def list_indexed_sources() -> list[str]:
    """Utility: return the names of all indexed PDF files."""
    load_index()
    return _unique_sources()


# ---------------------------------------------------------------------------
# Self-test:  python -m rag.retriever
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\n=== College Advisor RAG — Self-Test ===\n")
    print(f"Indexed sources: {list_indexed_sources()}\n")

    test_queries = [
        "What is classifier-free guidance?",
        "How does the latent diffusion model work?",
        "What is the FID score of DiT-XL/2?",
        "How does SDXL improve over previous stable diffusion models?",
        "What is DDIM sampling?",
    ]

    for q in test_queries:
        print(f"Question : {q}")
        print(retrieve_context(q))
        print("-" * 70)