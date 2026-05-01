# rag/build_index.py
# Run this ONCE (or whenever you update the PDFs in rag/docs/).
# It reads every PDF, splits it into overlapping chunks, embeds them,
# and saves the FAISS index + chunk metadata to rag/index/.
#
# Usage:
#   python -m rag.build_index
#   -- or --
#   cd rag && python build_index.py

import pickle
import numpy as np
from pathlib import Path

import fitz                          # PyMuPDF  →  pip install pymupdf
import faiss                         # pip install faiss-cpu
from sentence_transformers import SentenceTransformer   # pip install sentence-transformers

# ---------------------------------------------------------------------------
# Paths & settings
# ---------------------------------------------------------------------------
DOCS_FOLDER  = Path(__file__).parent / "docs"
INDEX_FOLDER = Path(__file__).parent / "index"

CHUNK_SIZE    = 600    # characters per chunk (increased slightly for richer context)
CHUNK_OVERLAP = 80     # characters of overlap so sentences aren't cut off at boundaries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------
def load_pdf(path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Text chunking  (sentence-aware)
# ---------------------------------------------------------------------------
def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks, trying to break at sentence
    boundaries (period, newline, !, ?) rather than mid-word.

    Returns list of dicts:
        { "text": str, "source": filename, "chunk_id": int }
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # If not at the very end, try to find the last sentence boundary
        # within the current window so we don't cut mid-sentence.
        if end < len(text):
            boundary = max(
                text.rfind(". ",  start, end),
                text.rfind("\n",  start, end),
                text.rfind("! ",  start, end),
                text.rfind("? ",  start, end),
                text.rfind(".\n", start, end),
            )
            # Only use the boundary if it's in the second half of the chunk
            # (avoids tiny chunks when a sentence boundary is near the start)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1

        chunk_text = text[start:end].strip()

        # Skip whitespace-only or very short chunks (e.g. page headers)
        if len(chunk_text) > 40:
            chunks.append({
                "text":     chunk_text,
                "source":   source,
                "chunk_id": len(chunks),
            })

        start = end - CHUNK_OVERLAP   # step back by overlap amount

    return chunks


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  RAG Index Builder — College Advisor")
    print("="*60)

    DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
    INDEX_FOLDER.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all PDFs ──────────────────────────────────────────────────
    pdf_files = sorted(DOCS_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[!] No PDF files found in:  {DOCS_FOLDER.resolve()}")
        print("    Drop your bylaws / course-catalogue PDFs there and re-run.\n")
        return

    all_chunks: list[dict] = []

    for pdf_path in pdf_files:
        print(f"\n[+] Loading: {pdf_path.name}")
        text = load_pdf(pdf_path)

        if not text.strip():
            print(f"    [!] Could not extract text from {pdf_path.name} — skipping.")
            continue

        print(f"    Extracted {len(text):,} characters")
        chunks = chunk_text(text, source=pdf_path.name)
        print(f"    Created   {len(chunks):,} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\n[!] No chunks were created. Check that your PDFs contain selectable text.\n")
        return

    print(f"\n[RAG] Total chunks across all documents: {len(all_chunks)}")

    # ── 2. Embed all chunks ───────────────────────────────────────────────
    print(f"\n[RAG] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("[RAG] Encoding chunks (this may take a minute)…")
    texts      = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # ── 3. Build & save FAISS index ───────────────────────────────────────
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"\n[RAG] FAISS index built: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, str(INDEX_FOLDER / "faiss.index"))
    print(f"[RAG] Saved: {INDEX_FOLDER / 'faiss.index'}")

    with open(INDEX_FOLDER / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"[RAG] Saved: {INDEX_FOLDER / 'chunks.pkl'}")

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Index build complete!")
    print(f"  Documents indexed : {len(pdf_files)}")
    print(f"  Total chunks      : {len(all_chunks)}")
    print(f"  Embedding dim     : {dim}")
    print("="*60)
    print("\nYou can now start the system with:  python main.py\n")


if __name__ == "__main__":
    main()