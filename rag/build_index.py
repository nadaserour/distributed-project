# rag/build_index.py

import pickle
import time
from pathlib import Path

import fitz
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Settings (OPTIMIZED FOR SPEED)
# ------------------------------------------------------------------
DOCS_FOLDER  = Path(__file__).parent / "docs"
INDEX_FOLDER = Path(__file__).parent / "index"

CHUNK_SIZE    = 800      # ↑ bigger chunks = fewer embeddings (FASTER)
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_CHUNKS = 5000        # 🔥 HARD LIMIT (prevents laptop death)
BATCH_SIZE = 128         # ↑ faster embedding

# ------------------------------------------------------------------
# PDF loader
# ------------------------------------------------------------------
def load_pdf(path: Path) -> str:
    doc = fitz.open(path)
    text = []
    for page in doc:
        t = page.get_text()
        if t.strip():
            text.append(t.strip())
    doc.close()
    return "\n".join(text)


# ------------------------------------------------------------------
# Chunking (FAST version)
# ------------------------------------------------------------------
def chunk_text(text: str, source: str):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i in range(0, len(text), step):
        chunk = text[i:i + CHUNK_SIZE].strip()

        if len(chunk) > 50:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": len(chunks)
            })

        # 🔥 Stop early if too many chunks
        if len(chunks) >= MAX_CHUNKS:
            break

    return chunks


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    t0 = time.time()

    print("\n" + "="*60)
    print("  FAST RAG Index Builder")
    print("="*60)

    DOCS_FOLDER.mkdir(exist_ok=True)
    INDEX_FOLDER.mkdir(exist_ok=True)

    pdf_files = list(DOCS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    all_chunks = []

    # -------------------------------
    # LOAD + CHUNK (WITH PROGRESS)
    # -------------------------------
    print("\n[RAG] Processing PDFs...\n")

    for pdf in tqdm(pdf_files, desc="PDFs"):
        text = load_pdf(pdf)

        chunks = chunk_text(text, pdf.name)
        all_chunks.extend(chunks)

        if len(all_chunks) >= MAX_CHUNKS:
            print("\n⚠ Reached MAX_CHUNKS limit → stopping early")
            break

    print(f"\n[RAG] Total chunks: {len(all_chunks)}")

    # -------------------------------
    # EMBEDDING (FAST + PROGRESS)
    # -------------------------------
    print("\n[RAG] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]

    print("[RAG] Encoding (this is the heavy part)...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # 🔥 faster similarity later
    ).astype("float32")

    # -------------------------------
    # FAISS
    # -------------------------------
    print("\n[RAG] Building FAISS index...")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # 🔥 faster than L2 when normalized
    index.add(embeddings)

    # -------------------------------
    # SAVE
    # -------------------------------
    faiss.write_index(index, str(INDEX_FOLDER / "faiss.index"))

    with open(INDEX_FOLDER / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)

    print(f"Chunks indexed : {len(all_chunks)}")
    print(f"Time taken     : {time.time() - t0:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()