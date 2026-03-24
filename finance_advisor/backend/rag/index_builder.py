# backend/rag/index_builder.py

"""
Index Builder for RAG Pipeline
------------------------------

This script scans your dataset folders inside `data/`:
- data/sebi_guidelines/
- data/mutual_funds/
- data/sample_portfolios/
- data/tax_rules/

It extracts text, embeds it using Azure OpenAI, and stores the
FAISS index + metadata inside `data/rag_index/`.

Run manually:
    python backend/rag/index_builder.py
"""

import os
import json
import pickle
import faiss
import numpy as np

from .embedder import embed_texts
from .vector_store import INDEX_DIR, INDEX_FILE, META_FILE


# -------------------------------------------------------------------
# 1. Helper: Load text from PDFs / JSON / TXT
# -------------------------------------------------------------------

def load_text_from_file(filepath: str) -> str:
    """
    Loads text depending on file type.
    This is a simple version:
    - .txt → read directly
    - .json → read string fields
    - .pdf → requires PyPDF2
    """
    ext = filepath.lower().split(".")[-1]

    # TXT FILES
    if ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # JSON FILES
    if ext == "json":
        with open(filepath, "r") as f:
            data = json.load(f)

        # If JSON is object with key-value, convert nicely to text
        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        return str(data)

    # PDF FILES
    if ext == "pdf":
        try:
            import PyPDF2
            text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception:
            print(f"[Warning] Could not read PDF: {filepath}")
            return ""

    print(f"[Warning] Unsupported file type: {filepath}")

    if ext == "csv":
        try:
            import csv
            rows = []
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    # join cells with a comma and space for readability
                    rows.append(", ".join(cell for cell in row))
                # Informational print: how many rows were read from this CSV
            print(f"[RAG][CSV] Read {len(rows)} rows from: {filepath}")
                # Return as newline-separated text (header + rows)
            return "\n".join(rows)
        except Exception:
            print(f"[Warning] Could not read CSV: {filepath}")
            return ""

    print(f"[Warning] Unsupported file type: {filepath}")
    return ""


# -------------------------------------------------------------------
# 2. Gather ALL documents from data directories
# -------------------------------------------------------------------

def collect_documents() -> tuple[list[str], list[str]]:
    """
    Returns:
        texts:   list of document chunks
        sources: list of filenames for metadata
    """

    base_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data")
    )

    folders = [
        "sebi_guidelines",
        "mutual_funds",
        "sample_portfolios",
        "financial_definitions"
    ]

    docs = []
    sources = []

    for folder in folders:
        folder_path = os.path.join(base_data, folder)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            filepath = os.path.join(folder_path, file)
            if os.path.isfile(filepath):
                text = load_text_from_file(filepath)
                if len(text.strip()) > 0:
                    # Chunk large documents into smaller pieces
                    chunks = chunk_text(text)

                    for c in chunks:
                        docs.append(c)
                        sources.append(f"{folder}/{file}")

    return docs, sources


# -------------------------------------------------------------------
# 3. Chunk text to 300–500 token pieces (safe for embeddings)
# -------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800) -> list[str]:
    """
    Simple word-based chunking.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------------------------------------------
# 4. Build & Save FAISS Index
# -------------------------------------------------------------------

def build_index():
    print("[RAG] Collecting documents...")
    texts, sources = collect_documents()
    print(f"[RAG] Loaded {len(texts)} chunks.")

    if len(texts) == 0:
        print("[RAG] No documents found. Fill your data folder first.")
        return

    print("[RAG] Generating embeddings (this may take a while)...")
    embeddings = embed_texts(texts)

    embeddings_np = np.array(embeddings).astype("float32")

    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("[RAG] Adding vectors to FAISS index...")
    index.add(embeddings_np)

    print("[RAG] Saving index and metadata...")
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    meta = [{"text": t, "source": s} for t, s in zip(texts, sources)]
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

    print(f"[RAG] Done! Index built with {len(texts)} vectors.")
    print(f"[RAG] Index stored at: {INDEX_FILE}")
    print(f"[RAG] Metadata stored at: {META_FILE}")


# -------------------------------------------------------------------
# 5. Run when executed directly
# -------------------------------------------------------------------

if __name__ == "__main__":
    build_index()
