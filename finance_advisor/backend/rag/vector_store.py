# backend/rag/vector_store.py

import os
import pickle
import faiss
from typing import List, Tuple, Dict, Any
import numpy as np


from .embedder import embed_texts


# ---------------------------------------------------------
# File Paths (as per your final folder structure)
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INDEX_DIR = os.path.join(BASE_DIR, "data", "rag_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.pkl")


# ---------------------------------------------------------
# Ensure folder exists
# ---------------------------------------------------------
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------------------------------------------------------
# Vector Store Class
# ---------------------------------------------------------
class VectorStore:
    def __init__(self):
        self.index = None
        self.meta = []  # List of { "text": ..., "source": ... }

        self._load()

    # -----------------------------------------------------
    # Load FAISS + metadata
    # -----------------------------------------------------
    def _load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.meta = pickle.load(f)
        else:
            # Empty index — L2 normalized vectors of dimension 1536 (Azure embeddings)
            self.index = faiss.IndexFlatL2(1536)
            self.meta = []

    # -----------------------------------------------------
    # Add documents to vector store
    # -----------------------------------------------------
    def add_documents(self, texts: List[str], sources: List[str]):
        embeddings = embed_texts(texts)

        self.index.add(np.array(embeddings).astype("float32"))

        # Store metadata
        for t, s in zip(texts, sources):
            self.meta.append({"text": t, "source": s})

        self._save()

    # -----------------------------------------------------
    # Save FAISS + metadata
    # -----------------------------------------------------
    def _save(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(self.meta, f)

    # -----------------------------------------------------
    # Semantic Search
    # -----------------------------------------------------
    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_vec = embed_texts([query])[0]
        query_vec = np.array([query_vec]).astype("float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])

        return results


# ---------------------------------------------------------
# Export singleton instance
# ---------------------------------------------------------
vector_store = VectorStore()
