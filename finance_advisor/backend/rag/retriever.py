# backend/rag/retriever.py

from typing import List, Dict, Any

from .vector_store import vector_store


def retrieve_top_k(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Main RAG retrieval function.
    Returns list of { "text": ..., "source": ... }
    """
    try:
        results = vector_store.search(query, top_k)
        print(f"[RAG] Retrieved {len(results)} results for query: {query}") #Remove it before LIVE deployment
        # Each result is already a dict with text + source
        return results

    except Exception as ex:
        # On any failure, return empty list (safer for the advisor)
        print(f"[RAG] Retrieval error: {ex}")
        return []
