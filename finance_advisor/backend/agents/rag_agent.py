# backend/agents/rag_agent.py

from typing import List, Dict, Any

from ..rag.retriever import retrieve_top_k


class RAGAgent:
    """
    The agent responsible for regulatory lookup.
    """

    @staticmethod
    def lookup(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns top-k relevant text chunks from the vector store.
        """
        chunks = retrieve_top_k(query, top_k)
        return [
            {"text": c["text"], "source": c["source"]}
            for c in chunks
        ]


rag_agent = RAGAgent()
