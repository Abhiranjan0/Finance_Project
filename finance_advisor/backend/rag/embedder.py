# backend/rag/embedder.py

from typing import List
from ..groq_client import create_embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Wrapper around Azure OpenAI embedding API.
    Accepts list of texts and returns list of embeddings.
    """
    if not isinstance(texts, list):
        texts = [texts]

    embeddings = create_embeddings(texts)
    return embeddings
