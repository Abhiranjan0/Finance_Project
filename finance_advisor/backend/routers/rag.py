# backend/routers/rag.py

from fastapi import APIRouter, HTTPException

from ..models.rag import RAGRequest, RAGResponse, RAGChunk
from ..rag.retriever import retrieve_top_k

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("", response_model=RAGResponse)
def rag_search(payload: RAGRequest):
    """
    Retrieves the top-k relevant SEBI rules / MF definitions / tax rules
    using Azure embeddings + vector database (FAISS/Chroma/CosmosDB).
    """
    try:
        chunks = retrieve_top_k(payload.query, payload.top_k)

        rag_chunks = [
            RAGChunk(text=c["text"], source=c["source"])
            for c in chunks
        ]

        return RAGResponse(context=rag_chunks)

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
