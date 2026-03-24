# backend/models/rag.py

from pydantic import BaseModel
from typing import List


class RAGRequest(BaseModel):
    query: str                      # User's natural language question
    top_k: int = 5                  # Number of chunks to retrieve


class RAGChunk(BaseModel):
    text: str                       # Retrieved content chunk
    source: str                     # Origin document (SEBI, MF, etc.)


class RAGResponse(BaseModel):
    context: List[RAGChunk]         # List of returned chunks
