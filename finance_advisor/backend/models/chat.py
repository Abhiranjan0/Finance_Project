# backend/models/chat.py

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class ChatMetadata(BaseModel):
    channel: Optional[str] = "streamlit"
    language: Optional[str] = "en"


class ChatRequest(BaseModel):
    session_id: str
    message: str
    metadata: Optional[ChatMetadata] = None


class ChatResponse(BaseModel):
    reply: str
    used_tools: Optional[List[str]] = []
    memory_updates: Optional[Dict[str, Any]] = {}
