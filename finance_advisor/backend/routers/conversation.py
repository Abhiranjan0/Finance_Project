from fastapi import APIRouter
from finance_advisor.backend.db.conversation_store import get_conversation

router = APIRouter(prefix="/conversation", tags=["conversation"])

@router.get("/{session_id}")
def fetch_conversation(session_id: str):
    messages = get_conversation(session_id)
    formatted = [
        {"role": m.role, "message": m.message, "timestamp": m.timestamp}
        for m in messages
    ]
    return {"history": formatted}
