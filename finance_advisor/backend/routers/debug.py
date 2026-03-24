# backend/routers/debug.py

from fastapi import APIRouter
from ..memory.store import memory_store

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/reset")
def reset_all():
    memory_store._store = {}
    return {"status": "memory cleared", "message": "All sessions wiped"}
