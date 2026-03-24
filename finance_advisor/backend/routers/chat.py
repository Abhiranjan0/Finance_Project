# backend/routers/chat.py

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json

from finance_advisor.backend.groq_client import client
from ..config import settings
from finance_advisor.backend.memory.store import memory_store
from finance_advisor.backend.models.chat import ChatRequest, ChatResponse

# Our custom MCP-like tool server
from finance_advisor.backend.mcp.server import get_mcp_schema, call_mcp_tool
from finance_advisor.backend.guardrails.input_guard import check_user_input
from finance_advisor.backend.guardrails.output_guard import sanitize_output, append_disclaimer

from finance_advisor.backend.db.conversation_store import save_message
from finance_advisor.backend.db.user_store import ensure_user





router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    PURE MCP mode:
    - No manual intent detection
    - No agents
    - Azure GPT sees all tools and decides which one to call
    - Tools executed by our MCP tool registry
    """

    try:
        session_id = payload.session_id

        ensure_user(session_id)

        # -----------------------------
        # 0. Input guardrails
        # -----------------------------
        allowed, guard_msg = check_user_input(payload.message)
        if not allowed:
            return ChatResponse(reply=guard_msg)
        
        # Save the user message into SQLite
        save_message(session_id, "user", payload.message)

        # -----------------------------
        # Load memory
        # -----------------------------
        entity = memory_store.get_entity(session_id)
        summary = memory_store.get_summary(session_id)

        system_prompt = (
            "You are a qualified Indian financial advisor. "
            "You are a SEBI-aware Indian financial advisor assistant. "
            "You MUST follow these rules strictly:\n"
            "1. Do NOT provide guaranteed, risk-free, or sure-shot returns.\n"
            "2. Do NOT suggest illegal, unethical, or non-compliant practices "
            "(including insider trading, market manipulation, tax evasion, or misuse of financial products).\n"
            "3. For product-definitions or financial terms, provide accurate definitions based on your knowledge.\n"
            "4. For regulatory, SEBI, or product-definition questions, provide accurate information.\n"
            "5. Make risk disclosures explicit and remind the user that all market-linked products carry risk.\n"
            "6. If a user asks for something unsafe, illegal, or outside allowed scope, politely refuse and explain why.\n"
            "7. Always ensure safety, SEBI compliance, and clarity."
        )

        memory_context = (
            f"User Profile Memory: {entity}\n"
            f"Summary Memory: {summary or 'None'}"
        )

        # -----------------------------
        # Build LLM messages
        # -----------------------------
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": memory_context},
            {"role": "user", "content": payload.message},
        ]

        # -----------------------------
        # Ask LLM
        # -----------------------------
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
        )

        raw_reply = response.choices[0].message.content or ""

        # --------------------------------------
        # Apply output guardrails
        # --------------------------------------
        cleaned_text, _ = sanitize_output(raw_reply)
        final_reply = append_disclaimer(cleaned_text)

        # Save assistant reply
        save_message(session_id, "assistant", final_reply)
        memory_store.save_summary(session_id, final_reply)


        return ChatResponse(reply=final_reply)

    except Exception as ex:
        print("----------- BACKEND /chat ERROR -----------")
        import traceback
        traceback.print_exc()
        print("--------------------------------------------")
        raise HTTPException(status_code=500, detail=str(ex))
