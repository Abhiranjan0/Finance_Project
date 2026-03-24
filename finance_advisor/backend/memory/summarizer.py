# backend/memory/summarizer.py

from typing import List, Dict
from ..groq_client import chat_completion_text
from .store import memory_store


class ConversationSummarizer:
    """
    Produces a concise summary of the conversation using Azure GPT.
    """

    SYSTEM_INSTRUCTION = (
        "You are an assistant that summarizes financial advisory conversations. "
        "Create a short, structured summary capturing goals, constraints, risk indicators, "
        "preferences, and important facts shared by the user. "
        "Do NOT include irrelevant small-talk. Keep it factual and compact."
    )

    @staticmethod
    def summarize(
        session_id: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Given a list of chat messages, generate a summary using Azure GPT.
        Messages must be in [{"role": "...", "content": "..."}] format.
        """
        messages = [
            {"role": "system", "content": ConversationSummarizer.SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Here is the conversation history:\n{chat_history}"},
        ]

        summary = chat_completion_text(
            messages,
            temperature=0.1,
            max_tokens=250
        )

        # Persist summary in memory store
        memory_store.update_summary(session_id, summary)
        return summary


# Export singleton
summarizer = ConversationSummarizer()
