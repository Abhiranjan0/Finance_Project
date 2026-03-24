# backend/agents/intake_agent.py

from typing import List, Dict, Any

from ..memory.store import memory_store
from ..groq_client import chat_completion_text


class IntakeAgent:
    """
    Responsible for collecting missing user information.
    GPT calls this agent when critical profile details are missing.
    """

    SYSTEM_PROMPT = (
        "You are the intake agent. Your job is to identify what financial "
        "information is missing from the user's profile and ask clear, "
        "simple follow-up questions to gather it."
    )

    @staticmethod
    def ask_for_missing_info(session_id: str, user_message: str) -> str:
        """
        GPT asks: what info is missing?
        GPT responds: what should we ask the user next?
        """
        entity = memory_store.get_entity(session_id)

        missing = []

        if "age" not in entity:
            missing.append("age")

        if "tenure_years" not in entity:
            missing.append("investment horizon (tenure in years)")

        if "risk_category" not in entity:
            missing.append("risk profile")

        # At least one investment amount must be set
        if "monthly_investment" not in entity and "lumpsum_investment" not in entity:
            missing.append("investment amount (SIP or lumpsum)")

        if missing:
            missing_str = ", ".join(missing)
            return f"Before I can proceed, I need the following information: {missing_str}. Please provide them."

        summary = f"Known user info: {entity}"

        messages = [
            {"role": "system", "content": IntakeAgent.SYSTEM_PROMPT},
            {"role": "system", "content": summary},
            {"role": "user", "content": user_message},
        ]

        return chat_completion_text(messages)


intake_agent = IntakeAgent()
