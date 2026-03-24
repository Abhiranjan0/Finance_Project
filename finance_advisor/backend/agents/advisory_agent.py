# backend/agents/advisory_agent.py

from ..groq_client import chat_completion_text
from ..memory.store import memory_store


class AdvisoryAgent:
    """
    Uses GPT to produce final personalized advisory summary.
    """

    SYSTEM_PROMPT = (
        "You are the final advisory agent. Based on the user's goals, "
        "risk profile, recommended portfolio, and simulation results, "
        "create a clear, friendly financial plan summary. "
        "Do not promise guaranteed returns. Follow SEBI-compliant language."
    )

    @staticmethod
    def generate_advice(session_id: str) -> str:
        entity = memory_store.get_entity(session_id)

        messages = [
            {"role": "system", "content": AdvisoryAgent.SYSTEM_PROMPT},
            {"role": "system", "content": f"User Data: {entity}"},
        ]

        portfolio = entity.get("recommended_portfolio", {})
        sim = entity.get("simulation_results", {})

        messages.append({"role": "system", "content": f"Portfolio: {portfolio}"})
        messages.append({"role": "system", "content": f"Simulation: {sim}"})


        return chat_completion_text(messages, temperature=0.3)


advisory_agent = AdvisoryAgent()
