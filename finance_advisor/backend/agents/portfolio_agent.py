# backend/agents/portfolio_agent.py

from typing import Dict, Any

from ..tools.portfolio_engine import build_portfolio, explain_portfolio
from ..memory.store import memory_store


class PortfolioAgent:
    """
    Agent responsible for constructing asset allocation.
    """

    @staticmethod
    def construct_portfolio(session_id: str) -> Dict[str, Any]:
        entity = memory_store.get_entity(session_id)
        risk = entity.get("risk_category", "moderate")
        tenure = entity.get("tenure_years", None)

        allocation = build_portfolio(risk_category=risk, tenure_years=tenure)
        explanation = explain_portfolio(allocation, risk)

        # Save allocation in memory
        memory_store.save_entity(session_id, {"recommended_portfolio": allocation})


        return {
            "allocation": allocation,
            "explanation": explanation
        }


portfolio_agent = PortfolioAgent()
