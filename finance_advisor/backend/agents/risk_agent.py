# backend/agents/risk_agent.py

from ..models.risk import RiskProfileRequest, RiskProfileResponse
from ..tools.risk_profile import compute_risk_score
from ..memory.store import memory_store


class RiskAgent:
    """
    Agent responsible for running the risk profile engine.
    """

    @staticmethod
    def evaluate_risk(payload: RiskProfileRequest) -> RiskProfileResponse:
        result = compute_risk_score(payload)

        # Save memory
        memory_store.save_entity(
            payload.session_id,
            {
                "risk_category": result.risk_category,
                "risk_score": result.score,
            }
        )

        return result


risk_agent = RiskAgent()
