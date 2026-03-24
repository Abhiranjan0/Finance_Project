# backend/routers/risk_profile.py

from fastapi import APIRouter, HTTPException

from ..models.risk import RiskProfileRequest, RiskProfileResponse
from ..memory.store import memory_store
from ..tools.risk_profile import compute_risk_score

router = APIRouter(prefix="/risk_profile", tags=["risk_profile"])


@router.post("", response_model=RiskProfileResponse)
def calculate_risk_profile(payload: RiskProfileRequest):
    """
    Processes user questionnaire + demographics to produce a risk category.
    Saves results into Redis session memory for later use.
    """
    try:
        # Compute risk profile
        result: RiskProfileResponse = compute_risk_score(payload)

        # Save entity memory into Redis
        memory_store.save_entity(
            payload.session_id,
            {
                "age": payload.age,
                "income_stability": payload.income_stability,
                "liquidity_needs": payload.liquidity_needs,
                "investment_knowledge": payload.investment_knowledge,
                "risk_category": result.risk_category,
                "risk_score": result.score
            }
        )

        return result

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
