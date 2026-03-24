# backend/tools/risk_profile.py

from ..models.risk import RiskProfileRequest, RiskProfileResponse


def compute_risk_score(payload: RiskProfileRequest) -> RiskProfileResponse:
    """
    Computes a weighted score and risk category based on:
    - Age
    - Income stability
    - Liquidity needs
    - Investment knowledge
    - Questionnaire answers (MCQ)
    """

    score = 0

    # -----------------------------------
    # Age factor (younger => more aggressive)
    # -----------------------------------
    if payload.age < 30:
        score += 25
    elif payload.age < 45:
        score += 15
    elif payload.age < 60:
        score += 5
    else:
        score -= 10

    # -----------------------------------
    # Income Stability
    # -----------------------------------
    stability_map = {
        "low": -10,
        "medium": 5,
        "high": 15
    }
    score += stability_map.get(payload.income_stability.lower(), 0)

    # -----------------------------------
    # Liquidity Needs
    # Higher liquidity needs => less aggressive
    # -----------------------------------
    liquidity_map = {
        "low": 15,
        "medium": 0,
        "high": -15
    }
    score += liquidity_map.get(payload.liquidity_needs.lower(), 0)

    # -----------------------------------
    # Investment Knowledge
    # -----------------------------------
    knowledge_map = {
        "low": -5,
        "medium": 5,
        "high": 10
    }
    score += knowledge_map.get(payload.investment_knowledge.lower(), 0)

    # -----------------------------------
    # Questionnaire Scores
    # -----------------------------------
    score += sum(payload.answers.values())

    # -----------------------------------
    # Risk Category Classification
    # -----------------------------------
    if score >= 70:
        category = "aggressive"
    elif score >= 45:
        category = "moderate"
    else:
        category = "conservative"

    # -----------------------------------
    # Explanation
    # -----------------------------------
    explanation = (
        f"Your risk score is {score}. Based on your age ({payload.age}), "
        f"income stability ({payload.income_stability}), liquidity needs "
        f"({payload.liquidity_needs}), investment knowledge "
        f"({payload.investment_knowledge}), and questionnaire responses, "
        f"you are categorized as a '{category}' investor."
    )

    return RiskProfileResponse(
        risk_category=category,
        score=score,
        explanation=explanation
    )
