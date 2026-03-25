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
    # Normalize questionnaire score to 0-25 range
    # -----------------------------------
    try:
        q_sum = sum(int(v) for v in payload.answers.values())
    except Exception:
        q_sum = 0

    normalized_questionnaire_score = int((q_sum / 20.0) * 25)  # 4..20 -> 5..25
    score += normalized_questionnaire_score

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
    # Range context (for all categories guidance)
    # -----------------------------------
    explanation = (
        f"Your risk score is {score}. "
        f"Based on your age ({payload.age}), income stability ({payload.income_stability}), "
        f"liquidity needs ({payload.liquidity_needs}), investment knowledge "
        f"({payload.investment_knowledge}), and questionnaire responses, "
        f"you are categorized as a '{category}' investor." +
        "\n\n" +
        "Risk category ranges: conservative (0-44), moderate (45-69), aggressive (70-100).\n" +
        "\nFor reference, here are all categories based on the computed profile:\n" +
        "- conservative: Focus on capital preservation, minimal risky allocation.\n" +
        "- moderate: Balanced allocation in equities and fixed income.\n" +
        "- aggressive: Higher equity focus, suitable for longer horizon and higher risk tolerance."
    )


    return RiskProfileResponse(
        risk_category=category,
        score=score,
        explanation=explanation
    )
