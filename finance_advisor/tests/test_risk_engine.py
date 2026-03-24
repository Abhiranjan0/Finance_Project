from finance_advisor.backend.tools.risk_profile import compute_risk_score
from finance_advisor.backend.models.risk import RiskProfileRequest


def test_risk_score():
    req = RiskProfileRequest(
        session_id="123",
        age=25,
        income_stability="high",
        liquidity_needs="low",
        investment_knowledge="medium",
        answers={"q1": 5, "q2": 4, "q3": 3, "q4": 5}
    )
    result = compute_risk_score(req)
    assert result.risk_category in ["conservative", "moderate", "aggressive"]
    assert isinstance(result.score, int)
