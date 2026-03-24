# backend/tools/portfolio_engine.py

from typing import Dict, Any


def build_portfolio(risk_category: str, tenure_years: int = None) -> Dict[str, float]:
    """
    Builds a simple, deterministic, SEBI-friendly portfolio allocation
    based solely on risk category. 
    Later, additional factors (goal horizon, liquidity needs, etc.) can be added.

    Returns allocation in percentages (must sum to 100).
    """

    risk_category = risk_category.lower()

    # ------------------------------
    # Conservative Portfolio
    # ------------------------------
    if risk_category == "conservative":
        return {
            "equity": 20.0,
            "debt": 60.0,
            "gold": 10.0,
            "other": 10.0,  # REITs / Arbitrage / Liquid funds
        }

    # ------------------------------
    # Moderate Portfolio
    # ------------------------------
    if risk_category == "moderate":
        return {
            "equity": 40.0,
            "debt": 40.0,
            "gold": 10.0,
            "other": 10.0,
        }

    # ------------------------------
    # Aggressive Portfolio
    # ------------------------------
    if risk_category == "aggressive":
        return {
            "equity": 70.0,
            "debt": 20.0,
            "gold": 5.0,
            "other": 5.0,
        }

    # --------------------------------------------
    # Fallback if unknown risk category is passed
    # --------------------------------------------
    return {
        "equity": 50.0,
        "debt": 40.0,
        "gold": 5.0,
        "other": 5.0,
    }


def explain_portfolio(allocation: Dict[str, float], risk_category: str) -> str:
    """
    Returns a friendly explanation GPT can use or frontend can show.
    No GPT reasoning is done here — purely rule-based generation.
    """

    explanation = (
        f"The recommended portfolio for a '{risk_category}' investor is:\n\n"
        f"- Equity: {allocation['equity']}%\n"
        f"- Debt: {allocation['debt']}%\n"
        f"- Gold: {allocation['gold']}%\n"
        f"- Other Assets (REITs, Arbitrage funds): {allocation['other']}%\n\n"
        "This allocation is based on your risk appetite and aims to balance "
        "growth potential with stability. Equity provides long-term growth, "
        "debt offers stability and predictable returns, gold acts as a hedge "
        "against inflation, and other assets provide diversification."
    )

    return explanation
