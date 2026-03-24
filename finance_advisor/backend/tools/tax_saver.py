# backend/tools/tax_saver.py

from typing import Dict, Any


def suggest_tax_saving_options(
    income: float,
    regime: str,
    hra: float,
    investment_amount: float
) -> Dict[str, Any]:
    """
    Simple deterministic tax-saving recommendation engine.
    
    This tool suggests:
        - ELSS
        - PPF / EPF / LIC / Other 80C instruments
        - NPS Tier I
    
    It does not calculate tax liability (frontend or advisor agent will do that).
    Regime logic:
        - Only Old Regime allows 80C deductions.
        - New Regime allows NPS 80CCD(1B) deduction.

    Returns a dict of recommended allocations.
    """

    regime = regime.lower()
    recommendations: Dict[str, Any] = {}

    # ---------------------------------------------------
    # OLD REGIME (eligible for 80C deductions up to 1.5L)
    # ---------------------------------------------------
    if regime == "old":
        limit_80c = 150000.0
        portion_80c = min(investment_amount * 0.6, limit_80c)

        recommendations["80C_ELSS"] = min(portion_80c * 0.50, 150000.0)
        recommendations["80C_PPF_EPF_LIC"] = portion_80c - recommendations["80C_ELSS"]

        # NPS optional (80CCD(1B))
        recommendations["NPS_80CCD_1B"] = min(investment_amount * 0.20, 50000.0)

    # ---------------------------------------------------
    # NEW REGIME (limited deductions - only NPS optional)
    # ---------------------------------------------------
    elif regime == "new":
        # New regime doesn't give 80C benefit — only NPS allowed
        recommendations["NPS_80CCD_1B"] = min(investment_amount * 0.20, 50000.0)

    # ---------------------------------------------------
    # Unknown regime
    # ---------------------------------------------------
    else:
        recommendations["message"] = (
            "Unknown tax regime. Please specify 'old' or 'new'."
        )

    # ---------------------------------------------------
    # Add contextual info
    # ---------------------------------------------------
    recommendations["user_income"] = income
    recommendations["user_regime"] = regime
    recommendations["user_hra"] = hra
    recommendations["investment_amount"] = investment_amount

    return recommendations
