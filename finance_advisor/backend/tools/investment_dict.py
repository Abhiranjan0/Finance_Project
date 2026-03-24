# backend/tools/investment_dict.py

from typing import Dict


# -----------------------------------------------------
# Financial Dictionary
# Add/remove terms as needed.
# -----------------------------------------------------
INVESTMENT_DICTIONARY: Dict[str, str] = {
    "sip": "SIP (Systematic Investment Plan) is a method of investing a fixed amount at regular intervals into a mutual fund.",
    
    "stp": "STP (Systematic Transfer Plan) allows investors to periodically transfer a fixed sum from one mutual fund to another.",
    
    "swp": "SWP (Systematic Withdrawal Plan) enables periodic withdrawals from your investment while remaining invested.",
    
    "cagr": "CAGR (Compound Annual Growth Rate) is the annualized growth rate of an investment over a given period.",
    
    "expense_ratio": "Expense Ratio is the annual fee charged by a mutual fund, expressed as a percentage of its assets.",
    
    "sharpe": "Sharpe Ratio measures risk-adjusted returns by comparing excess return to volatility.",
    
    "alpha": "Alpha measures a fund's ability to outperform its benchmark on a risk-adjusted basis.",
    
    "beta": "Beta measures a fund's volatility relative to the market. A beta above 1 indicates higher volatility.",
    
    "nav": "NAV (Net Asset Value) is the market value per unit of a mutual fund.",
    
    "aum": "AUM (Assets Under Management) is the total market value managed by a mutual fund or asset manager.",
    
    "reit": "REIT (Real Estate Investment Trust) is a company that owns or operates income-generating real estate.",
    
    "etf": "ETF (Exchange Traded Fund) is a fund traded on stock exchanges, tracking an index or asset class.",
    
    "arbitrage_fund": "Arbitrage Funds exploit price differences in cash and derivatives markets, typically low-risk."
}


# -----------------------------------------------------
# Lookup Function
# -----------------------------------------------------
def lookup_term(term: str) -> str:
    """
    Fetches a clean definition for a given financial term.
    Returns 'Definition not found.' if term is unknown.
    """
    term = term.lower().strip()
    return INVESTMENT_DICTIONARY.get(term, "Definition not found.")
