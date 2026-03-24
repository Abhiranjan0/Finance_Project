# backend/models/nav.py

from pydantic import BaseModel


class NAVResponse(BaseModel):
    symbol: str            # mutual fund or stock symbol
    date: str              # YYYY-MM-DD
    nav: float             # latest NAV or price
    fund_type: str         # e.g., equity_large_cap, debt_short_term, etc.
    risk_level: str        # e.g., low, moderate, high
