# backend/routers/nav.py

from fastapi import APIRouter, Query, HTTPException
from datetime import date

from ..models.nav import NAVResponse
from ..tools.finance_data import fetch_nav_data

router = APIRouter(prefix="/get_nav", tags=["market_data"])


@router.get("", response_model=NAVResponse)
def get_nav(
    symbol: str = Query(..., description="Mutual fund or stock symbol"),
    date_str: str = Query(None, alias="date", description="Optional date (YYYY-MM-DD)")
):
    """
    Universal NAV lookup endpoint.
    Uses the finance_data tool for handling live/static API logic.
    """
    try:
        if not date_str:
            date_str = date.today().isoformat()

        result = fetch_nav_data(symbol=symbol, date_str=date_str)

        return NAVResponse(
            symbol=result["symbol"],
            date=result["date"],
            nav=result["nav"],
            fund_type=result["fund_type"],
            risk_level=result["risk_level"]
        )

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
