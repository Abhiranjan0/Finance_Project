# backend/routers/portfolio.py

from fastapi import APIRouter, HTTPException, Query

from ..agents.portfolio_agent import portfolio_agent
from finance_advisor.backend.memory.store import memory_store
import traceback
from finance_advisor.backend.models.portfolio import PortfolioResponse


router = APIRouter(prefix="/portfolio", tags=["portfolio"])

@router.get("", response_model=PortfolioResponse)
def get_portfolio(session_id: str):
    try:
        result = portfolio_agent.construct_portfolio(session_id)
        allocation = result["allocation"]
        explanation = result["explanation"]

        # SAVE memory
        memory_store.save_entity(session_id, {"last_portfolio": allocation})

        return PortfolioResponse(allocation=allocation, explanation=explanation)

    except Exception as ex:
        print("\n\n------------ PORTFOLIO ROUTER ERROR ------------")
        traceback.print_exc()
        print("------------------------------------------------\n\n")
        raise HTTPException(status_code=500, detail=str(ex))
