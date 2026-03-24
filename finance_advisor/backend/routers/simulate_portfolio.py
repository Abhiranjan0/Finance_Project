# backend/routers/simulate_portfolio.py

from fastapi import APIRouter, HTTPException

from ..models.simulate import (
    PortfolioSimulationRequest,
    PortfolioSimulationResponse,
)
from ..tools.portfolio_sim import run_monte_carlo_simulation

router = APIRouter(prefix="/simulate_portfolio", tags=["portfolio_simulation"])


@router.post("", response_model=PortfolioSimulationResponse)
def simulate_portfolio(payload: PortfolioSimulationRequest):
    """
    Runs Monte Carlo simulation or expected return calculations
    based on user allocation and investment type.
    """
    try:
        result: PortfolioSimulationResponse = run_monte_carlo_simulation(payload)
        
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
