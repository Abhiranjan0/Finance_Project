# backend/models/simulate.py

from pydantic import BaseModel
from typing import Optional, Literal


# -----------------------------------------
# Asset Allocation Schema
# -----------------------------------------
class Allocation(BaseModel):
    equity: float                    # percentage (0–100)
    debt: float                      # percentage (0–100)
    gold: float                      # percentage (0–100)
    other: Optional[float] = 0.0     # percentage (could be REITs, intl equity, etc.)


# -----------------------------------------
# Investment Details Schema
# -----------------------------------------
class InvestmentDetails(BaseModel):
    type: Literal["sip", "lumpsum"]  # investment mode

    # SIP fields
    monthly_amount: Optional[float] = None

    # Lumpsum field
    lumpsum_amount: Optional[float] = None

    # Duration
    duration_years: int              # investment horizon in years


# -----------------------------------------
# Simulation Parameters Schema
# -----------------------------------------
class SimulationParams(BaseModel):
    num_simulations: int = 5000      # number of Monte Carlo runs


# -----------------------------------------
# Full Simulation Request Schema
# -----------------------------------------
class PortfolioSimulationRequest(BaseModel):
    session_id: str
    allocation: Allocation
    investment: InvestmentDetails
    simulation_params: SimulationParams


# -----------------------------------------
# Simulation Response Schema
# -----------------------------------------
class PortfolioSimulationResponse(BaseModel):
    expected_value: float
    best_case: float
    worst_case: float
    probability_of_goal_achievement: float
