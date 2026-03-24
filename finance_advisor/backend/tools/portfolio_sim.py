# backend/tools/portfolio_sim.py

import random
import math
from typing import Dict, Tuple, List

from ..models.simulate import (
    PortfolioSimulationRequest,
    PortfolioSimulationResponse
)


# -------------------------------------------------------------
# Expected Returns & Volatility Assumptions (Yearly)
# -------------------------------------------------------------
EXPECTED_RETURNS = {
    "equity": 0.12,     # 12% expected annual return
    "debt": 0.07,       # 7% annual
    "gold": 0.08,       # 8% annual
    "other": 0.06       # REITs / arbitrage / alternatives
}

VOLATILITY = {
    "equity": 0.18,     # High volatility
    "debt": 0.05,       # Stable
    "gold": 0.12,       # Moderate
    "other": 0.08
}


# -------------------------------------------------------------
# Helper Function: Weighted Expected Return of the Portfolio
# -------------------------------------------------------------
def compute_portfolio_parameters(allocation: Dict[str, float]) -> Tuple[float, float]:
    """
    Computes combined portfolio expected return (mu) and volatility (sigma)
    using weighted average model.

    NOTE — This is simplified.
    Real models use covariance matrices.
    """

    mu = 0.0
    sigma = 0.0

    for asset, weight in allocation.items():
        w = weight / 100.0
        mu += w * EXPECTED_RETURNS.get(asset, 0.06)
        sigma += w * VOLATILITY.get(asset, 0.10)

    return mu, sigma


# -------------------------------------------------------------
# Simulation Function
# -------------------------------------------------------------
def run_monte_carlo_simulation(
    payload: PortfolioSimulationRequest,
) -> PortfolioSimulationResponse:

    years = payload.investment.duration_years
    num_sims = payload.simulation_params.num_simulations

    # Convert model into dict
    allocation = {
        "equity": payload.allocation.equity,
        "debt": payload.allocation.debt,
        "gold": payload.allocation.gold,
        "other": payload.allocation.other or 0.0
    }

    mu, sigma = compute_portfolio_parameters(allocation)

    # Store final outcomes
    final_values: List[float] = []

    # ---------------------------------------------------------
    # Monte Carlo Simulation
    # ---------------------------------------------------------
    for _ in range(num_sims):
        # Start value
        portfolio_value = 0.0

        # SIP Mode
        if payload.investment.type == "sip":
            monthly = payload.investment.monthly_amount or 0.0
            total_months = years * 12

            for _ in range(total_months):
                yearly_return = random.gauss(mu, sigma)
                monthly_factor = (1 + yearly_return) ** (1 / 12)
                portfolio_value = (portfolio_value * monthly_factor) + monthly

        # Lumpsum Mode
        else:
            portfolio_value = payload.investment.lumpsum_amount or 0.0
            for _ in range(years):
                yearly_return = random.gauss(mu, sigma)
                portfolio_value *= (1 + yearly_return)

        final_values.append(portfolio_value)

    # ---------------------------------------------------------
    # Compute Output Statistics
    # ---------------------------------------------------------
    final_values.sort()
    n = len(final_values)

    expected_value = sum(final_values) / n
    worst_case = final_values[int(0.05 * (n - 1))]       # 5th percentile
    best_case = final_values[int(0.95 * (n - 1))]        # 95th percentile

    # Default goal assumption (can be customized)
    goal_amount = 10_000_000  # 1 Cr
    probability_of_goal = sum(1 for v in final_values if v >= goal_amount) / n

    return PortfolioSimulationResponse(
        expected_value=expected_value,
        worst_case=worst_case,
        best_case=best_case,
        probability_of_goal_achievement=probability_of_goal
    )
