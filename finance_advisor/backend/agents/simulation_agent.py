# backend/agents/simulation_agent.py

from typing import Dict, Any

from ..tools.portfolio_sim import run_monte_carlo_simulation
from ..models.simulate import (
    PortfolioSimulationRequest,
    Allocation,
    InvestmentDetails,
    SimulationParams
)
from ..memory.store import memory_store


class SimulationAgent:
    """
    Agent responsible for running portfolio simulations.
    """

    @staticmethod
    def run_simulation(session_id: str) -> Dict[str, Any]:

        # Load entity memory from Redis
        entity = memory_store.get_entity(session_id)

        # FIX: read the correct stored key from Redis
        portfolio = entity.get("last_portfolio")

        if not portfolio:
            raise ValueError("No portfolio available to simulate. Run PortfolioAgent first.")

        # Extract investment preferences (fallback defaults)
        investment_type = entity.get("investment_type", "sip")
        monthly_amount = entity.get("monthly_investment", 10000)
        lumpsum_amount = entity.get("lumpsum_investment")
        duration = entity.get("tenure_years", 10)

        # Build simulation request object
        request = PortfolioSimulationRequest(
            session_id=session_id,
            allocation=Allocation(**portfolio),
            investment=InvestmentDetails(
                type=investment_type,
                monthly_amount=monthly_amount,
                lumpsum_amount=lumpsum_amount,
                duration_years=duration
            ),
            simulation_params=SimulationParams(num_simulations=5000)
        )

        # Run Monte Carlo
        result = run_monte_carlo_simulation(request)

        # Return results in user-friendly format
        return {
            "expected_value": result.expected_value,
            "best_case": result.best_case,
            "worst_case": result.worst_case,
            "probability_of_goal_achievement": result.probability_of_goal_achievement
        }


simulation_agent = SimulationAgent()
