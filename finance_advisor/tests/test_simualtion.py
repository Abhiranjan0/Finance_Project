from finance_advisor.backend.tools.portfolio_sim import run_monte_carlo_simulation
from finance_advisor.backend.models.simulate import *


def test_simulation_basic():
    req = PortfolioSimulationRequest(
        session_id="1",
        allocation=Allocation(equity=50, debt=40, gold=5, other=5),
        investment=InvestmentDetails(type="sip", monthly_amount=10000, duration_years=10),
        simulation_params=SimulationParams(num_simulations=100)
    )
    result = run_monte_carlo_simulation(req)
    assert result.expected_value > 0
