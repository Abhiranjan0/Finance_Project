# backend/routers/simulation.py

from fastapi import APIRouter, HTTPException, Query

from ..agents.simulation_agent import simulation_agent
from finance_advisor.backend.memory.store import memory_store
router = APIRouter(prefix="/simulate", tags=["simulation"])
import traceback



@router.get("")
def simulate_portfolio(session_id: str = Query(..., description="User session ID")):
    """
    Runs Monte Carlo simulation using SimulationAgent.
    """
    try:
        result = simulation_agent.run_simulation(session_id)

        # Save into Redis memory
        memory_store.save_entity(
            session_id,
            {"last_simulation": result}
        )

        return {
            "expected_value": result["expected_value"],
            "best_case": result["best_case"],
            "worst_case": result["worst_case"],
            "probability_of_goal_achievement": result["probability_of_goal_achievement"],
        }

    except Exception as ex:
        print("\n\n------------ SIMULATION ROUTER ERROR ------------")
        traceback.print_exc()
        print("--------------------------------------------------\n\n")
        raise HTTPException(status_code=500, detail=str(ex))