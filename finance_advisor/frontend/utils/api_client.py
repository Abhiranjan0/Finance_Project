# frontend/utils/api_client.py

import requests
from typing import Any, Dict, Optional
import streamlit as st


class APIClient:
    """
    Wrapper for FastAPI finance_advisor.backend.
    All frontend API calls go through this client.
    """

    def __init__(self):
        # Read backend URL from Streamlit secrets or fallback
        self.base_url = st.secrets.get("BACKEND_URL", "http://localhost:8000")

    # ------------------------------------------------------------
    # Helper method for making HTTP requests
    # ------------------------------------------------------------
    def _post(self, path, payload):
        try:
            url = f"{self.base_url}{path}"
            resp = requests.post(url, json=payload)
            
            if resp.status_code != 200:
                st.error(f"API Error {resp.status_code}: {resp.text}")
                return None
            
            return resp.json()
        except Exception as ex:
            st.error(f"API Exception: {ex}")
            return None


    def _get(self, endpoint: str) -> Optional[Any]:
        url = f"{self.base_url}{endpoint}"

        try:
            resp = requests.get(url)
            resp.raise_for_status()

            # If it's a PDF or bytes, return raw content
            if resp.headers.get("content-type") == "application/pdf":
                return resp.content

            return resp.json()

        except Exception as ex:
            st.error(f"API Error: {ex}")
            return None

    # ------------------------------------------------------------
    # Chat Endpoint
    # ------------------------------------------------------------
    def send_chat_message(self, session_id: str, message: str) -> Dict:
        payload = {
            "session_id": session_id,
            "message": message,
            "metadata": {"channel": "streamlit"}
        }
        return self._post("/chat", payload)

    # ------------------------------------------------------------
    # Risk Profiling Endpoint
    # ------------------------------------------------------------
    def send_risk_profile(self, payload: Dict[str, Any]) -> Dict:
        return self._post("/risk_profile", payload)

    # ------------------------------------------------------------
    # Portfolio (via AI Agent)
    # ------------------------------------------------------------
    def fetch_portfolio(self, session_id: str) -> Optional[Dict]:
        """
        Calls a backend route to activate the PortfolioAgent.
        We need to add /portfolio endpoint to finance_advisor.backend.
        """
        return self._get(f"/portfolio?session_id={session_id}")

    # ------------------------------------------------------------
    # Simulation Endpoint
    # ------------------------------------------------------------
    def run_simulation(self, session_id: str) -> Optional[Dict]:
        """
        Calls backend to run simulation using SimulationAgent.
        """
        return self._get(f"/simulate?session_id={session_id}")

    # ------------------------------------------------------------
    # Full simulation route with variable inputs
    # ------------------------------------------------------------
    def simulate_portfolio(self, session_id: str, allocation: Dict[str, float], investment_type: str,
                            monthly_amount: float, lumpsum_amount: float, duration_years: int,
                            num_simulations: int = 5000) -> Optional[Dict]:
        payload = {
            "session_id": session_id,
            "allocation": allocation,
            "investment": {
                "type": investment_type,
                "monthly_amount": monthly_amount if investment_type == "sip" else None,
                "lumpsum_amount": lumpsum_amount if investment_type in ["lumpsum", "stocks", "portfolio"] else None,
                "duration_years": duration_years,
            },
            "simulation_params": {"num_simulations": num_simulations},
        }
        return self._post("/simulate_portfolio", payload)

    # ------------------------------------------------------------
    # Download Report
    # ------------------------------------------------------------
    def download_report(self, session_id: str) -> Optional[bytes]:
        return self._get(f"/download_plan?session_id={session_id}")
    

    
    # ------------------------------------------------------------
    # Conversation endpoint
    # ------------------------------------------------------------
    def get_conversation(self, session_id: str):
        return self._get(f"/conversation/{session_id}")
    

    # ------------------------------------------------------------
    # Auth endpoint
    # ------------------------------------------------------------
    def login(self, email, password):
        return self._post("/auth/login", {"email": email, "password": password})

    def register(self, email, password):
        return self._post("/auth/register", {"email": email, "password": password})

    def logout(self):
        return self._post("/auth/logout", {})
