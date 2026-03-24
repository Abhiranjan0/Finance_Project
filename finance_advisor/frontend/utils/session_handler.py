# frontend/utils/session_handler.py

import streamlit as st
import uuid


def init_session() -> str:
    """
    Ensures that every visitor gets a unique session_id.
    This ID persists across:
      - Page refresh
      - Navigation between tabs
      - Multiple backend calls

    If already exists, it simply returns it.
    """

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    return st.session_state["session_id"]


def logout_session():
    """
    Clears all session data and resets to unauthenticated state.
    Called when user clicks logout button.
    """
    keys_to_remove = [
        "user_id",
        "user_email",
        "session_id",
        "chat_history",
        "risk_profile",
        "portfolio",
        "simulation_results",
        "nav_main"
    ]
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    return True


def is_authenticated() -> bool:
    """
    Check if user is currently logged in.
    """
    return "user_id" in st.session_state and st.session_state["user_id"] is not None
