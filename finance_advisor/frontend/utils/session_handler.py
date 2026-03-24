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
