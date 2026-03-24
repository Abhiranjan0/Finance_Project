# frontend/app.py

import streamlit as st
from utils.api_client import APIClient
from components.chat_box import chat_interface
from components.risk_form import risk_profile_form

# MUST be the first Streamlit command:
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💹",
    layout="wide"
)

# Handle page redirection BEFORE rendering radio menu
if "pending_redirect" in st.session_state:
    redirect_target = st.session_state.pop("pending_redirect")
    st.session_state["nav_main"] = redirect_target
    st.rerun()

# Load styles
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS AFTER set_page_config but BEFORE any UI layout
try:
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        @keyframes fadeInPage {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main {
            animation: fadeInPage 0.6s ease-out;
        }
        </style>
        """, unsafe_allow_html=True)
except:
    pass

from utils.api_client import APIClient
from utils.session_handler import init_session
from components.chat_box import chat_interface
from components.risk_form import risk_profile_form
from components.portfolio_charts import show_portfolio_chart
from components.simulation_charts import show_simulation_results
from streamlit_lottie import st_lottie
import json
from utils.lottie_loaders import render_lottie

def load_lottie(path):
    with open(path) as f:
        return json.load(f)



# -------------------------------------------------------------
# Initialize Session
# -------------------------------------------------------------
session_id = init_session()
api = APIClient()
if "user_id" in st.session_state:
    # Create session_id only once per login
    if "session_id" not in st.session_state:
        import uuid
        new_session = str(uuid.uuid4())
        st.session_state["session_id"] = f"{st.session_state['user_id']}-{new_session}"

session_id = st.session_state.get("session_id")
short_id = session_id[:6].upper()

user_email = st.session_state.get("user_email", "Guest")


st.sidebar.markdown("## Finance Advisor")
st.sidebar.markdown(
    "<p style='font-size:12px; color:#6B7280;'>SEBI-aware AI assistant</p>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Login/Register","Chat Advisor", "Risk Profiling", "Portfolio", "Simulation", "Download Report"],
    label_visibility="collapsed", key="nav_main"
)

# ---- Apple-style header + shell ----
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="app-header-title">AI Financial Advisor</div>
            <div class="app-header-subtitle">Welcome, {user_email}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



# -------------------------------------------------------------
# PAGE LOGIC
# -------------------------------------------------------------
if page == "Chat Advisor":
    st.title("💬 AI Financial Advisor")
    if "user_id" not in st.session_state:
        st.warning("Please login to continue.")
        st.stop()
    chat_interface(api, session_id)

elif page == "Risk Profiling":
    if "user_id" not in st.session_state:
        st.warning("Please login to continue.")
        st.stop()
    from utils.lottie_loaders import render_lottie
    st.title("🧭 Risk Assessment")
    risk_profile_form(api, session_id)

elif page == "Portfolio":
    if "user_id" not in st.session_state:
        st.warning("Please login to continue.")
        st.stop()
    from utils.lottie_loaders import render_lottie
    render_lottie("assets/animations/portfolio_animation.json", height=220, key="portfolio_anim")

    st.markdown('<div class="apple-card fade-in">', unsafe_allow_html=True)

    st.markdown("<h2 class='apple-section-heading'>Recommended Portfolio</h2>", unsafe_allow_html=True)
    st.markdown("<p class='apple-section-subtitle'>Optimized asset allocation based on your risk profile.</p>", unsafe_allow_html=True)

    result = api.fetch_portfolio(session_id)
    if result:
        allocation = result["allocation"]
        explanation = result["explanation"]

        show_portfolio_chart(allocation)
        st.markdown(f"<p style='font-size:14px;'>{explanation}</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Simulation":
    if "user_id" not in st.session_state:
        st.warning("Please login to continue.")
        st.stop()
    from utils.lottie_loaders import render_lottie

    render_lottie("assets/animations/simulation_graph.json", height=220, key="simulation_anim")

    st.markdown('<div class="apple-card fade-in">', unsafe_allow_html=True)

    st.markdown("<h2 class='apple-section-heading'>Monte Carlo Simulation</h2>", unsafe_allow_html=True)
    st.markdown("<p class='apple-section-subtitle'>Projected outcomes based on randomized return paths.</p>", unsafe_allow_html=True)

    output = api.run_simulation(session_id)
    if output:
        show_simulation_results(output)

    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Download Report":
    if "user_id" not in st.session_state:
        st.warning("Please login to continue.")
        st.stop()
    st.title("📄 Download Your Financial Plan")

    pdf_bytes = api.download_report(session_id)
    if pdf_bytes:
        st.download_button(
            label="⬇️ Download Financial Plan PDF",
            data=pdf_bytes,
            file_name="financial_plan.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Generate your portfolio & simulation first.")

elif page == "Login/Register":
    st.markdown("<h2>Login / Register</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            result = api.login(login_email, login_password)
            if result:
                st.session_state["user_id"] = result["user_id"]
                st.session_state["user_email"] = login_email  
                st.success("Login successful!")
                st.session_state["pending_redirect"] = "Chat Advisor"
                st.rerun()


    with tab2:
        st.subheader("Register")
        reg_email = st.text_input("New Email", key="reg_email")
        reg_password = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            result = api.register(reg_email, reg_password)
            if result:
                st.success("Registration successful!")
                st.session_state["user_id"] = result.get("user_id", reg_email) 
                st.session_state["user_email"] = reg_email 
                st.session_state["pending_redirect"] = "Chat Advisor"
                st.rerun()
