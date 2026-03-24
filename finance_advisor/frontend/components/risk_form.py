# frontend/components/risk_form.py

import streamlit as st
from utils.lottie_loaders import render_lottie

def risk_profile_form(api, session_id: str):

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    render_lottie("assets/animations/risk_profile.json", height=200, key="risk_anim")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="apple-card">
            <div class="apple-section-heading">Risk Profiling</div>
            <div class="apple-section-subtitle">
                Help the advisor understand your capacity and willingness to take risk. 
                This drives asset allocation and simulation recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("risk_profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Your Age", min_value=18, max_value=85, value=30)
            liquidity_needs = st.selectbox("Liquidity Needs", ["Low", "Medium", "High"])

        with col2:
            income_stability = st.selectbox("Income Stability", ["Low", "Medium", "High"])
            investment_knowledge = st.selectbox("Investment Knowledge", ["Low", "Medium", "High"])

        st.markdown("#### Risk Tolerance")
        q1 = st.slider("Comfort with short-term losses", 1, 5, 3)
        q2 = st.slider("Investment horizon (longer = more risk)", 1, 5, 3)
        q3 = st.slider("Familiarity with equity markets", 1, 5, 3)
        q4 = st.slider("How you respond to financial uncertainty", 1, 5, 3)

        submitted = st.form_submit_button("Calculate My Risk Profile")

    if submitted:
        with st.spinner("Analyzing your risk profile..."):
            payload = {
                "session_id": session_id,
                "age": age,
                "income_stability": income_stability,
                "liquidity_needs": liquidity_needs,
                "investment_knowledge": investment_knowledge,
                "answers": {"q1": q1, "q2": q2, "q3": q3, "q4": q4},
            }
            result = api.send_risk_profile(payload)

        st.markdown(
            """
            <div class="apple-card fade-in">
            """,
            unsafe_allow_html=True,
        )

        st.success(f"Your Risk Category: {result['risk_category'].title()}")
        st.markdown(f"<p style='font-size:14px; color:#374151;'>{result['explanation']}</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
