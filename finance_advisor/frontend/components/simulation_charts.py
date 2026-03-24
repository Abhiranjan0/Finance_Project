# frontend/components/simulation_charts.py

import streamlit as st
import plotly.graph_objects as go
from utils.lottie_loaders import render_lottie

st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
render_lottie("assets/animations/simulation_graph.json", height=220, key="simulation_anim")
st.markdown("</div>", unsafe_allow_html=True)



def show_simulation_results(results: dict):
    """
    Visualizes Monte Carlo results.
    Expected structure:
    {
        "expected_value": float,
        "best_case": float,
        "worst_case": float,
        "probability_of_goal_achievement": float
    }
    """

    expected = results["expected_value"]
    best = results["best_case"]
    worst = results["worst_case"]
    prob = results["probability_of_goal_achievement"]

    # ----------------------------------------------------
    # Metrics Row
    # ----------------------------------------------------
    st.subheader("Simulation Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Expected Value",
        value=f"₹ {expected:,.0f}"
    )

    col2.metric(
        label="Best Case (95th percentile)",
        value=f"₹ {best:,.0f}"
    )

    col3.metric(
        label="Worst Case (5th percentile)",
        value=f"₹ {worst:,.0f}"
    )

    # Probability metric
    st.metric(
        label="Probability of Reaching Goal (Default ₹1 Cr)",
        value=f"{prob * 100:.2f} %"
    )

    # ----------------------------------------------------
    # Bar/Column Chart for Comparison
    # ----------------------------------------------------
    st.subheader("Simulation Outcome Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Worst Case", "Expected", "Best Case"],
        y=[worst, expected, best],
        text=[f"₹ {worst:,.0f}", f"₹ {expected:,.0f}", f"₹ {best:,.0f}"],
        textposition="outside",
        marker_color=["#EF553B", "#636EFA", "#00CC96"]
    ))

    fig.update_layout(
        title="Monte Carlo Simulation Results",
        yaxis_title="Portfolio Value (₹)",
        xaxis_title="Outcome Type",
        showlegend=False,
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # Interpretation Note
    # ----------------------------------------------------
    st.info(
        "These results are based on randomized simulations using assumed expected returns "
        "and volatility levels for each asset class. They are not guaranteed. "
        "Use them only as an estimate of potential future outcomes."
    )
