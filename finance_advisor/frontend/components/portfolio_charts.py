# frontend/components/portfolio_charts.py

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.lottie_loaders import render_lottie

st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
render_lottie("assets/animations/portfolio_animation.json", height=220, key="portfolio_anim")
st.markdown("</div>", unsafe_allow_html=True)

def show_portfolio_chart(allocation: dict):
    """
    Displays a pie chart of the portfolio allocation.
    allocation: Dict[str, float]
    """

    st.subheader("Asset Allocation Breakdown")

    # Convert dict to dataframe
    df = pd.DataFrame(
        {
            "Asset Class": list(allocation.keys()),
            "Allocation (%)": list(allocation.values())
        }
    )

    # ------------------------------
    # Pie Chart
    # ------------------------------
    fig = px.pie(
        df,
        names="Asset Class",
        values="Allocation (%)",
        title="Portfolio Allocation (%)",
        hole=0.35,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Optional: Show table below
    # ------------------------------
    st.write("### Allocation Table")
    st.dataframe(df, use_container_width=True)
