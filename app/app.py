import streamlit as st

st.set_page_config(page_title="ShipSense", layout="wide")

st.title("🚚 ShipSense")
st.caption("AI-powered delivery risk & ETA intelligence")

tab1, tab2, tab3 = st.tabs(["Operations Dashboard", "Demand Heatmap", "Executive Summary"])

with tab1:
    st.info("Operations Dashboard — built on Day 6 (reads data/processed/predictions.csv)")

with tab2:
    st.info("Demand Heatmap — built on Day 7")

with tab3:
    st.info("Executive Summary — built on Day 7")