import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="ShipSense", layout="wide", page_icon="🚚")

PREDICTIONS_PATH = "data/processed/predictions.csv"


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    """Loads the scored predictions CSV once per session (cached)."""
    df = pd.read_csv(path)
    return df


def apply_filters(df: pd.DataFrame, region, risk, hour_range) -> pd.DataFrame:
    """Applies sidebar filters to the predictions DataFrame."""
    filtered = df.copy()

    if region != "All":
        filtered = filtered[filtered["region_id"] == region]

    if risk != "All":
        filtered = filtered[filtered["risk_status"] == risk]

    filtered = filtered[
        (filtered["hour_of_day"] >= hour_range[0])
        & (filtered["hour_of_day"] <= hour_range[1])
    ]

    return filtered


# ---------- Header ----------
st.title("🚚 ShipSense")
st.caption("AI-powered delivery risk & ETA intelligence")

tab1, tab2, tab3 = st.tabs(["Operations Dashboard", "Demand Heatmap", "Executive Summary"])

# ================= TAB 1: OPERATIONS DASHBOARD =================
with tab1:
    if not os.path.exists(PREDICTIONS_PATH):
        st.error(
            "Predictions data not found. Run `python src/predict.py` first "
            "to generate `data/processed/predictions.csv`."
        )
    else:
        df = load_predictions(PREDICTIONS_PATH)

        # --- Filters ---
        st.subheader("Filters")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            region_options = ["All"] + sorted(df["region_id"].unique().tolist())
            selected_region = st.selectbox("Area / Region", region_options)

        with col2:
            selected_risk = st.selectbox("Risk Status", ["All", "At Risk", "On Track"])

        with col3:
            selected_hours = st.slider("Hour of Day", 0, 23, (0, 23))

        filtered_df = apply_filters(df, selected_region, selected_risk, selected_hours)

        st.divider()

        # --- Summary strip ---
        total = len(filtered_df)
        if total == 0:
            st.warning("No deliveries match the current filters. Try widening your selection.")
        else:
            at_risk_count = (filtered_df["risk_status"] == "At Risk").sum()
            on_track_count = total - at_risk_count
            at_risk_pct = (at_risk_count / total) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Orders", f"{total:,}")
            m2.metric("🔴 At Risk", f"{at_risk_count:,}", f"{at_risk_pct:.1f}%")
            m3.metric("🟢 On Track", f"{on_track_count:,}", f"{100 - at_risk_pct:.1f}%")

            st.divider()

            # --- Table ---
            st.subheader("Deliveries")

            display_df = filtered_df.copy()
            display_df["Risk"] = display_df["risk_status"].map(
                {"At Risk": "🔴 At Risk", "On Track": "🟢 On Track"}
            )
            display_df["Predicted ETA (min)"] = display_df["predicted_eta_minutes"].round(1)
            display_df["Promised ETA (min)"] = display_df["promised_eta_minutes"].round(1)
            display_df["Delay (min)"] = display_df["delay_minutes"].round(1)
            display_df["Top Delay Factor"] = display_df["top_delay_factor"].fillna("—")

            show_cols = [
                "order_id", "region_id", "hour_of_day",
                "Predicted ETA (min)", "Promised ETA (min)", "Delay (min)",
                "Risk", "Top Delay Factor",
            ]
            display_df = display_df.rename(columns={
                "order_id": "Order ID", "region_id": "Region", "hour_of_day": "Hour",
            })
            show_cols_renamed = [
                "Order ID", "Region", "Hour",
                "Predicted ETA (min)", "Promised ETA (min)", "Delay (min)",
                "Risk", "Top Delay Factor",
            ]

            # Sort by highest delay first so the most urgent orders surface immediately
            display_df = display_df.sort_values("Delay (min)", ascending=False)

            st.dataframe(
                display_df[show_cols_renamed].head(500),
                use_container_width=True,
                height=450,
            )
            if total > 500:
                st.caption(f"Showing top 500 of {total:,} matching deliveries (sorted by delay).")

# ================= TAB 2: DEMAND HEATMAP (placeholder, built next) =================
with tab2:
    st.info("Demand Heatmap — coming next.")

# ================= TAB 3: EXECUTIVE SUMMARY (placeholder, built next) =================
with tab3:
    st.info("Executive Summary — coming next.")

# ---------- Footer ----------
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Built with Claude as part of the AB Talks 60-Day Claude AI Challenge."
    "</div>",
    unsafe_allow_html=True,
)