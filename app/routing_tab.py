"""
Vehicle Routing tab (post-v1.0, module 3a).

Reads the files made by src/optimize_routes.py.
No live solving happens in the app.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROUTES_PATH = "data/processed/routes.csv"
SUMMARY_PATH = "data/processed/routes_summary.csv"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_routing(routes_path: str, summary_path: str):
    routes = pd.read_csv(routes_path)
    summary = pd.read_csv(summary_path)
    return routes, summary


def route_figure(batch_routes: pd.DataFrame, seq_col: str, title: str, color: str):
    pts = batch_routes.sort_values(seq_col)
    lats = pts["lat"].tolist()
    lngs = pts["lng"].tolist()
    # close the loop: the courier returns to the depot
    lats.append(lats[0])
    lngs.append(lngs[0])

    stops = pts[pts["stop_type"] == "stop"]
    depot = pts[pts["stop_type"] == "depot"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lngs,
            y=lats,
            mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stops["lng"],
            y=stops["lat"],
            mode="markers+text",
            text=stops[seq_col].astype(int).astype(str),
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=8, color=color),
            name="Stops (numbered in visit order)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=depot["lng"],
            y=depot["lat"],
            mode="markers",
            marker=dict(size=15, color="black", symbol="star"),
            name="Depot",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def render_routing_tab():
    st.subheader("Vehicle Routing (simulation)")
    st.info(
        "Simulation, not live routing. Distances are straight-line (haversine), "
        "not road distances. Each batch is a sample of one courier's orders from "
        "one weekday and hour, because the dataset has no dates. The depot is the "
        "courier's median accept location. Savings are measured against a "
        "nearest-neighbor route (realistic) and a random order (upper bound only)."
    )

    if not (os.path.exists(ROUTES_PATH) and os.path.exists(SUMMARY_PATH)):
        st.error(
            "Routing data not found. Run `python src/optimize_routes.py` "
            "(in the venv_routing environment) to generate "
            "`data/processed/routes.csv` and `data/processed/routes_summary.csv`."
        )
        return

    routes, summary = load_routing(ROUTES_PATH, SUMMARY_PATH)

    c1, c2, c3 = st.columns(3)
    c1.metric("Batches solved", "%d" % len(summary))
    c2.metric(
        "Median saving vs nearest-neighbor",
        "%.1f%%" % summary["saved_pct_vs_nn"].median(),
    )
    c3.metric(
        "Median saving vs random order (upper bound)",
        "%.1f%%" % summary["saved_pct_vs_random"].median(),
    )
    st.caption(
        "The nearest-neighbor figure is the realistic one. The random-order "
        "figure is shown only as an upper bound, because no courier drives in "
        "random order."
    )

    labels = {}
    for r in summary.itertuples(index=False):
        labels[r.batch_id] = "%s | courier %d | %s %02d:00 | %d stops" % (
            r.batch_id,
            r.courier_id,
            DAY_NAMES[int(r.day_of_week)],
            r.hour_of_day,
            r.n_stops,
        )

    st.markdown("#### Explore one batch")
    batch_id = st.selectbox(
        "Choose a batch",
        options=summary["batch_id"].tolist(),
        format_func=lambda b: labels[b],
    )
    row = summary[summary["batch_id"] == batch_id].iloc[0]
    batch_routes = routes[routes["batch_id"] == batch_id]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Random order", "%.1f km" % row["random_km"])
    m2.metric("Nearest-neighbor", "%.1f km" % row["nn_km"])
    m3.metric(
        "Optimized",
        "%.1f km" % row["optimized_km"],
        delta="%.1f km vs nearest-neighbor" % (row["optimized_km"] - row["nn_km"]),
        delta_color="inverse",
    )
    m4.metric("Saving vs nearest-neighbor", "%.1f%%" % row["saved_pct_vs_nn"])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            route_figure(
                batch_routes, "nn_seq", "Nearest-neighbor route (baseline)", "#d62728"
            )
        )
    with right:
        st.plotly_chart(
            route_figure(
                batch_routes, "optimized_seq", "Optimized route", "#2ca02c"
            )
        )

    st.markdown("#### Saving across all batches")
    hist = px.histogram(
        summary,
        x="saved_pct_vs_nn",
        nbins=20,
        labels={"saved_pct_vs_nn": "Saving vs nearest-neighbor (%)"},
        title="Distribution of savings vs nearest-neighbor",
    )
    hist.update_layout(height=350, yaxis_title="Number of batches")
    st.plotly_chart(hist)

    with st.expander("Assumptions and limits"):
        st.markdown(
            "- Straight-line (haversine) distances, not road distances.\n"
            "- One vehicle per batch, no capacity limit, no time windows "
            "(planned for the next step).\n"
            "- A batch is up to 30 of one courier's orders from one weekday and "
            "hour. The dataset has no dates, so this is not a real historical day.\n"
            "- Depot = median accept location of that courier. Stops more than "
            "30 km from the depot are dropped as likely GPS errors.\n"
            "- Routes are computed offline by `src/optimize_routes.py` with "
            "OR-Tools and saved to CSV. The app does not solve anything live."
        )

    with st.expander("All batch results"):
        st.dataframe(summary)