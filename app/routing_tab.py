"""
Vehicle Routing tab (post-v1.0, module 3b).

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
VEHICLE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


@st.cache_data
def load_routing(routes_path: str, summary_path: str):
    routes = pd.read_csv(routes_path)
    summary = pd.read_csv(summary_path)
    return routes, summary


def route_figure(batch_routes: pd.DataFrame, method: str, title: str):
    data = batch_routes[batch_routes["method"] == method]
    fig = go.Figure()

    for vehicle, grp in data.groupby("vehicle"):
        grp = grp.sort_values("seq")
        color = VEHICLE_COLORS[(int(vehicle) - 1) % len(VEHICLE_COLORS)]
        lats = grp["lat"].tolist()
        lngs = grp["lng"].tolist()
        # close the loop: the vehicle returns to the depot
        lats.append(lats[0])
        lngs.append(lngs[0])
        fig.add_trace(
            go.Scatter(
                x=lngs,
                y=lats,
                mode="lines",
                line=dict(color=color, width=2),
                name="Vehicle %d" % int(vehicle),
                hoverinfo="skip",
            )
        )
        stops = grp[grp["stop_type"] == "stop"]
        hover = [
            "Stop %d | arrives %.0f min | deadline %.0f min"
            % (s, a, d)
            for s, a, d in zip(
                stops["seq"], stops["arrival_min"], stops["deadline_min"]
            )
        ]
        fig.add_trace(
            go.Scatter(
                x=stops["lng"],
                y=stops["lat"],
                mode="markers+text",
                text=stops["seq"].astype(int).astype(str),
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=8, color=color),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )

    late = data[(data["stop_type"] == "stop") & (data["late_min"] > 0)]
    if not late.empty:
        fig.add_trace(
            go.Scatter(
                x=late["lng"],
                y=late["lat"],
                mode="markers",
                marker=dict(
                    size=18,
                    color="rgba(0,0,0,0)",
                    line=dict(color="red", width=3),
                ),
                name="Late stop (stress deadline)",
                hoverinfo="skip",
            )
        )

    depot = data[data["stop_type"] == "depot"].drop_duplicates(subset=["lat", "lng"])
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
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def render_routing_tab():
    st.subheader("Vehicle Routing (simulation)")
    st.info(
        "Simulation, not live routing. Distances are straight-line (haversine), "
        "not road distances. Each batch is up to 30 orders from one region, "
        "weekday and hour (the dataset has no dates), split across vehicles "
        "with at most 10 stops each. The depot is the median accept location of "
        "the batch's orders. Savings are measured against a nearest-neighbor "
        "baseline. Deadlines are a STRESS TEST (30% of each order's promised "
        "time), not real promises."
    )

    if not (os.path.exists(ROUTES_PATH) and os.path.exists(SUMMARY_PATH)):
        st.error(
            "Routing data not found. Run `python src/optimize_routes.py` "
            "(in the venv_routing environment) to generate "
            "`data/processed/routes.csv` and `data/processed/routes_summary.csv`."
        )
        return

    routes, summary = load_routing(ROUTES_PATH, SUMMARY_PATH)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches solved", "%d" % len(summary))
    c2.metric(
        "Median distance saving vs nearest-neighbor",
        "%.1f%%" % summary["saved_pct_vs_nn"].median(),
    )
    c3.metric(
        "Late stops, stress test (NN to optimized)",
        "%d to %d"
        % (summary["nn_late_stress"].sum(), summary["opt_late_stress"].sum()),
    )
    c4.metric(
        "Late stops, real promised times",
        "%d to %d" % (summary["nn_late_real"].sum(), summary["opt_late_real"].sum()),
    )
    st.caption(
        "Out of %d stops in total. Real promised times are about 2.5 to 8 hours, "
        "so almost no stop is late under them. The stress test (30%% of the "
        "promised time) is only there to give the solver something to trade "
        "against distance." % int(summary["n_stops"].sum())
    )

    labels = {}
    for r in summary.itertuples(index=False):
        labels[r.batch_id] = "%s | region %d | %s %02d:00 | %d stops, %d vehicles" % (
            r.batch_id,
            r.region_id,
            DAY_NAMES[int(r.day_of_week)],
            r.hour_of_day,
            r.n_stops,
            r.n_vehicles,
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
    m1.metric("Nearest-neighbor", "%.1f km" % row["nn_km"])
    m2.metric(
        "Optimized",
        "%.1f km" % row["optimized_km"],
        delta="%.1f km vs nearest-neighbor" % (row["optimized_km"] - row["nn_km"]),
        delta_color="inverse",
    )
    m3.metric("Distance saving", "%.1f%%" % row["saved_pct_vs_nn"])
    m4.metric(
        "Late stops, stress test",
        "%d to %d" % (row["nn_late_stress"], row["opt_late_stress"]),
    )
    st.caption(
        "Late stops under the real promised times: nearest-neighbor %d, "
        "optimized %d. Red circles on the maps mark stops that miss the "
        "stress deadline." % (row["nn_late_real"], row["opt_late_real"])
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            route_figure(batch_routes, "nn", "Nearest-neighbor routes (baseline)")
        )
    with right:
        st.plotly_chart(route_figure(batch_routes, "optimized", "Optimized routes"))

    with st.expander("Stop-by-stop schedule (optimized)"):
        sched = batch_routes[
            (batch_routes["method"] == "optimized")
            & (batch_routes["stop_type"] == "stop")
        ].copy()
        sched["order_id"] = sched["order_id"].astype("int64")
        st.dataframe(
            sched[
                ["vehicle", "seq", "order_id", "arrival_min", "deadline_min", "late_min"]
            ].sort_values(["vehicle", "seq"])
        )

    st.markdown("#### Saving across all batches")
    hist = px.histogram(
        summary,
        x="saved_pct_vs_nn",
        nbins=20,
        labels={"saved_pct_vs_nn": "Distance saving vs nearest-neighbor (%)"},
        title="Distribution of distance savings vs nearest-neighbor",
    )
    hist.update_layout(height=350, yaxis_title="Number of batches")
    st.plotly_chart(hist)

    with st.expander("Assumptions and limits"):
        st.markdown(
            "- Straight-line (haversine) distances, not road distances.\n"
            "- Each batch is up to 30 orders from one region, weekday and hour. "
            "The dataset has no dates, so this is not a real historical day.\n"
            "- Vehicles are anonymous, at most 10 stops each (my choice, not from "
            "the data). All leave the depot at the start of the hour.\n"
            "- Depot = median accept location of the batch's orders. Stops more "
            "than 30 km from it are dropped as likely GPS errors.\n"
            "- Travel at 20 km/h plus 3 minutes per stop (both assumptions).\n"
            "- Stress-test deadline = 30% of the order's promised time. This is a "
            "test scenario, not a real promise. Deadlines are soft: the solver "
            "pays a penalty for late stops instead of forbidding them.\n"
            "- Routes are computed offline by `src/optimize_routes.py` with "
            "OR-Tools and saved to CSV. The app does not solve anything live."
        )

    with st.expander("All batch results"):
        st.dataframe(summary)