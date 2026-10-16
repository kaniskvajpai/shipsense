"""
Rider Allocation tab (post-v1.0, module 4).

Reads the files made by src/optimize_allocation.py.
No live solving happens in the app.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ORDERS_PATH = "data/processed/allocation.csv"
RIDERS_PATH = "data/processed/allocation_riders.csv"
SUMMARY_PATH = "data/processed/allocation_summary.csv"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
RIDER_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#17becf"]


@st.cache_data
def load_allocation(orders_path: str, riders_path: str, summary_path: str):
    orders = pd.read_csv(orders_path)
    riders = pd.read_csv(riders_path)
    summary = pd.read_csv(summary_path)
    return orders, riders, summary


def allocation_map(batch_orders, batch_riders, rider_col, title):
    rider_ids = sorted(batch_riders["rider_id"].tolist())
    color_of = {
        rid: RIDER_COLORS[i % len(RIDER_COLORS)] for i, rid in enumerate(rider_ids)
    }
    fig = go.Figure()
    for rid in rider_ids:
        pts = batch_orders[batch_orders[rider_col] == rid]
        fig.add_trace(
            go.Scatter(
                x=pts["lng"],
                y=pts["lat"],
                mode="markers",
                marker=dict(size=9, color=color_of[rid]),
                name="Rider %d (%d orders)" % (rid, len(pts)),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=batch_riders["lng"],
            y=batch_riders["lat"],
            mode="markers+text",
            marker=dict(
                size=15,
                color=[color_of[r] for r in batch_riders["rider_id"]],
                symbol="star",
                line=dict(color="black", width=1),
            ),
            text=[str(r) for r in batch_riders["rider_id"]],
            textposition="top center",
            textfont=dict(size=9),
            name="Rider location (proxy)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def render_allocation_tab():
    st.subheader("Rider Allocation (simulation)")
    st.info(
        "Simulation, not live dispatch. Each batch is up to 60 orders and up to "
        "6 real riders from one region, weekday and hour (the dataset has no "
        "dates). Distances are straight-line, not road distances. A rider's "
        "location is the median accept location of the orders they delivered in "
        "the batch, a proxy rather than live GPS. The baseline is the actual "
        "historical assignment. The optimized assignment minimizes distance, but "
        "no rider may get more than 1.25 times the average number of orders."
    )

    needed = [ORDERS_PATH, RIDERS_PATH, SUMMARY_PATH]
    if not all(os.path.exists(p) for p in needed):
        st.error(
            "Allocation data not found. Run `python src/optimize_allocation.py` "
            "(in the venv_routing environment) to generate the three "
            "`data/processed/allocation*.csv` files."
        )
        return

    orders, riders, summary = load_allocation(ORDERS_PATH, RIDERS_PATH, SUMMARY_PATH)

    shorter = int((summary["optimized_km"] < summary["actual_km"]).sum())
    med_saving = summary["saved_pct_vs_actual"].median()
    avg_saving = summary["saved_pct_vs_actual"].mean()
    act_load = summary["actual_max_over_avg"].median()
    opt_load = summary["optimized_max_over_avg"].median()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches solved", "%d" % len(summary))
    c2.metric("Median distance saving vs actual", "%.1f%%" % med_saving)
    c3.metric("Busiest rider vs average workload", "%.2fx to %.2fx" % (act_load, opt_load))
    c4.metric("Optimized shorter than actual", "%d of %d" % (shorter, len(summary)))
    st.caption(
        "Median distance saving %.1f%% (average %.1f%%, pulled up by a minority "
        "of batches). Under this proxy the real assignment is already close to "
        "distance-efficient, so the main effect of optimizing is balancing "
        "workload: the busiest rider carries about %.2fx the average load in the "
        "real data and %.2fx after optimization. Rider locations are inferred from "
        "the orders they actually delivered, which favours the actual assignment, "
        "so the savings shown here are conservative."
        % (med_saving, avg_saving, act_load, opt_load)
    )

    labels = {}
    for r in summary.itertuples(index=False):
        labels[r.batch_id] = "%s | region %d | %s %02d:00 | %d orders, %d riders" % (
            r.batch_id,
            r.region_id,
            DAY_NAMES[int(r.day_of_week)],
            r.hour_of_day,
            r.n_orders,
            r.n_riders,
        )

    st.markdown("#### Explore one batch")
    batch_id = st.selectbox(
        "Choose a batch",
        options=summary["batch_id"].tolist(),
        format_func=lambda b: labels[b],
    )
    row = summary[summary["batch_id"] == batch_id].iloc[0]
    b_orders = orders[orders["batch_id"] == batch_id]
    b_riders = riders[riders["batch_id"] == batch_id]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual assignment", "%.1f km" % row["actual_km"])
    m2.metric(
        "Optimized (workload-capped)",
        "%.1f km" % row["optimized_km"],
        delta="%.1f km vs actual" % (row["optimized_km"] - row["actual_km"]),
        delta_color="inverse",
    )
    m3.metric(
        "Busiest rider vs average",
        "%.2fx to %.2fx" % (row["actual_max_over_avg"], row["optimized_max_over_avg"]),
    )
    m4.metric("Orders moved to another rider", "%.0f%%" % row["reassigned_pct"])
    st.caption(
        "If every order went to its nearest rider, ignoring workload, the total "
        "would be %.1f km (a lower bound, not a realistic plan). The cap in this "
        "batch is %d orders per rider." % (row["nearest_rider_km"], row["cap_per_rider"])
    )

    load = b_riders.melt(
        id_vars="rider_id",
        value_vars=["actual_orders", "optimized_orders"],
        var_name="assignment",
        value_name="orders",
    )
    load["assignment"] = load["assignment"].map(
        {"actual_orders": "Actual", "optimized_orders": "Optimized"}
    )
    load["rider"] = "Rider " + load["rider_id"].astype(str)
    bar = px.bar(
        load,
        x="rider",
        y="orders",
        color="assignment",
        barmode="group",
        title="Orders per rider",
    )
    bar.update_layout(height=320, xaxis_title="", yaxis_title="Orders")
    st.plotly_chart(bar)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            allocation_map(b_orders, b_riders, "actual_rider", "Actual assignment")
        )
    with right:
        st.plotly_chart(
            allocation_map(
                b_orders, b_riders, "optimized_rider", "Optimized assignment"
            )
        )

    with st.expander("Orders that change rider in this batch"):
        moved = b_orders[b_orders["actual_rider"] != b_orders["optimized_rider"]]
        st.dataframe(
            moved[
                [
                    "order_id",
                    "actual_rider",
                    "optimized_rider",
                    "actual_km",
                    "optimized_km",
                ]
            ]
        )

    st.markdown("#### Saving across all batches")
    hist = px.histogram(
        summary,
        x="saved_pct_vs_actual",
        nbins=20,
        labels={"saved_pct_vs_actual": "Distance saving vs actual (%)"},
        title="Distribution of distance savings vs the actual assignment",
    )
    hist.update_layout(height=350, yaxis_title="Number of batches")
    st.plotly_chart(hist)
    st.caption(
        "Bars at or below zero are batches where the workload cap makes the "
        "optimized assignment no shorter than the actual one."
    )

    with st.expander("Assumptions and limits"):
        st.markdown(
            "- Straight-line (haversine) distances, not road distances.\n"
            "- A batch is up to 60 orders and up to 6 real riders (the busiest "
            "ones) from one region, weekday and hour. The dataset has no dates, so "
            "this is not a real historical day.\n"
            "- A rider's location is the median accept location of their orders in "
            "the batch. This is a proxy, not live GPS, and it favours the actual "
            "assignment.\n"
            "- Optimized = the assignment with the smallest total distance where no "
            "rider gets more than 1.25 times the average workload (my choice, not "
            "from the data). Moving an order away from its actual rider carries a "
            "100 m penalty, so only moves that matter are counted.\n"
            "- Hidden factors such as parcel size, shifts and delivery zones are not "
            "in the data and may explain parts of the actual assignment.\n"
            "- Stops more than 30 km from the batch centre are dropped as likely "
            "GPS errors.\n"
            "- Assignments are computed offline by `src/optimize_allocation.py` "
            "with OR-Tools and saved to CSV. The app does not solve anything live."
        )

    with st.expander("All batch results"):
        st.dataframe(summary)