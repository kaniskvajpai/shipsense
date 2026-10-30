"""
Live ETA tab (module 6 + module 7 direct-travel estimate).

Key changes:
  - Loads the new bundle from train_live_eta.py (p10/p50/p90 models
    + category maps) instead of a single raw model.
  - Casts categorical columns to the exact trained categories before
    predict (region_id etc. are categorical now, not numeric).
  - Shows an ETA window (p10-p90) with the p50 as the headline number.
  - Module 7: if data/processed/travel_stats.json exists (from
    src/build_travel_stats.py), also shows the DIRECT TRAVEL estimate
    (distance / regional median courier speed + median handover) next to
    the realistic route estimate, and explains why they differ (batching).
  - Distance input is capped at 2x the region's p90 distance (floor 10 km)
    instead of an arbitrary 100 km: the model was never trained on 100 km
    deliveries, so predictions there would be pure extrapolation.

"Live" still means computed on request, not live traffic or rider tracking.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "models/eta_live_model.pkl"
METRICS_PATH = "models/eta_live_metrics.json"
DEFAULTS_PATH = "data/processed/eta_defaults.json"
TRAVEL_STATS_PATH = "data/processed/travel_stats.json"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKEND_DAYS = (5, 6)

# Distance input bounds: never below the region's p90 distance, never
# above 2x it (floor 10 km). Keeps predictions inside the training data.
MIN_INPUT_KM = 0.1
P90_MULTIPLE = 2.0
MAX_INPUT_FLOOR_KM = 10.0


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


@st.cache_data
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def fmt_minutes(minutes):
    h, m = divmod(int(round(minutes)), 60)
    if h == 0:
        return "%d min" % m
    return "%d h %02d min" % (h, m)


def render_live_eta_tab():
    st.subheader("Live ETA")

    needed = [MODEL_PATH, METRICS_PATH, DEFAULTS_PATH]
    if not all(os.path.exists(p) for p in needed):
        st.error(
            "Live ETA files not found. Run `python src/train_live_eta.py` to "
            "generate `models/eta_live_model.pkl`, `models/eta_live_metrics.json` "
            "and `data/processed/eta_defaults.json`."
        )
        return

    bundle = load_bundle(MODEL_PATH)
    models = bundle["models"]
    cat_maps = bundle["cat_maps"]
    feature_order = bundle["feature_order"]

    metrics = load_json(METRICS_PATH)
    defaults = load_json(DEFAULTS_PATH)
    regions = defaults["regions"]
    threshold = defaults["threshold_minutes"]

    travel = None
    if os.path.exists(TRAVEL_STATS_PATH):
        try:
            travel = load_json(TRAVEL_STATS_PATH)
        except (json.JSONDecodeError, OSError):
            travel = None

    st.info(
        "Estimated on request from a compact gradient-boosting model trained on "
        "the same historical orders as the main ETA model. Because courier, live "
        "traffic, weather and rider position are unknown, we show a likely range "
        "(p10-p90), not a single promise. On %d past orders held out from "
        "training the median estimate was off by about %.0f minutes on average, "
        "and the true delivery time fell inside the shown range about %.0f%% of "
        "the time. Courier and delivery area default to the region's most "
        "common ones."
        % (metrics["test_rows"], metrics["mae_minutes"],
           100 * metrics["p10_p90_coverage"])
    )

    region_ids = sorted(regions.keys(), key=int)
    c1, c2, c3, c4 = st.columns(4)
    region = c1.selectbox(
        "Region",
        options=region_ids,
        format_func=lambda r: "Region %s" % r,
        key="live_eta_region",
    )
    info = regions[region]
    day = c2.selectbox(
        "Weekday",
        options=list(range(7)),
        format_func=lambda d: DAY_NAMES[d],
        key="live_eta_day",
    )
    hour = c3.selectbox(
        "Hour",
        options=list(range(24)),
        format_func=lambda h: "%02d:00" % h,
        key="live_eta_hour",
    )
    # Data-grounded distance bounds: cap at 2x the region's p90 distance
    # (floor 10 km) so predictions stay inside the training distribution.
    p90_km = max(float(info["p90_distance_km"]), MIN_INPUT_KM)
    max_km = round(max(p90_km * P90_MULTIPLE, MAX_INPUT_FLOOR_KM), 1)
    start_km = min(max(float(info["median_distance_km"]), MIN_INPUT_KM), max_km)
    distance = c4.number_input(
        "Distance (km)",
        min_value=MIN_INPUT_KM,
        max_value=max_km,
        value=start_km,
        step=0.5,
        key="live_eta_distance_%s" % region,
    )

    with st.expander("Advanced: courier and delivery area"):
        st.caption(
            "A visitor usually does not know these, so the region's most common "
            "courier and delivery area are used unless you pick others here."
        )
        courier = st.selectbox(
            "Courier",
            options=info["couriers"],
            format_func=lambda c: "Courier %d" % c,
            key="live_eta_courier_%s" % region,
        )
        aoi_pos = st.selectbox(
            "Delivery area",
            options=list(range(len(info["aois"]))),
            format_func=lambda i: "Area %d (type %d)"
            % (info["aois"][i][0], info["aois"][i][1]),
            key="live_eta_area_%s" % region,
        )
    aoi_id, aoi_type = info["aois"][aoi_pos]

    def make_rows(hours):
        n = len(hours)
        raw = pd.DataFrame(
            {
                "distance_km": [float(distance)] * n,
                "hour_of_day": list(hours),
                "day_of_week": [int(day)] * n,
                "region_id": [int(region)] * n,
                "courier_id": [int(courier)] * n,
                "aoi_id": [int(aoi_id)] * n,
                "aoi_type": [int(aoi_type)] * n,
                "is_weekend": [day in WEEKEND_DAYS] * n,
            }
        )[feature_order]
        # cast to the exact categories the model was trained with
        for c, cats in cat_maps.items():
            raw[c] = pd.Categorical(raw[c], categories=cats)
        return raw

    rows = make_rows([hour])
    eta50 = float(np.expm1(models["p50"].predict(rows)[0]))
    eta10 = float(np.expm1(models["p10"].predict(rows)[0]))
    eta90 = float(np.expm1(models["p90"].predict(rows)[0]))
    eta10 = min(eta10, eta50)
    eta90 = max(eta90, eta50)

    promised = float(info["promised_eta_minutes"])
    delay = eta50 - promised
    at_risk = delay > threshold

    # ---- Module 7: direct travel estimate (optional) ----
    direct_min = None
    speed_kmph = None
    if travel is not None:
        t_region = travel.get("regions", {}).get(str(region))
        t_global = travel.get("global", {})
        if t_region:
            speed_kmph = t_region.get("median_speed_kmph")
            handover = t_region.get("median_handover_min", 0.0)
        else:
            speed_kmph = t_global.get("median_speed_kmph")
            handover = t_global.get("median_handover_min", 0.0)
        if speed_kmph:
            direct_min = distance / speed_kmph * 60.0 + float(handover)

    if direct_min is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Direct travel estimate", fmt_minutes(direct_min))
        m2.metric("Realistic route ETA (median)", fmt_minutes(eta50))
        m3.metric(
            "Realistic range (p10-p90)",
            "%s - %s" % (fmt_minutes(eta10), fmt_minutes(eta90)),
        )
        m4.metric("Status", "At Risk" if at_risk else "On Track")
        st.caption(
            "Direct travel = distance divided by the median riding speed of "
            "couriers in this region, plus the typical handover stop - what "
            "the trip would take if your package went straight out. Realistic "
            "route ETA = what actually happened to similar orders: couriers "
            "carry a batch of packages and deliver them one by one, so yours "
            "waits while others on the route are dropped first. The gap "
            "between the two is the batching/queueing delay, which is real "
            "and dominates delivery time in this data."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated ETA (median)", fmt_minutes(eta50))
        m2.metric("Likely range", "%s - %s" % (fmt_minutes(eta10), fmt_minutes(eta90)))
        m3.metric("Delay vs promised", "%+.0f min" % delay)
        m4.metric("Status", "At Risk" if at_risk else "On Track")
        st.caption(
            "Direct travel stats not found - run `python src/build_travel_stats.py` "
            "to also show the direct travel estimate alongside the route ETA."
        )

    st.caption(
        "At Risk means the median route estimate is more than %d minutes above "
        "the region's promised ETA (the 75th percentile of its past delivery "
        "times), the same rule as the other tabs. Treat the numbers as rough "
        "estimates, not promises." % threshold
    )
    if distance > p90_km:
        st.warning(
            "This distance is longer than 90 percent of past orders in this "
            "region (p90: %.1f km), so the estimate is less reliable." % p90_km
        )

    hours = list(range(24))
    rows24 = make_rows(hours)
    curve50 = np.expm1(models["p50"].predict(rows24))
    curve10 = np.minimum(np.expm1(models["p10"].predict(rows24)), curve50)
    curve90 = np.maximum(np.expm1(models["p90"].predict(rows24)), curve50)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hours, y=curve90, mode="lines",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip", name="p90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours, y=curve10, mode="lines", fill="tonexty",
            fillcolor="rgba(0,118,255,0.18)", line=dict(width=0),
            name="p10-p90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hours, y=curve50, mode="lines+markers", name="Median ETA"
        )
    )
    if direct_min is not None:
        fig.add_hline(
            y=direct_min, line_dash="solid", line_color="green",
            annotation_text="Direct travel estimate",
        )
    fig.add_trace(
        go.Scatter(
            x=[hour], y=[eta50], mode="markers",
            marker=dict(size=14, color="red"), name="Selected hour",
        )
    )
    fig.add_hline(y=promised, line_dash="dash", annotation_text="Promised ETA")
    fig.add_hline(
        y=promised + threshold, line_dash="dot",
        annotation_text="At Risk above this line",
    )
    fig.update_layout(
        title="Route ETA by hour (same region, weekday, distance, courier and area)",
        xaxis_title="Hour of day",
        yaxis_title="ETA (minutes)",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig)

    with st.expander("Assumptions and limits"):
        st.markdown(
            "- Live means computed on request. The model sees none of the real "
            "conditions of a delivery: no live traffic, weather or rider "
            "position.\n"
            "- A compact gradient-boosting model (a few MB), saved with "
            "scikit-learn 1.5.2. It only loads on that version, so the version "
            "pinned for the app must not change without retraining.\n"
            "- Its inputs are the same eight the main model uses: distance, hour, "
            "weekday, region, courier, delivery area, area type and a weekend "
            "flag (Saturday and Sunday). Region and area type are treated as "
            "categories, so unrelated IDs are never averaged together.\n"
            "- Predictions are made in log space and converted back, and the "
            "p50 model is a true median (quantile loss), which directly "
            "minimizes average error.\n"
            "- The shown range is the 10th-90th percentile band of similar past "
            "orders; the median (p50) is the headline number.\n"
            "- The distance input is capped at twice the region's 90th-percentile "
            "delivery distance (at least 10 km) because the model has no "
            "training examples beyond that range.\n"
            "- The direct travel estimate comes from empirical courier speeds "
            "measured on consecutive drops in this city's data "
            "(src/build_travel_stats.py). It ignores batching by design.\n"
            "- Courier and delivery area default to the region's most common "
            "ones, which can shift the estimate.\n"
            "- The promised ETA is a proxy: the 75th percentile of the region's "
            "past delivery times, not a real promise.\n"
            "- The batch tabs use the larger model, so the two can differ for "
            "the same order."
        )