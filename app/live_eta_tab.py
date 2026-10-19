"""
Live ETA tab (post-v1.0, module 6).

The only tab that runs a model on request: a compact gradient-boosting ETA model
(about 1 MB) saved by src/train_live_eta.py. It uses the same promised-ETA and
At Risk rule as src/predict.py. "Live" means computed on request, not live
traffic or rider tracking.
"""
import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "models/eta_live_model.pkl"
METRICS_PATH = "models/eta_live_metrics.json"
DEFAULTS_PATH = "data/processed/eta_defaults.json"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKEND_DAYS = (5, 6)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def fmt_minutes(minutes):
    h, m = divmod(int(round(minutes)), 60)
    return "%d h %02d min" % (h, m)


def render_live_eta_tab():
    st.subheader("Live ETA (compact model)")

    needed = [MODEL_PATH, METRICS_PATH, DEFAULTS_PATH]
    if not all(os.path.exists(p) for p in needed):
        st.error(
            "Live ETA files not found. Run `python src/train_live_eta.py` to "
            "generate `models/eta_live_model.pkl`, `models/eta_live_metrics.json` "
            "and `data/processed/eta_defaults.json`."
        )
        return

    metrics = load_json(METRICS_PATH)
    defaults = load_json(DEFAULTS_PATH)
    regions = defaults["regions"]
    threshold = defaults["threshold_minutes"]
    feature_order = metrics["feature_order"]
    model = load_model(MODEL_PATH)

    st.info(
        "Live here means computed on request, not live traffic or rider "
        "tracking. This is the only tab where the app runs a model: a compact "
        "gradient-boosting model (about 1 MB) trained on the same historical "
        "orders as the main ETA model. On %d past orders held out from training "
        "its average error was %.1f minutes (the full batch model: 70.9), and it "
        "gives the same At Risk label as the batch model for about 96 percent of "
        "orders. The courier and delivery area are usually unknown, so by "
        "default the region's most common ones are used."
        % (metrics["test_rows"], metrics["mae_minutes"])
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
    start_km = min(max(float(info["median_distance_km"]), 0.1), 100.0)
    distance = c4.number_input(
        "Distance (km)",
        min_value=0.1,
        max_value=100.0,
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
        return pd.DataFrame(
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

    eta = float(model.predict(make_rows([hour]))[0])
    promised = float(info["promised_eta_minutes"])
    delay = eta - promised
    at_risk = delay > threshold

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted ETA", fmt_minutes(eta))
    m2.metric("Promised ETA (region)", fmt_minutes(promised))
    m3.metric("Delay vs promised", "%+.0f min" % delay)
    m4.metric("Status", "At Risk" if at_risk else "On Track")
    st.caption(
        "At Risk means the predicted ETA is more than %d minutes above the "
        "region's promised ETA (the 75th percentile of its past delivery "
        "times), the same rule as the other tabs. On past orders this model is "
        "off by about 72 minutes on average, so treat the number as a rough "
        "estimate, not a promise." % threshold
    )
    if distance > float(info["p90_distance_km"]):
        st.warning(
            "This distance is longer than 90 percent of past orders in this "
            "region, so the estimate is less reliable."
        )

    hours = list(range(24))
    curve = model.predict(make_rows(hours))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=hours, y=curve, mode="lines+markers", name="Predicted ETA")
    )
    fig.add_trace(
        go.Scatter(
            x=[hour],
            y=[eta],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="Selected hour",
        )
    )
    fig.add_hline(y=promised, line_dash="dash", annotation_text="Promised ETA")
    fig.add_hline(
        y=promised + threshold,
        line_dash="dot",
        annotation_text="At Risk above this line",
    )
    fig.update_layout(
        title="Predicted ETA by hour (same region, weekday, distance, courier and area)",
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
            "- A compact gradient-boosting model (about 1 MB), saved with "
            "scikit-learn 1.5.2. It only loads on that version, so the version "
            "pinned for the app must not change without retraining.\n"
            "- Its inputs are the same eight the main model uses: distance, hour, "
            "weekday, region, courier, delivery area, area type and a weekend "
            "flag (Saturday and Sunday).\n"
            "- Courier and delivery area default to the region's most common "
            "ones, which can shift the estimate.\n"
            "- The promised ETA is a proxy: the 75th percentile of the region's "
            "past delivery times, not a real promise.\n"
            "- The batch tabs use the larger model, so the two can differ by "
            "about 20 minutes on average for the same order."
        )