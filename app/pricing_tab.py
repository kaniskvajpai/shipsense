"""
Dynamic Pricing tab (post-v1.0, module 5).

Reads data/processed/pricing.csv made by src/build_pricing.py.
Nothing is computed live. This is a rule-based price INDEX, not a trained model.
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

PRICING_PATH = "data/processed/pricing.csv"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_pricing(path: str):
    return pd.read_csv(path)


def render_pricing_tab():
    st.subheader("Dynamic Pricing (illustrative price index)")
    st.info(
        "A rule-based price index, not a trained model. The dataset has no "
        "prices, fees or revenue, so nothing here is learned from real prices or "
        "customer behaviour, and no revenue effect is claimed. Each region, "
        "weekday and hour slot gets a multiplier on a base index of 1.00, between "
        "0.90x and 1.40x. It combines three signals, each ranked against the "
        "other slots of the same region: predicted demand (50 percent weight), "
        "predicted congestion (30 percent) and delay risk (20 percent). The "
        "weights and the range are my choices."
    )

    if not os.path.exists(PRICING_PATH):
        st.error(
            "Pricing data not found. Run `python src/build_pricing.py` to "
            "generate `data/processed/pricing.csv`."
        )
        return

    df = load_pricing(PRICING_PATH)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slots priced", "%d" % len(df))
    c2.metric("Regions", "%d" % df["region_id"].nunique())
    c3.metric("Median multiplier", "%.2fx" % df["multiplier"].median())
    c4.metric(
        "Slots at 1.20x or more", "%.1f%%" % (100.0 * (df["multiplier"] >= 1.2).mean())
    )
    st.caption(
        "Multipliers range from %.2fx to %.2fx. A slot in the middle of its "
        "region on all three signals gets 1.00x."
        % (df["multiplier"].min(), df["multiplier"].max())
    )

    st.markdown("#### Explore one region")
    regions = sorted(df["region_id"].unique().tolist())
    region = st.selectbox(
        "Choose a region", options=regions, format_func=lambda r: "Region %d" % r
    )
    reg = df[df["region_id"] == region].copy()
    reg["day"] = reg["day_of_week"].map(lambda d: DAY_NAMES[int(d)])
    reg["hour"] = reg["hour_of_day"].map(lambda h: "%02d:00" % int(h))

    hi = reg.loc[reg["multiplier"].idxmax()]
    lo = reg.loc[reg["multiplier"].idxmin()]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Region median", "%.2fx" % reg["multiplier"].median())
    m2.metric("Highest: %s %s" % (hi["day"], hi["hour"]), "%.2fx" % hi["multiplier"])
    m3.metric("Lowest: %s %s" % (lo["day"], lo["hour"]), "%.2fx" % lo["multiplier"])
    m4.metric(
        "Slots at 1.20x or more",
        "%d of %d" % (int((reg["multiplier"] >= 1.2).sum()), len(reg)),
    )

    pivot = reg.pivot(
        index="day_of_week", columns="hour_of_day", values="multiplier"
    ).reindex(index=range(7))
    pivot.index = DAY_NAMES
    heat = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0.9,
        zmax=1.4,
        labels=dict(x="Hour of day", y="Weekday", color="Multiplier"),
        title="Price multiplier by weekday and hour (Region %d)" % region,
    )
    heat.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(heat)

    show = [
        "day",
        "hour",
        "multiplier",
        "predicted_orders",
        "predicted_min_per_km",
        "risk_share",
    ]
    left, right = st.columns(2)
    with left:
        st.markdown("**Most expensive slots**")
        st.dataframe(reg.nlargest(5, "multiplier")[show])
    with right:
        st.markdown("**Cheapest slots**")
        st.dataframe(reg.nsmallest(5, "multiplier")[show])

    st.markdown("#### Inspect one slot")
    s1, s2 = st.columns(2)
    day = s1.selectbox(
        "Weekday", options=list(range(7)), format_func=lambda d: DAY_NAMES[d]
    )
    hour = s2.selectbox(
        "Hour", options=list(range(24)), format_func=lambda h: "%02d:00" % h
    )
    slot = reg[(reg["day_of_week"] == day) & (reg["hour_of_day"] == hour)]
    if slot.empty:
        st.warning("No data for this slot.")
    else:
        r = slot.iloc[0]
        breakdown = pd.DataFrame(
            {
                "Signal": ["Demand", "Congestion", "Delay risk"],
                "Real value": [
                    "%.1f orders" % r["predicted_orders"],
                    "%.1f min per km" % r["predicted_min_per_km"],
                    "%.1f percent at risk" % (100.0 * r["risk_share"]),
                ],
                "Vs region average": [
                    "%.2fx" % r["demand_ratio"],
                    "%.2fx" % r["congestion_ratio"],
                    "%.2fx" % r["risk_ratio"],
                ],
                "Rank in region (0 low, 1 high)": [
                    "%.2f" % r["demand_rank"],
                    "%.2f" % r["congestion_rank"],
                    "%.2f" % r["risk_rank"],
                ],
                "Weight": ["50 percent", "30 percent", "20 percent"],
            }
        )
        st.dataframe(breakdown)
        st.caption(
            "Weighted rank score %.2f gives a multiplier of %.2fx. This slot has "
            "%d past orders behind its delay-risk figure; slots with few past "
            "orders are blended toward the region average."
            % (r["score"], r["multiplier"], int(r["history_orders"]))
        )

    st.markdown("#### All slots")
    hist = px.histogram(
        df,
        x="multiplier",
        nbins=30,
        labels={"multiplier": "Price multiplier"},
        title="Distribution of multipliers across all slots",
    )
    hist.update_layout(height=350, yaxis_title="Number of slots")
    st.plotly_chart(hist)

    with st.expander("Assumptions and limits"):
        n_all = len(df)
        n_none = int((df["history_orders"] == 0).sum())
        st.markdown(
            "- A price index with a base of 1.00, not a currency amount. The "
            "dataset has no prices, fees or revenue.\n"
            "- A rule, not a trained model. No claim is made about revenue, profit "
            "or how customers would react.\n"
            "- Signals: predicted orders (demand forecast), predicted minutes per "
            "km (traffic proxy, not live traffic) and the share of past orders "
            "marked At Risk by the ETA model.\n"
            "- Each signal is ranked within its own region, so ranks show where a "
            "slot stands relative to its neighbours, not how big the gap is. The "
            "real values are shown in the slot inspector.\n"
            "- Weights (50, 30 and 20 percent) and the range (0.90x to 1.40x, "
            "with 1.00x for a slot in the middle) are my choices, not from the "
            "data.\n"
            "- Delay-risk history is blended toward the region average for slots "
            "with few past orders."
        )
        st.markdown(
            "- %d of %d slots have no past orders, so for them delay risk equals "
            "the region average and the multiplier is driven by demand and "
            "congestion alone." % (n_none, n_all)
        )
        st.markdown(
            "- The forecast files contain impossible weekday and weekend-flag "
            "pairs. They were removed, leaving 17 regions x 24 hours x 7 "
            "weekdays.\n"
            "- A first version using plain ratios to the region average put "
            "about two thirds of slots at the minimum price, so it was replaced "
            "by the ranks.\n"
            "- Computed offline by `src/build_pricing.py` and saved to CSV. The "
            "app does not compute anything live."
        )

    with st.expander("All slots (table)"):
        st.dataframe(df)