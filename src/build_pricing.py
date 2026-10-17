"""
ShipSense - Dynamic Pricing (Module 5), offline script.

Run with the app environment (venv):
    python src/build_pricing.py

Reads:   data/processed/demand_forecast.csv
         data/processed/traffic_forecast.csv
         data/processed/predictions.csv
Writes:  data/processed/pricing.csv
The app only reads this file, nothing is computed live.

IMPORTANT: this is a transparent, rule-based PRICE INDEX, not a trained model.
The dataset has no prices, fees or revenue, so nothing here is learned from real
prices or customer behaviour. Weights and limits are my choices.

Data note: the forecast files list every weekday with BOTH weekend flags, which
includes impossible pairs. Only rows where the weekend flag matches the weekday
(weekend days are read from the order data) are kept.

Rule, for each region + weekday + hour slot (base price index = 1.00):
    Each signal is turned into a rank within its own region
    (0 = lowest slot of the region, 1 = highest):
        demand rank     = rank of predicted orders
        congestion rank = rank of predicted min per km
        risk rank       = rank of the slot's At Risk share (slots with little
                          history are blended toward the region average using
                          PRIOR_ORDERS pseudo-orders)
    score      = W_DEMAND*demand + W_TRAFFIC*congestion + W_RISK*risk  (0..1)
    multiplier = 1.00 at score 0.5, falling to FLOOR at score 0,
                 rising to CAP at score 1
The plain ratios to the region average are kept as information columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEMAND_FILE = ROOT / "data" / "processed" / "demand_forecast.csv"
TRAFFIC_FILE = ROOT / "data" / "processed" / "traffic_forecast.csv"
PRED_FILE = ROOT / "data" / "processed" / "predictions.csv"
OUT_FILE = ROOT / "data" / "processed" / "pricing.csv"

W_DEMAND = 0.5
W_TRAFFIC = 0.3
W_RISK = 0.2
FLOOR = 0.90
CAP = 1.40
PRIOR_ORDERS = 20

KEYS = ["region_id", "day_of_week", "hour_of_day"]


def to_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def region_rank(df, col):
    """Rank of each slot within its own region, scaled 0 (lowest) to 1 (highest)."""
    grp = df.groupby("region_id")[col]
    r = grp.rank(method="average")
    n = grp.transform("count")
    return ((r - 1) / (n - 1).replace(0, np.nan)).fillna(0.5)


def main():
    # delay-risk history, per slot and per region
    pred = pd.read_csv(PRED_FILE, usecols=KEYS + ["risk_status", "is_weekend"])
    pred["is_weekend"] = to_bool(pred["is_weekend"])
    pred["at_risk"] = (pred["risk_status"] == "At Risk").astype(int)

    weekend_share = pred.groupby("day_of_week")["is_weekend"].mean()
    weekend_days = set(weekend_share[weekend_share > 0.5].index)
    print("Weekend days found in the order data:", sorted(weekend_days))

    def consistent(df, name):
        flag = to_bool(df["is_weekend"])
        ok = flag == df["day_of_week"].isin(weekend_days)
        print("%s: %d rows, %d kept after the weekday check, %d regions" % (
            name, len(df), int(ok.sum()), df["region_id"].nunique()))
        return df[ok]

    demand = pd.read_csv(DEMAND_FILE)
    demand = consistent(demand, "demand forecast")[KEYS + ["predicted_orders"]]
    traffic = pd.read_csv(TRAFFIC_FILE)
    traffic = consistent(traffic, "traffic forecast")[
        KEYS + ["predicted_min_per_km", "congestion_level"]
    ]
    grid = demand.merge(traffic, on=KEYS, how="inner", validate="one_to_one")
    print("Grid slots:", len(grid), "| regions:", grid["region_id"].nunique())

    global_share = float(pred["at_risk"].mean())
    hist = (
        pred.groupby(KEYS)
        .agg(history_orders=("at_risk", "size"), history_at_risk=("at_risk", "sum"))
        .reset_index()
    )
    region_share = (
        pred.groupby("region_id")["at_risk"]
        .mean()
        .rename("region_risk_share")
        .reset_index()
    )

    grid = grid.merge(hist, on=KEYS, how="left")
    grid = grid.merge(region_share, on="region_id", how="left")
    grid["history_orders"] = grid["history_orders"].fillna(0).astype(int)
    grid["history_at_risk"] = grid["history_at_risk"].fillna(0)
    grid["region_risk_share"] = grid["region_risk_share"].fillna(global_share)
    grid["risk_share"] = (
        grid["history_at_risk"] + PRIOR_ORDERS * grid["region_risk_share"]
    ) / (grid["history_orders"] + PRIOR_ORDERS)

    # plain ratios to the region average (information only)
    by_region = grid.groupby("region_id")
    demand_mean = by_region["predicted_orders"].transform("mean")
    traffic_mean = by_region["predicted_min_per_km"].transform("mean")
    grid["demand_ratio"] = (
        grid["predicted_orders"] / demand_mean.replace(0, np.nan)
    ).fillna(1.0)
    grid["congestion_ratio"] = (
        grid["predicted_min_per_km"] / traffic_mean.replace(0, np.nan)
    ).fillna(1.0)
    grid["risk_ratio"] = (
        grid["risk_share"] / grid["region_risk_share"].replace(0, np.nan)
    ).fillna(1.0)

    # for comparison only: the plain-ratio rule that was tried first
    raw_old = (
        W_DEMAND * grid["demand_ratio"]
        + W_TRAFFIC * grid["congestion_ratio"]
        + W_RISK * grid["risk_ratio"]
    )

    # the rank-based rule that is actually used
    grid["demand_rank"] = region_rank(grid, "predicted_orders")
    grid["congestion_rank"] = region_rank(grid, "predicted_min_per_km")
    grid["risk_rank"] = region_rank(grid, "risk_share")
    grid["score"] = (
        W_DEMAND * grid["demand_rank"]
        + W_TRAFFIC * grid["congestion_rank"]
        + W_RISK * grid["risk_rank"]
    )
    s = grid["score"]
    grid["multiplier"] = np.where(
        s < 0.5,
        1.0 - (1.0 - FLOOR) * (0.5 - s) / 0.5,
        1.0 + (CAP - 1.0) * (s - 0.5) / 0.5,
    )
    grid["multiplier"] = grid["multiplier"].clip(FLOOR, CAP)

    cols = KEYS + [
        "predicted_orders",
        "predicted_min_per_km",
        "congestion_level",
        "history_orders",
        "risk_share",
        "demand_ratio",
        "congestion_ratio",
        "risk_ratio",
        "demand_rank",
        "congestion_rank",
        "risk_rank",
        "score",
        "multiplier",
    ]
    out = grid[cols].sort_values(KEYS).reset_index(drop=True)
    for c, n in [
        ("predicted_orders", 2),
        ("predicted_min_per_km", 2),
        ("risk_share", 4),
        ("demand_ratio", 3),
        ("congestion_ratio", 3),
        ("risk_ratio", 3),
        ("demand_rank", 3),
        ("congestion_rank", 3),
        ("risk_rank", 3),
        ("score", 3),
        ("multiplier", 3),
    ]:
        out[c] = out[c].round(n)
    out.to_csv(OUT_FILE, index=False)

    m = out["multiplier"]
    q = m.quantile([0, 0.25, 0.5, 0.75, 1]).round(3).tolist()
    print("Multiplier min / 25% / median / 75% / max:", q)
    print("Slots below 1.00x: %.1f%%, above 1.00x: %.1f%%, 1.20x or more: %.1f%%" % (
        100.0 * float((m < 1.0).mean()),
        100.0 * float((m > 1.0).mean()),
        100.0 * float((m >= 1.2).mean())))
    print("For comparison, the plain-ratio rule would put %.1f%% of slots at the floor and %.1f%% at the cap" % (
        100.0 * float((raw_old < FLOOR).mean()),
        100.0 * float((raw_old > CAP).mean())))
    print("Slots with no order history:", int((out["history_orders"] == 0).sum()))
    print("Wrote", OUT_FILE)


if __name__ == "__main__":
    main()