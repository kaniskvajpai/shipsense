"""
ShipSense - Rider Allocation (Module 4), offline batch script.

Run with the venv_routing environment (it has OR-Tools):
    python src/optimize_allocation.py        (full run, 200 batches)
    python src/optimize_allocation.py 5      (quick test, 5 batches)

Reads:   data/processed/predictions.csv
Writes:  data/processed/allocation.csv           (one row per order)
         data/processed/allocation_riders.csv    (one row per rider per batch)
         data/processed/allocation_summary.csv   (one row per batch)
The app only reads these files, no live solving.

Assumptions (also shown in the app):
- Distances are straight-line (haversine), NOT road distances.
- A batch = up to MAX_RIDERS real riders (couriers) and up to MAX_ORDERS of
  the orders they actually delivered, from one region, weekday and hour.
  The dataset has no dates, so this is a simulation, not a real day.
- Rider location = median accept location of that rider's orders in the
  batch. This is a proxy, not live GPS, and it favours the actual assignment.
- Baseline = the actual historical assignment.
- Optimized = cheapest assignment where no rider gets more than
  CAP_FACTOR x the average number of orders per rider (my choice).
- Tie-break: moving an order away from its actual rider costs an extra
  CHANGE_PENALTY_M metres, so equal-cost options keep the actual rider and
  the "reassigned" share only counts moves that matter.
- "Nearest rider" = every order goes to the closest rider, ignoring workload.
  It is a distance lower bound, not a realistic plan.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "processed" / "predictions.csv"
OUT_ORDERS = ROOT / "data" / "processed" / "allocation.csv"
OUT_RIDERS = ROOT / "data" / "processed" / "allocation_riders.csv"
OUT_SUMMARY = ROOT / "data" / "processed" / "allocation_summary.csv"

MIN_RIDERS = 3
MIN_ORDERS = 20
MAX_RIDERS = 6
MAX_ORDERS = 60
MAX_RADIUS_KM = 30
CAP_FACTOR = 1.25
CHANGE_PENALTY_M = 100
N_BATCHES = 200
SOLVER_SECONDS = 3
SEED = 42

COLS = [
    "order_id",
    "region_id",
    "day_of_week",
    "hour_of_day",
    "courier_id",
    "accept_gps_lat",
    "accept_gps_lng",
    "delivery_gps_lat",
    "delivery_gps_lng",
]


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def solve_assignment(cost, cap, actual):
    """Cheapest assignment of every order to one rider, max cap orders per rider."""
    n_r, n_o = cost.shape
    scaled = np.rint(cost * 1000).astype(int)

    model = cp_model.CpModel()
    x = [
        [model.new_bool_var("x_%d_%d" % (r, o)) for o in range(n_o)]
        for r in range(n_r)
    ]
    for o in range(n_o):
        model.add(sum(x[r][o] for r in range(n_r)) == 1)
    for r in range(n_r):
        model.add(sum(x[r][o] for o in range(n_o)) <= cap)
    model.minimize(
        sum(
            (int(scaled[r][o]) + (0 if actual[o] == r else CHANGE_PENALTY_M)) * x[r][o]
            for r in range(n_r)
            for o in range(n_o)
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_SECONDS
    solver.parameters.random_seed = SEED
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    assign = np.zeros(n_o, dtype=int)
    for o in range(n_o):
        for r in range(n_r):
            if solver.value(x[r][o]):
                assign[o] = r
                break
    return assign


def pct(saved, base):
    return round(100 * saved / base, 1) if base > 0 else 0.0


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else N_BATCHES
    df = pd.read_csv(IN_FILE, usecols=COLS)
    print("Loaded", len(df), "orders")

    keys = ["region_id", "day_of_week", "hour_of_day"]
    grouped = df.groupby(keys)
    stats = grouped.agg(orders=("order_id", "size"), riders=("courier_id", "nunique"))
    eligible = stats[(stats.riders >= MIN_RIDERS) & (stats.orders >= MIN_ORDERS)]
    eligible = eligible.reset_index()[keys].sample(frac=1, random_state=SEED)
    print("Eligible region+weekday+hour slots:", len(eligible))

    order_rows = []
    rider_rows = []
    summary_rows = []
    batch_no = 0

    for row in eligible.itertuples(index=False):
        if batch_no >= n_batches:
            break
        sub = grouped.get_group(
            (int(row.region_id), int(row.day_of_week), int(row.hour_of_day))
        )

        # keep the busiest riders, then drop likely GPS outliers
        top = sub["courier_id"].value_counts().head(MAX_RIDERS).index
        sub = sub[sub["courier_id"].isin(top)]
        clat = sub["accept_gps_lat"].median()
        clng = sub["accept_gps_lng"].median()
        far = haversine_km(
            clat, clng, sub["delivery_gps_lat"].values, sub["delivery_gps_lng"].values
        )
        sub = sub[far <= MAX_RADIUS_KM]
        if len(sub) > MAX_ORDERS:
            sub = sub.sample(n=MAX_ORDERS, random_state=SEED)
        sub = sub.sort_values("order_id").reset_index(drop=True)

        riders = sorted(sub["courier_id"].unique())
        if len(riders) < MIN_RIDERS or len(sub) < MIN_ORDERS:
            continue

        loc = sub.groupby("courier_id")[["accept_gps_lat", "accept_gps_lng"]].median()
        loc = loc.loc[riders]
        rlat = loc["accept_gps_lat"].values
        rlng = loc["accept_gps_lng"].values
        olat = sub["delivery_gps_lat"].values
        olng = sub["delivery_gps_lng"].values

        cost = haversine_km(rlat[:, None], rlng[:, None], olat[None, :], olng[None, :])
        n_r, n_o = cost.shape
        idx = np.arange(n_o)

        rider_index = {r: i for i, r in enumerate(riders)}
        actual = sub["courier_id"].map(rider_index).values.astype(int)
        actual_km = float(cost[actual, idx].sum())
        nearest_km = float(cost.min(axis=0).sum())

        cap = math.ceil(CAP_FACTOR * n_o / n_r)
        assign = solve_assignment(cost, cap, actual)
        if assign is None:
            continue
        opt_km = float(cost[assign, idx].sum())

        batch_no += 1
        batch_id = "A%03d" % batch_no

        act_load = np.bincount(actual, minlength=n_r)
        opt_load = np.bincount(assign, minlength=n_r)
        avg_load = n_o / n_r

        for o in range(n_o):
            order_rows.append(
                {
                    "batch_id": batch_id,
                    "order_id": int(sub["order_id"].iloc[o]),
                    "lat": round(float(olat[o]), 6),
                    "lng": round(float(olng[o]), 6),
                    "actual_rider": int(riders[actual[o]]),
                    "optimized_rider": int(riders[assign[o]]),
                    "actual_km": round(float(cost[actual[o], o]), 2),
                    "optimized_km": round(float(cost[assign[o], o]), 2),
                }
            )
        for i, rid in enumerate(riders):
            rider_rows.append(
                {
                    "batch_id": batch_id,
                    "rider_id": int(rid),
                    "lat": round(float(rlat[i]), 6),
                    "lng": round(float(rlng[i]), 6),
                    "actual_orders": int(act_load[i]),
                    "optimized_orders": int(opt_load[i]),
                }
            )
        summary_rows.append(
            {
                "batch_id": batch_id,
                "region_id": int(row.region_id),
                "day_of_week": int(row.day_of_week),
                "hour_of_day": int(row.hour_of_day),
                "n_orders": n_o,
                "n_riders": n_r,
                "cap_per_rider": cap,
                "actual_km": round(actual_km, 2),
                "optimized_km": round(opt_km, 2),
                "nearest_rider_km": round(nearest_km, 2),
                "saved_pct_vs_actual": pct(actual_km - opt_km, actual_km),
                "actual_max_over_avg": round(float(act_load.max()) / avg_load, 2),
                "optimized_max_over_avg": round(float(opt_load.max()) / avg_load, 2),
                "reassigned_pct": round(100.0 * float((actual != assign).mean()), 1),
            }
        )

        if batch_no % 10 == 0:
            print("Solved", batch_no, "batches")

    if not summary_rows:
        print("No batches were solved - nothing written.")
        return

    pd.DataFrame(order_rows).to_csv(OUT_ORDERS, index=False)
    pd.DataFrame(rider_rows).to_csv(OUT_RIDERS, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    shorter = int((summary["optimized_km"] < summary["actual_km"]).sum())
    print("Done:", len(summary), "batches,", int(summary["n_orders"].sum()), "orders")
    print("Distance saved vs ACTUAL: median %.1f%%, average %.1f%%" % (
        summary["saved_pct_vs_actual"].median(), summary["saved_pct_vs_actual"].mean()))
    print("Optimized shorter than actual in %d of %d batches" % (shorter, len(summary)))
    print("Busiest rider vs average workload: actual median %.2fx, optimized median %.2fx" % (
        summary["actual_max_over_avg"].median(), summary["optimized_max_over_avg"].median()))
    print("Median share of orders given to a different rider: %.1f%%" % (
        summary["reassigned_pct"].median()))
    print("Wrote", OUT_ORDERS)
    print("Wrote", OUT_RIDERS)
    print("Wrote", OUT_SUMMARY)


if __name__ == "__main__":
    main()