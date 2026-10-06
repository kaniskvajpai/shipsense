"""
ShipSense - Vehicle Routing (Module 3a), offline batch script.

Run with the venv_routing environment (it has OR-Tools):
    python src/optimize_routes.py        (full run, about 200 batches)
    python src/optimize_routes.py 5      (quick test, 5 batches)

Outputs (the app only reads these, no live solving):
    data/processed/routes.csv
    data/processed/routes_summary.csv

Assumptions (also shown in the app):
- Distances are straight-line (haversine), NOT road distances.
- Depot = median accept location of the courier.
- A batch = a sample of one courier's orders from one weekday + hour.
  The dataset has no dates, so this is a simulation, not a real day.
- Two baselines are compared against the optimized route:
  1. Random order (stops in order_id order) = upper bound, not realistic.
  2. Nearest-neighbor (always drive to the closest unvisited stop)
     = realistic estimate of a sensible human route.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "processed" / "cleaned_data.csv"
OUT_ROUTES = ROOT / "data" / "processed" / "routes.csv"
OUT_SUMMARY = ROOT / "data" / "processed" / "routes_summary.csv"

MIN_STOPS = 8
MAX_STOPS = 30
MAX_RADIUS_KM = 30
N_BATCHES = 200
SOLVER_SECONDS = 2
SEED = 42


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def route_km(dist, order):
    total = 0.0
    for a, b in zip(order, order[1:]):
        total += dist[a][b]
    total += dist[order[-1]][order[0]]
    return float(total)


def nearest_neighbor(dist):
    n = len(dist)
    order = [0]
    left = set(range(1, n))
    while left:
        last = order[-1]
        nxt = min(left, key=lambda j: dist[last][j])
        order.append(nxt)
        left.remove(nxt)
    return order


def solve_tsp(dist_km):
    n = len(dist_km)
    scaled = np.rint(dist_km * 1000).astype(int)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cost(i, j):
        return int(scaled[manager.IndexToNode(i)][manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = SOLVER_SECONDS
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None
    order = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return order


def pct(saved, base):
    return round(100 * saved / base, 1) if base > 0 else 0.0


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else N_BATCHES
    df = pd.read_csv(IN_FILE)
    print("Loaded", len(df), "orders")

    depots = df.groupby("courier_id")[["accept_gps_lat", "accept_gps_lng"]].median()
    keys = ["courier_id", "day_of_week", "hour_of_day"]
    grouped = df.groupby(keys)
    sizes = grouped.size()
    eligible = sizes[sizes >= MIN_STOPS].reset_index()[keys]
    print("Eligible courier+weekday+hour slots:", len(eligible))

    chosen = eligible.sample(n=min(n_batches, len(eligible)), random_state=SEED)

    route_rows = []
    summary_rows = []
    batch_no = 0

    for row in chosen.itertuples(index=False):
        cid, dow, hod = int(row.courier_id), int(row.day_of_week), int(row.hour_of_day)
        sub = grouped.get_group((cid, dow, hod))
        dlat, dlng = depots.loc[cid, "accept_gps_lat"], depots.loc[cid, "accept_gps_lng"]

        from_depot = haversine_km(
            dlat, dlng, sub["delivery_gps_lat"].values, sub["delivery_gps_lng"].values
        )
        sub = sub[from_depot <= MAX_RADIUS_KM]
        if len(sub) < MIN_STOPS:
            continue
        if len(sub) > MAX_STOPS:
            sub = sub.sample(n=MAX_STOPS, random_state=SEED)
        sub = sub.sort_values("order_id").reset_index(drop=True)

        lats = np.concatenate([[dlat], sub["delivery_gps_lat"].values])
        lngs = np.concatenate([[dlng], sub["delivery_gps_lng"].values])
        dist = haversine_km(lats[:, None], lngs[:, None], lats[None, :], lngs[None, :])

        random_order = list(range(len(lats)))
        random_km = route_km(dist, random_order)

        nn_order = nearest_neighbor(dist)
        nn_km = route_km(dist, nn_order)

        opt_order = solve_tsp(dist)
        if opt_order is None:
            continue
        opt_km = route_km(dist, opt_order)
        # Never report a route worse than a baseline: fall back to the best of the three.
        if opt_km > nn_km or opt_km > random_km:
            if nn_km <= random_km:
                opt_order, opt_km = nn_order, nn_km
            else:
                opt_order, opt_km = random_order, random_km

        nn_pos = {node: pos for pos, node in enumerate(nn_order)}

        batch_no += 1
        batch_id = "B%03d" % batch_no
        order_ids = [None] + sub["order_id"].tolist()

        for position, node in enumerate(opt_order):
            route_rows.append(
                {
                    "batch_id": batch_id,
                    "courier_id": cid,
                    "day_of_week": dow,
                    "hour_of_day": hod,
                    "stop_type": "depot" if node == 0 else "stop",
                    "random_seq": node,
                    "nn_seq": nn_pos[node],
                    "optimized_seq": position,
                    "order_id": order_ids[node],
                    "lat": round(float(lats[node]), 6),
                    "lng": round(float(lngs[node]), 6),
                }
            )

        summary_rows.append(
            {
                "batch_id": batch_id,
                "courier_id": cid,
                "day_of_week": dow,
                "hour_of_day": hod,
                "n_stops": len(sub),
                "random_km": round(random_km, 2),
                "nn_km": round(nn_km, 2),
                "optimized_km": round(opt_km, 2),
                "saved_pct_vs_random": pct(random_km - opt_km, random_km),
                "saved_pct_vs_nn": pct(nn_km - opt_km, nn_km),
            }
        )

        if batch_no % 10 == 0:
            print("Solved", batch_no, "batches")

    if not summary_rows:
        print("No batches were solved - nothing written.")
        return

    pd.DataFrame(route_rows).to_csv(OUT_ROUTES, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("Done:", len(summary), "batches")
    print("Saved vs RANDOM order (upper bound): median %.1f%%, average %.1f%%" % (
        summary["saved_pct_vs_random"].median(), summary["saved_pct_vs_random"].mean()))
    print("Saved vs NEAREST-NEIGHBOR (realistic): median %.1f%%, average %.1f%%" % (
        summary["saved_pct_vs_nn"].median(), summary["saved_pct_vs_nn"].mean()))
    print("Wrote", OUT_ROUTES)
    print("Wrote", OUT_SUMMARY)


if __name__ == "__main__":
    main()