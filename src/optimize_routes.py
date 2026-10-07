"""
ShipSense - Vehicle Routing (Module 3b), offline batch script.

Run with the venv_routing environment (it has OR-Tools):
    python src/optimize_routes.py        (full run, 200 batches)
    python src/optimize_routes.py 5      (quick test, 5 batches)

Reads:   data/processed/predictions.csv
Writes:  data/processed/routes.csv
         data/processed/routes_summary.csv
The app only reads these files, no live solving.

Assumptions (also shown in the app):
- Distances are straight-line (haversine), NOT road distances.
- A batch = up to 30 orders from one region, weekday and hour.
  The dataset has no dates, so this is a simulation, not a real day.
- Depot = median accept location of the batch's orders.
- Vehicles are anonymous (Vehicle 1, 2, 3...), max CAPACITY stops each.
- All vehicles leave the depot at the start of the hour.
- Travel = SPEED_KMH, plus SERVICE_SEC at every stop.
- STRESS SCENARIO: deadline = STRESS_FRACTION x the order's promised ETA.
  This is a test scenario, NOT the real promise. Lateness against the
  real promised time is also reported.
- Deadlines are soft: late stops are penalised, not forbidden.
- Baseline = nearest-neighbor, filling vehicle 1 up to CAPACITY stops,
  then vehicle 2, and so on.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "processed" / "predictions.csv"
OUT_ROUTES = ROOT / "data" / "processed" / "routes.csv"
OUT_SUMMARY = ROOT / "data" / "processed" / "routes_summary.csv"

MIN_STOPS = 12
MAX_STOPS = 30
MAX_RADIUS_KM = 30
CAPACITY = 10
SPEED_KMH = 20
SERVICE_SEC = 180
STRESS_FRACTION = 0.30
LATE_COST_PER_SEC = 8   # about 0.5 km of extra driving per late minute
N_BATCHES = 200
SOLVER_SECONDS = 3
SEED = 42


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def travel_sec(km):
    return km / SPEED_KMH * 3600.0


def evaluate(routes, dist, deadline_s):
    """Total km, late stops, late seconds and arrival time of every stop."""
    km = 0.0
    late_stops = 0
    late_sec = 0.0
    arrivals = {}
    for route in routes:
        if not route:
            continue
        t = 0.0
        prev = 0
        for node in route:
            km += dist[prev][node]
            t += travel_sec(dist[prev][node])
            arrivals[node] = t
            if t > deadline_s[node]:
                late_stops += 1
                late_sec += t - deadline_s[node]
            t += SERVICE_SEC
            prev = node
        km += dist[prev][0]
    return float(km), late_stops, float(late_sec), arrivals


def nearest_neighbor(dist, n_veh):
    left = set(range(1, len(dist)))
    routes = []
    for _ in range(n_veh):
        route = []
        cur = 0
        while left and len(route) < CAPACITY:
            nxt = min(left, key=lambda j: dist[cur][j])
            route.append(nxt)
            left.remove(nxt)
            cur = nxt
        routes.append(route)
    return routes


def solve_vrp(dist, deadline_s, n_veh):
    n = len(dist)
    dist_m = np.rint(dist * 1000).astype(int)
    time_s = np.rint(dist / SPEED_KMH * 3600).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, n_veh, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_m[manager.IndexToNode(i)][manager.IndexToNode(j)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    def time_cb(i, j):
        a = manager.IndexToNode(i)
        b = manager.IndexToNode(j)
        service = 0 if a == 0 else SERVICE_SEC
        return int(time_s[a][b]) + service

    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 0, 10**7, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        time_dim.SetCumulVarSoftUpperBound(
            manager.NodeToIndex(node), int(deadline_s[node]), LATE_COST_PER_SEC
        )

    def demand_cb(i):
        return 0 if manager.IndexToNode(i) == 0 else 1

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [CAPACITY] * n_veh, True, "Stops"
    )

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

    routes = []
    for v in range(n_veh):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)
    return routes


def pct(saved, base):
    return round(100 * saved / base, 1) if base > 0 else 0.0


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else N_BATCHES
    df = pd.read_csv(IN_FILE)
    print("Loaded", len(df), "orders")

    keys = ["region_id", "day_of_week", "hour_of_day"]
    grouped = df.groupby(keys)
    sizes = grouped.size()
    eligible = sizes[sizes >= MIN_STOPS].reset_index()[keys]
    print("Eligible region+weekday+hour slots:", len(eligible))

    chosen = eligible.sample(n=min(n_batches, len(eligible)), random_state=SEED)

    route_rows = []
    summary_rows = []
    batch_no = 0

    for row in chosen.itertuples(index=False):
        rid, dow, hod = int(row.region_id), int(row.day_of_week), int(row.hour_of_day)
        sub = grouped.get_group((rid, dow, hod))

        dlat = float(sub["accept_gps_lat"].median())
        dlng = float(sub["accept_gps_lng"].median())
        from_depot = haversine_km(
            dlat, dlng, sub["delivery_gps_lat"].values, sub["delivery_gps_lng"].values
        )
        sub = sub[from_depot <= MAX_RADIUS_KM]
        if len(sub) < MIN_STOPS:
            continue
        if len(sub) > MAX_STOPS:
            sub = sub.sample(n=MAX_STOPS, random_state=SEED)
        sub = sub.sort_values("order_id").reset_index(drop=True)

        n_stops = len(sub)
        n_veh = math.ceil(n_stops / CAPACITY)

        lats = np.concatenate([[dlat], sub["delivery_gps_lat"].values])
        lngs = np.concatenate([[dlng], sub["delivery_gps_lng"].values])
        dist = haversine_km(lats[:, None], lngs[:, None], lats[None, :], lngs[None, :])

        promised_s = np.concatenate([[0.0], sub["promised_eta_minutes"].values * 60.0])
        stress_s = promised_s * STRESS_FRACTION

        nn_routes = nearest_neighbor(dist, n_veh)
        nn_km, nn_late, nn_late_sec, nn_arr = evaluate(nn_routes, dist, stress_s)

        opt_routes = solve_vrp(dist, stress_s, n_veh)
        if opt_routes is None:
            continue
        opt_km, opt_late, opt_late_sec, opt_arr = evaluate(opt_routes, dist, stress_s)

        # Never report a result worse than the baseline: fall back to it.
        nn_cost = nn_km * 1000 + LATE_COST_PER_SEC * nn_late_sec
        opt_cost = opt_km * 1000 + LATE_COST_PER_SEC * opt_late_sec
        if opt_cost > nn_cost:
            opt_routes = nn_routes
            opt_km, opt_late, opt_late_sec, opt_arr = nn_km, nn_late, nn_late_sec, nn_arr

        nn_late_real = evaluate(nn_routes, dist, promised_s)[1]
        opt_late_real = evaluate(opt_routes, dist, promised_s)[1]

        batch_no += 1
        batch_id = "B%03d" % batch_no
        order_ids = [None] + sub["order_id"].tolist()

        for method, routes, arr in (
            ("nn", nn_routes, nn_arr),
            ("optimized", opt_routes, opt_arr),
        ):
            for v, route in enumerate(routes, start=1):
                if not route:
                    continue
                route_rows.append(
                    {
                        "batch_id": batch_id,
                        "method": method,
                        "vehicle": v,
                        "seq": 0,
                        "stop_type": "depot",
                        "order_id": None,
                        "lat": round(dlat, 6),
                        "lng": round(dlng, 6),
                        "arrival_min": 0.0,
                        "deadline_min": None,
                        "late_min": 0.0,
                    }
                )
                for pos, node in enumerate(route, start=1):
                    arrival = arr[node]
                    deadline = stress_s[node]
                    route_rows.append(
                        {
                            "batch_id": batch_id,
                            "method": method,
                            "vehicle": v,
                            "seq": pos,
                            "stop_type": "stop",
                            "order_id": order_ids[node],
                            "lat": round(float(lats[node]), 6),
                            "lng": round(float(lngs[node]), 6),
                            "arrival_min": round(arrival / 60, 1),
                            "deadline_min": round(deadline / 60, 1),
                            "late_min": round(max(0.0, arrival - deadline) / 60, 1),
                        }
                    )

        summary_rows.append(
            {
                "batch_id": batch_id,
                "region_id": rid,
                "day_of_week": dow,
                "hour_of_day": hod,
                "n_stops": n_stops,
                "n_vehicles": n_veh,
                "nn_km": round(nn_km, 2),
                "optimized_km": round(opt_km, 2),
                "saved_pct_vs_nn": pct(nn_km - opt_km, nn_km),
                "nn_late_stress": nn_late,
                "opt_late_stress": opt_late,
                "nn_late_real": nn_late_real,
                "opt_late_real": opt_late_real,
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
    print("Distance saved vs NEAREST-NEIGHBOR: median %.1f%%, average %.1f%%" % (
        summary["saved_pct_vs_nn"].median(), summary["saved_pct_vs_nn"].mean()))
    print("Late stops, STRESS deadlines: nearest-neighbor %d, optimized %d" % (
        summary["nn_late_stress"].sum(), summary["opt_late_stress"].sum()))
    print("Late stops, REAL promised times: nearest-neighbor %d, optimized %d" % (
        summary["nn_late_real"].sum(), summary["opt_late_real"].sum()))
    print("Stops in total:", int(summary["n_stops"].sum()))
    print("Wrote", OUT_ROUTES)
    print("Wrote", OUT_SUMMARY)


if __name__ == "__main__":
    main()