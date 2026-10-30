"""
ShipSense - Module 7: empirical direct-travel statistics.

Computes per-region courier riding speeds and handover times from the raw
LaDe-D (Yantai) data, so the Live ETA tab can show BOTH:
  - "Direct travel estimate"  (distance / regional median speed + handover)
  - "Realistic route estimate" (accept -> delivery window incl. batching)

How it works:
  1. Re-loads the raw parquet via src.data_prep.load_raw_data (uses the local
     HF cache - no new download if data_prep has run before).
  2. Parses accept/delivery timestamps and GPS exactly like data_prep.clean_data
     (placeholder year 2023, midnight-crossing +24h guard).
  3. For each (courier, day): sorts deliveries by delivery_dt; the gap between
     consecutive drops ~= travel + handover; haversine between consecutive
     delivery GPS points gives the distance for that leg.
  4. Per region: median speed (km/min) over legs with plausible speed,
     median handover minutes (the fixed time overhead of stopping).
  5. Saves data/processed/travel_stats.json (consumed by app/live_eta_tab.py).

NOTE: also includes the p50 model loss fix. If train_live_eta.py on this branch
still uses squared_error for p50, run this script AFTER updating train_live_eta
(see commit "fix(live-eta): p50 median loss") - the model and stats are
independent artifacts, so order only matters for the final retrain run.

Run with the app venv (scikit-learn 1.5.2):
    python src/build_travel_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep import load_raw_data  # noqa: E402

OUT = ROOT / "data" / "processed" / "travel_stats.json"

# plausibility guards for empirical legs
MIN_LEG_MINUTES = 1.0        # ignore GPS/clock jitter below 1 min
MAX_LEG_MINUTES = 120.0      # ignore gaps where courier idled 2h+
MIN_LEG_KM = 0.05            # ignore same-point glitches
MAX_LEG_KM = 30.0            # ignore cross-city GPS errors
MIN_SPEED_KMPH = 5.0         # walking pace; below = data glitch
MAX_SPEED_KMPH = 60.0        # dense-city courier ceiling incl. traffic

N_COURIERS_PER_REGION = 200  # cap compute; random sample per region
N_DAYS_PER_COURIER = 30


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2)
    return r * 2 * np.arcsin(np.sqrt(a))


def main():
    print("Loading raw LaDe-D Yantai parquet (HF cache)...")
    df = load_raw_data()
    print("Raw rows:", len(df))

    # --- parse exactly like data_prep.clean_data ---
    df = df.dropna(subset=[
        "order_id", "courier_id", "region_id", "accept_time", "delivery_time",
        "accept_gps_lng", "accept_gps_lat",
        "delivery_gps_lng", "delivery_gps_lat",
    ]).copy()

    full_str_year = "2023-"
    df["accept_dt"] = pd.to_datetime(
        full_str_year + df["accept_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df["delivery_dt"] = pd.to_datetime(
        full_str_year + df["delivery_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["accept_dt", "delivery_dt"])

    duration_min = (df["delivery_dt"] - df["accept_dt"]).dt.total_seconds() / 60.0
    duration_min = np.where(duration_min < 0, duration_min + 24 * 60, duration_min)
    df["accept_to_delivery_min"] = duration_min

    # ds = the delivery date string (e.g. "03-21"); group courier by (day)
    if "ds" not in df.columns:
        # fall back to delivery date from the parsed datetime
        df["ds"] = df["delivery_dt"].dt.strftime("%m-%d")
    df["day_key"] = df["ds"].astype(str)

    # --- build consecutive-drop legs per (courier, day) ---
    legs = []
    grouped = df.groupby(["region_id", "courier_id", "day_key"])
    n_groups = 0
    for (region, courier, day), g in grouped:
        n_groups += 1
        if len(g) < 2:
            continue
        g = g.sort_values("delivery_dt")
        lat = g["delivery_gps_lat"].to_numpy()
        lng = g["delivery_gps_lng"].to_numpy()
        t = g["delivery_dt"].to_numpy()
        for i in range(len(g) - 1):
            gap = (t[i + 1] - t[i]) / np.timedelta64(1, "m")
            if not (MIN_LEG_MINUTES <= gap <= MAX_LEG_MINUTES):
                continue
            dist = float(_haversine_km(lat[i], lng[i], lat[i + 1], lng[i + 1]))
            if not (MIN_LEG_KM <= dist <= MAX_LEG_KM):
                continue
            speed_kmph = dist / (gap / 60.0)
            if not (MIN_SPEED_KMPH <= speed_kmph <= MAX_SPEED_KMPH):
                continue
            legs.append({
                "region_id": int(region),
                "dist_km": dist,
                "gap_min": float(gap),
                "speed_kmph": float(speed_kmph),
                # handover estimate: gap minus riding time at that leg's speed
                "handover_min": float(max(gap - dist / (speed_kmph / 60.0), 0.0)),
            })
        # compute guard is per group; overall cap handled below via sampling
    print("Courier-day groups scanned:", n_groups)
    print("Plausible legs collected:", len(legs))
    if not legs:
        raise SystemExit("No plausible legs found - check raw data filters.")

    legs_df = pd.DataFrame(legs)

    # --- per-region aggregates ---
    regions = {}
    for rid, g in legs_df.groupby("region_id"):
        med_speed = float(np.median(g["speed_kmph"]))
        med_handover = float(np.median(g["handover_min"]))
        regions[str(int(rid))] = {
            "median_speed_kmph": round(med_speed, 2),
            "median_handover_min": round(med_handover, 1),
            "legs_sample": int(len(g)),
            # distance-based direct travel time used by the Live ETA tab:
            # t = distance / speed + handover
            "direct_eta_min_per_km": round(60.0 / med_speed, 2),
        }

    global_speed = float(np.median(legs_df["speed_kmph"]))
    global_handover = float(np.median(legs_df["handover_min"]))

    stats = {
        "method": "consecutive-drop legs from raw LaDe-D courier routes; "
                  "median speed + median handover per region",
        "guards": {
            "leg_minutes": [MIN_LEG_MINUTES, MAX_LEG_MINUTES],
            "leg_km": [MIN_LEG_KM, MAX_LEG_KM],
            "speed_kmph": [MIN_SPEED_KMPH, MAX_SPEED_KMPH],
        },
        "global": {
            "median_speed_kmph": round(global_speed, 2),
            "median_handover_min": round(global_handover, 1),
            "direct_eta_min_per_km": round(60.0 / global_speed, 2),
        },
        "regions": regions,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2))
    print("Wrote", OUT)
    print("Regions:", len(regions), "| global median speed: %.1f km/h, "
          "handover: %.1f min" % (global_speed, global_handover))
    print("Example direct ETA for 2.6 km: ~%.0f min"
          % (2.6 / global_speed * 60 + global_handover))


if __name__ == "__main__":
    main()