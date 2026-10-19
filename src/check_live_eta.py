"""
ShipSense - Module 6 check: does the compact live model agree with the full model?

Run with the app environment (venv):
    python src/check_live_eta.py

Uses the saved batch predictions (data/processed/predictions.csv, made by the
full model) and the same train/test split as training, so it does not need the
full model file. Writes nothing.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DELAY_RISK_THRESHOLD_MINUTES, RANDOM_STATE  # noqa: E402
from src.train_model import FEATURE_ORDER  # noqa: E402

DATA = ROOT / "data" / "processed" / "cleaned_data.csv"
PRED = ROOT / "data" / "processed" / "predictions.csv"
MODEL = ROOT / "models" / "eta_live_model.pkl"
TARGET = "delivery_duration_minutes"


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURE_ORDER]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    test = df.loc[X_test.index, ["order_id", "region_id", TARGET]].copy()
    test["live_eta"] = joblib.load(MODEL).predict(X_test)

    pred = pd.read_csv(
        PRED,
        usecols=["order_id", "predicted_eta_minutes", "promised_eta_minutes", "risk_status"],
    )
    merged = test.merge(pred, on="order_id", how="inner")
    print("test rows:", len(test), "| matched in predictions.csv:", len(merged))

    mae_full = mean_absolute_error(merged[TARGET], merged["predicted_eta_minutes"])
    mae_live = mean_absolute_error(merged[TARGET], merged["live_eta"])
    print("Full model (saved batch predictions): MAE %.2f min" % mae_full)
    print("Compact live model: MAE %.2f min" % mae_live)

    gap = (merged["live_eta"] - merged["predicted_eta_minutes"]).abs()
    print("Average gap between the two predicted ETAs: %.1f min (median %.1f)" % (
        gap.mean(), gap.median()))

    risk_live = (merged["live_eta"] - merged["promised_eta_minutes"]) > DELAY_RISK_THRESHOLD_MINUTES
    risk_full = merged["risk_status"] == "At Risk"
    agree = float((risk_live == risk_full).mean())
    print("At Risk share: live %.1f%%, full %.1f%%" % (
        100.0 * float(risk_live.mean()), 100.0 * float(risk_full.mean())))
    print("At Risk label agreement: %.1f%%" % (100.0 * agree))


if __name__ == "__main__":
    main()