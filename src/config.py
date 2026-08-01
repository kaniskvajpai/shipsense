"""
Shared configuration constants for ShipSense.
Centralized here so every module (data_prep, train_model, predict, app)
references the same values — no magic numbers scattered across files.
"""

# Delay risk threshold, in minutes (see docs/API.md — flag_delay_risk)
DELAY_RISK_THRESHOLD_MINUTES = 10

# File paths (relative to project root)
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"
PREDICTIONS_PATH = "data/processed/predictions.csv"
MODELS_DIR = "models"

# Reproducibility
RANDOM_STATE = 42