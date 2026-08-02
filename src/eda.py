"""
ShipSense — Exploratory Data Analysis
Reads data/processed/cleaned_data.csv and produces:
  - Console summary statistics
  - 4 plots saved to docs/eda/ as PNG files
Run after src/data_prep.py has produced cleaned_data.csv.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import os

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_DIR = "docs/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}\n")

print("--- describe() ---")
print(df.describe().to_string())
print()

print("--- Correlation with delivery_duration_minutes ---")
numeric_cols = df.select_dtypes(include="number").columns
corr = df[numeric_cols].corr()["delivery_duration_minutes"].sort_values(ascending=False)
print(corr.to_string())
print()

# --- Plot 1: Distribution of delivery duration ---
plt.figure(figsize=(8, 5))
plt.hist(df["delivery_duration_minutes"], bins=50, color="#065A82", edgecolor="white")
plt.title("Distribution of Delivery Duration (minutes)")
plt.xlabel("Delivery Duration (minutes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_duration_distribution.png", dpi=120)
plt.close()

# --- Plot 2: Duration vs hour of day ---
plt.figure(figsize=(8, 5))
hourly_avg = df.groupby("hour_of_day")["delivery_duration_minutes"].mean()
plt.bar(hourly_avg.index, hourly_avg.values, color="#1C7293")
plt.title("Average Delivery Duration by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Delivery Duration (minutes)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_duration_by_hour.png", dpi=120)
plt.close()

# --- Plot 3: Duration vs distance (scatter, sampled for speed) ---
sample = df.sample(min(5000, len(df)), random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["distance_km"], sample["delivery_duration_minutes"],
            alpha=0.3, s=8, color="#21295C")
plt.title("Delivery Duration vs. Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Duration (minutes)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_duration_vs_distance.png", dpi=120)
plt.close()

# --- Plot 4: Correlation heatmap ---
plt.figure(figsize=(7, 6))
corr_matrix = df[numeric_cols].corr()
plt.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=120)
plt.close()

print(f"Saved 4 EDA plots to {OUTPUT_DIR}/")
print("\n--- Key Insights ---")
strongest = corr.drop("delivery_duration_minutes").abs().idxmax()
print(f"Strongest correlated feature with delivery duration: '{strongest}' "
      f"(r={corr[strongest]:.3f})")
print(f"Mean delivery duration: {df['delivery_duration_minutes'].mean():.1f} minutes")
print(f"Median delivery duration: {df['delivery_duration_minutes'].median():.1f} minutes")