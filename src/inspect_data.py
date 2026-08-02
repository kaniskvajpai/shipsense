"""
One-time inspection script — downloads the Yantai delivery dataset
(smallest city in LaDe-D) and prints its structure so we can confirm
exact column names/types before writing the full cleaning pipeline.

Run once, then discard (not part of the production pipeline).
"""

from huggingface_hub import hf_hub_download
import pandas as pd

print("Downloading delivery_yt.parquet from Cainiao-AI/LaDe-D ...")

path = hf_hub_download(
    repo_id="Cainiao-AI/LaDe-D",
    repo_type="dataset",
    filename="data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet",
)

print(f"Downloaded to: {path}")

df = pd.read_parquet(path)

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS & DTYPES ---")
print(df.dtypes)

print("\n--- FIRST 3 ROWS ---")
print(df.head(3).to_string())

print("\n--- NULL COUNTS ---")
print(df.isnull().sum())
