\# ShipSense — Data Cleaning \& EDA Summary



Dataset: Cainiao-AI/LaDe-D, Yantai city subset

Raw rows: 206,431 -> Cleaned rows: 197,524



Cleaning: dropped 2,422 null-GPS rows, 6,011 duration outliers,

474 near-zero-distance rows.



Features engineered: distance\_km (haversine), hour\_of\_day,

day\_of\_week, is\_weekend.



KEY FINDING: distance\_km has almost no correlation with delivery

duration (r=-0.005). hour\_of\_day is the strongest signal (r=-0.277),

still weak. Likely because accept\_time reflects when a courier accepts

their full route/batch, not per-package -- duration reflects route

queue position more than raw distance. This means Day 4's model should

lean on courier\_id/region\_id/hour\_of\_day interactions, and a tree-based

model (Random Forest/Gradient Boosting) is expected to beat Linear

Regression meaningfully.



Mean delivery duration: 233.1 min | Median: 201.0 min



Files produced:

\- src/data\_prep.py

\- src/eda.py

\- data/processed/cleaned\_data.csv (197,524 rows x 15 cols)

\- docs/eda/\*.png (4 plots)

