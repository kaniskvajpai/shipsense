# 🚚 ShipSense

**AI-powered delivery risk & ETA intelligence dashboard.**

🔗 **Live app:** [shipsense-7dggxhyfrhmphun2zyehyt.streamlit.app](https://shipsense-7dggxhyfrhmphun2zyehyt.streamlit.app)
📦 **Version:** v1.0.0
🛠️ Built with [Claude](https://claude.com) as part of the [AB Talks 60-Day Claude AI Challenge](https://www.abtalks.in/)

---

## What It Does

ShipSense predicts how long a delivery will actually take, flags orders that are likely to run late *before* they do, and gives ops managers a live dashboard to act on it — built end-to-end on 197,524 real last-mile delivery records.

- 🎯 **ETA Prediction** — Random Forest model, MAE 70.85 minutes (40% more accurate than a linear baseline)
- ⚠️ **Delay Risk Flag** — explainable threshold rule (predicted vs. promised time), not a black box
- 📊 **Operations Dashboard** — filterable table of every delivery, sorted by urgency
- 🗺️ **Demand Heatmap** — order density by hour and region
- 📈 **Executive Summary** — KPIs, risk breakdown, and top delay factors

## Screenshots

*(See `docs/` for full architecture diagrams and design docs.)*

## Tech Stack

| Layer | Choice |
|---|---|
| ML | scikit-learn (Random Forest + Linear Regression baseline) |
| App | Streamlit |
| Data | pandas, Hugging Face `datasets` |
| Charts | Plotly |
| Hosting | Streamlit Community Cloud (free tier) |
| Dataset | [Cainiao-AI/LaDe-D](https://huggingface.co/datasets/Cainiao-AI/LaDe-D) (Yantai city subset) |

No database, no backend server, no auth — a deliberate architecture decision. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for why.

## Key Finding

Distance alone has almost zero *linear* correlation with delivery duration (r = -0.005) — yet it's the model's #1 feature by importance (36.6%). The relationship is real, just non-linear: distance's effect depends on courier, region, and time of day. This is why the Random Forest beats the linear baseline by 40%. Full analysis in [`docs/DAY3-DATA-SUMMARY.md`](docs/DAY3-DATA-SUMMARY.md).

## Run It Locally

```bash
git clone https://github.com/kaniskvajpai/shipsense.git
cd shipsense
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app/app.py
```

Full setup guide: [`docs/SETUP.md`](docs/SETUP.md)

## Project Documentation

| Doc | Purpose |
|---|---|
| [PRD](ShipSense_PRD.docx) | Product requirements |
| [Architecture](docs/ARCHITECTURE.md) | System design, diagrams |
| [Schema](docs/SCHEMA.md) | Data model |
| [API](docs/API.md) | Internal function contracts |
| [UI Wireframes](docs/UI-WIREFRAMES.md) | User flow & screens |
| [Future Scope](future-scope.md) | Roadmap: 3/6/12 months |
| [Retrospective](challenge-retrospective.md) | Build journey, Day 1–10 |

## Model Performance

| Model | MAE (minutes) | RMSE (minutes) |
|---|---|---|
| Linear Regression (baseline) | 117.19 | 144.50 |
| **Random Forest (chosen)** | **70.85** | **98.81** |

Trained on 158,019 rows, tested on 39,505 held-out rows.

## Scope

**In v1.0:** ETA prediction, delay risk flag, 3-view dashboard, public deployment.
**Explicitly out (see [`future-scope.md`](future-scope.md)):** live weather/traffic, route optimization, rider allocation, dynamic pricing, auth.

## Credits

Built solo by [Kanishk Bajpai](https://github.com/kaniskvajpai) using [Claude](https://claude.com) as an AI pair programmer, as the capstone project for the [AB Talks 60-Day Claude AI Challenge](https://www.abtalks.in/) ([@ABTalksOnAI](https://www.linkedin.com/company/abtalks-on-ai/)).

## License

MIT — see [LICENSE](LICENSE).
