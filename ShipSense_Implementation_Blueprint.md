# ShipSense — Implementation Blueprint (Days 2–10)

**Project:** ShipSense — AI-Powered Delivery Risk & ETA Intelligence Dashboard
**Context:** AB Talks 60-Day Claude AI Challenge — 10-Day Capstone
**Purpose of this document:** This is the single source of truth for building ShipSense from Day 2 through Day 10. Each day's section is self-contained enough that a fresh AI conversation can pick up exactly where the previous day left off, without re-deciding architecture or scope.

**Non-negotiable scope reminders (see PRD Section 5):**
- ONE ML model: ETA prediction (regression). Delay risk is a derived threshold rule, NOT a second model.
- No live weather/traffic APIs, no route optimization, no rider allocation, no dynamic pricing, no auth.
- Dataset: Hugging Face `Cainiao-AI/LaDe` (fallback: a smaller/simpler public delivery dataset if LaDe is too large — decided Day 2).
- Deployment target: Streamlit Community Cloud (free).
- Time budget: ~3–4 hours/day.

---

> **Day 2 Addendum (added after Day 2 execution):** Tech stack, architecture, data schema, function contracts, and UI wireframes are now formally locked in `docs/ARCHITECTURE.md`, `docs/SCHEMA.md`, `docs/API.md`, and `docs/UI-WIREFRAMES.md`. Day 3 onward should treat those documents as binding alongside this Blueprint — in particular, `src/data_prep.py` (Day 3), `src/train_model.py` (Day 4), and `src/predict.py` (Day 5) must match the exact function signatures defined in `docs/API.md`, and their outputs must match the exact schema defined in `docs/SCHEMA.md`. No architectural decisions remain open for Days 3-10.

## Day 2 — Design & Architecture

### 🎯 Objective
Finalize the tech stack, system architecture, dataset, and repository structure. No feature code yet — this is the blueprint-to-code bridge.

### 📖 What I'll learn
- How to structure a small ML product repo (data / model / app separation)
- How to evaluate a dataset for feasibility before committing to it
- Basics of Streamlit app architecture

### 🛠 Features to build
- None yet — this is a planning/setup day. Deliverable is a working repo skeleton + confirmed dataset.

### 📝 Step-by-step implementation plan
1. Load the Hugging Face `Cainiao-AI/LaDe` dataset card and inspect available columns, size, and format (`datasets` library: `load_dataset("Cainiao-AI/LaDe", ...)`, or manual download if gated).
2. If LaDe is too large (millions of rows) or too complex (multi-file, heavy preprocessing needed) for a 3–4 hr/day budget, select the fallback: a smaller public delivery dataset with clear delivery-time/distance/timestamp columns (search Hugging Face Datasets Hub for "delivery" or "logistics").
3. Decide final tech stack (recommended, but confirm in-session): Python, pandas, scikit-learn (regression model), Streamlit (frontend/dashboard), Plotly or Streamlit's native charts (visualization), joblib (model serialization).
4. Create the repository folder structure (see below).
5. Write a one-paragraph `README.md` stub describing the project (expand later).
6. Set up a Python virtual environment and a `requirements.txt` with pinned versions of: `streamlit`, `pandas`, `scikit-learn`, `numpy`, `plotly`, `datasets` (Hugging Face), `joblib`.
7. Do a smoke test: load a small sample of the dataset in a scratch script and print `.head()` and `.info()` to confirm it's usable.

### 📂 Files and folders to create or modify
```
shipsense/
├── data/
│   ├── raw/              # original downloaded dataset (gitignored if large)
│   └── processed/        # cleaned dataset used for training
├── models/
│   └── (trained model .pkl goes here on Day 4)
├── src/
│   ├── data_prep.py       # loading + cleaning functions
│   ├── train_model.py     # model training script (Day 4)
│   └── predict.py         # inference helper functions (Day 5)
├── app/
│   └── app.py             # Streamlit entrypoint (Day 6+)
├── notebooks/
│   └── exploration.ipynb  # optional scratch EDA
├── requirements.txt
├── README.md
└── .gitignore
```

### 🔗 APIs, libraries, services, or tools to integrate
- Hugging Face `datasets` library (to pull LaDe or fallback dataset)
- scikit-learn (regression)
- Streamlit (app framework)
- GitHub (for version control + Streamlit Cloud deployment source)

### 🧪 Testing tasks
- Confirm the dataset loads without errors and has at least: an origin/destination or distance feature, a timestamp or time-of-day feature, and an actual delivery duration/ETA field.
- Confirm `requirements.txt` installs cleanly in a fresh virtual environment.

### 🐞 Common issues and debugging tips
- **Hugging Face dataset requires login/gating:** check the dataset card for access instructions; if gated and slow to resolve, switch to the fallback dataset immediately rather than losing a day.
- **Dataset too large to load in memory:** use `streaming=True` in `load_dataset()` or download only a subset/split.
- **Missing columns needed for ETA prediction:** if the dataset lacks a clear "distance" or "duration" field, this is a hard blocker — switch datasets now, not later.

### ✅ End-of-day checklist
- [ ] Final dataset chosen and confirmed loadable
- [ ] Repo folder structure created
- [ ] `requirements.txt` written and installs cleanly
- [ ] `README.md` stub exists
- [ ] Smoke-test script prints sample data successfully

### 📸 Expected project state and screenshots to capture
- Terminal/notebook output showing dataset `.head()` and `.info()`
- Folder structure screenshot (e.g. VS Code explorer or `tree` command output)

### ➡️ Handoff notes for Day 3
State clearly: which dataset was finalized (LaDe or fallback, with exact name/source), what columns are available, and the confirmed tech stack. Day 3 will do full data cleaning and exploratory data analysis (EDA) using this exact dataset — do not re-evaluate dataset choice on Day 3 unless Day 2 surfaced a blocking issue.

---

> **Day 3 Setup Addendum (added after foundation work):** Development environment is confirmed working — Python **3.12.7** (not 3.14 — see `docs/SETUP.md` for why), virtual environment (`venv`), and all dependencies from `requirements.txt` installed and verified. `app/app.py` now has a working three-tab navigation shell. `src/config.py` holds shared constants (delay threshold, file paths, random seed) — the cleaning work below should import from it rather than hardcoding paths locally.

## Day 3 — Data Cleaning & Exploratory Data Analysis (EDA)

### 🎯 Objective
Clean the chosen dataset and understand its patterns well enough to engineer good features for the ETA model.

### 📖 What I'll learn
- Practical data cleaning (missing values, outliers, type conversions)
- How to explore relationships between features and delivery time
- How EDA findings directly inform feature engineering decisions

### 🛠 Features to build
- `src/data_prep.py`: functions to load, clean, and output a processed dataset ready for modeling.

### 📝 Step-by-step implementation plan
1. Load the raw dataset confirmed on Day 2 into a pandas DataFrame.
2. Check for missing values per column; decide per-column strategy (drop row, impute median/mode, or drop column if mostly empty).
3. Check and fix data types (timestamps parsed as datetime, numeric columns as numeric, not strings).
4. Identify and handle outliers in the target variable (delivery duration) — e.g. remove deliveries with impossible durations (negative, or absurdly long) using domain-reasonable thresholds or IQR-based filtering.
5. Derive basic time features from timestamps: hour of day, day of week, is_weekend.
6. If distance isn't a direct column but lat/long coordinates are available, calculate straight-line (haversine) distance between pickup and drop-off as a feature.
7. Run EDA: plot distribution of delivery duration, delivery duration vs. hour of day, delivery duration vs. distance, and correlation matrix of numeric features.
8. Save the cleaned, feature-augmented dataset to `data/processed/cleaned_data.csv`.
9. Write findings (2-3 sentences) into `README.md` under a "Data Insights" section — this becomes pitch-deck material later.

### 📂 Files and folders to create or modify
- `src/data_prep.py` (new)
- `notebooks/exploration.ipynb` (EDA plots)
- `data/processed/cleaned_data.csv` (output)
- `README.md` (append Data Insights section)

### 🔗 APIs, libraries, services, or tools to integrate
- pandas, numpy (cleaning)
- matplotlib/seaborn or plotly (EDA visualization)
- haversine formula (plain Python, no extra library needed) if lat/long distance calculation is required

### 🧪 Testing tasks
- Confirm cleaned dataset has zero nulls in critical columns (target variable, key features).
- Confirm no negative or zero delivery durations remain.
- Print `.describe()` on the cleaned dataset to sanity-check ranges.

### 🐞 Common issues and debugging tips
- **Timestamp parsing errors:** use `pd.to_datetime(..., errors='coerce')` then check how many rows became `NaT` — decide whether to drop or investigate format inconsistency.
- **Outlier removal too aggressive:** if IQR filtering removes >10% of data, loosen thresholds — don't over-clean a dataset into uselessness.
- **Haversine distance seems wrong:** double check whether lat/long columns are swapped or in the wrong units (degrees vs. radians).

### ✅ End-of-day checklist
- [ ] `data_prep.py` written with reusable cleaning functions
- [ ] Cleaned dataset saved to `data/processed/`
- [ ] At least 4 EDA plots produced and reviewed
- [ ] Key findings documented in README

### 📸 Expected project state and screenshots to capture
- EDA plots (distribution + correlation)
- Cleaned dataset `.head()` and `.describe()` output

### ➡️ Handoff notes for Day 4
State clearly: final list of features available for modeling (e.g. distance, hour_of_day, day_of_week, is_weekend, any categorical features), the target variable name, and the path to the cleaned CSV. Day 4 will train the regression model directly on this processed file — do not redo cleaning on Day 4.

---

## Day 4 — Model Training (ETA Prediction)

### 🎯 Objective
Train, evaluate, and save a regression model that predicts delivery ETA (duration) from the cleaned dataset.

### 📖 What I'll learn
- End-to-end supervised regression workflow: train/test split, baseline model, model comparison, evaluation metrics
- How to interpret MAE/RMSE in a business context (minutes of error, not just abstract numbers)
- Model serialization for later use in the app

### 🛠 Features to build
- `src/train_model.py`: trains and saves the ETA prediction model.
- A saved model artifact (`models/eta_model.pkl`).

### 📝 Step-by-step implementation plan
1. Load `data/processed/cleaned_data.csv`.
2. Define feature columns (X) and target column (y = delivery duration).
3. One-hot encode or label-encode any categorical features (e.g. area/zone, weather category if present).
4. Split into train/test sets (e.g. 80/20), using a fixed `random_state` for reproducibility.
5. Train a **baseline model** first: simple Linear Regression — this gives an honest reference point.
6. Train a **stronger model**: Random Forest Regressor or Gradient Boosting Regressor (scikit-learn) — tree-based models handle mixed feature types well and are easy to explain.
7. Evaluate both models on the test set using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error), expressed in **minutes** for interpretability.
8. Pick the better-performing model as the final one (document both scores for the pitch deck's "technical approach" credibility).
9. Extract and save feature importances from the chosen model (used later for the "top delay factors" feature in the Executive Summary).
10. Save the trained model, the fitted encoders/scalers, and the feature column order using `joblib.dump()` into `models/`.
11. Write the final MAE/RMSE and a 2-sentence explanation of what the model learned into `README.md`.

### 📂 Files and folders to create or modify
- `src/train_model.py` (new)
- `models/eta_model.pkl`, `models/encoders.pkl` (new, saved artifacts)
- `README.md` (append Model Performance section)

### 🔗 APIs, libraries, services, or tools to integrate
- scikit-learn (`LinearRegression`, `RandomForestRegressor` or `GradientBoostingRegressor`, `train_test_split`, `mean_absolute_error`, `mean_squared_error`)
- joblib (model persistence)

### 🧪 Testing tasks
- Confirm MAE is in a believable range (e.g. within a reasonable number of minutes relative to average delivery duration — not near-zero, which would suggest data leakage).
- Manually predict on 3-5 sample rows and sanity-check the predicted ETA looks plausible.
- Confirm the saved `.pkl` file reloads correctly and produces identical predictions to the in-memory model.

### 🐞 Common issues and debugging tips
- **Suspiciously perfect accuracy (near-zero error):** likely data leakage — check if the target variable or a proxy for it accidentally ended up in the features (e.g. actual arrival timestamp used to derive both target and a feature).
- **Model performs worse than baseline linear regression:** check for feature scaling issues, misencoded categoricals, or too few meaningful features — don't be afraid to keep linear regression if it's genuinely the better/more explainable choice.
- **`joblib` version mismatch on reload:** pin scikit-learn and joblib versions in `requirements.txt` to match what was used for training.

### ✅ End-of-day checklist
- [ ] Baseline and stronger model both trained and evaluated
- [ ] Final model selected and documented with MAE/RMSE in minutes
- [ ] Feature importances extracted and saved
- [ ] Model artifacts saved to `models/` and reload-tested

### 📸 Expected project state and screenshots to capture
- Console output showing MAE/RMSE for both models
- Feature importance bar chart

### ➡️ Handoff notes for Day 5
State clearly: which model was chosen, its MAE/RMSE, the exact feature column order the model expects, and the paths to the saved `.pkl` files. Day 5 builds the inference + delay-risk logic directly on top of this saved model — do not retrain on Day 5 unless Day 4's model was fundamentally broken.

---

## Day 5 — Inference Pipeline & Delay Risk Logic

### 🎯 Objective
Build a clean, reusable inference function that takes new/existing order data, predicts ETA using the saved model, and derives the Delay Risk Flag.

### 📖 What I'll learn
- How to wrap a trained ML model into a reusable prediction function for an application
- How to design a simple, explainable business rule (delay risk threshold) on top of a model output
- Defensive coding for real-world messy inputs

### 🛠 Features to build
- `src/predict.py`: `predict_eta(order_data)` and `flag_delay_risk(predicted_eta, promised_eta, threshold)` functions.
- A precomputed predictions file (`data/processed/predictions.csv`) applying the model to the full cleaned dataset — this powers the dashboard without needing to re-run the model live for every page load.

### 📝 Step-by-step implementation plan
1. Write `load_model()` in `predict.py` to load the saved model + encoders from `models/`.
2. Write `predict_eta(df)`: takes a DataFrame of orders (already feature-engineered in the same way as training), applies encoders, and returns predicted ETA per row.
3. Decide the "promised/SLA time" definition: if the dataset has an explicit promised-time column, use it; if not, define a reasonable synthetic SLA (e.g. median historical delivery time for similar orders, or a fixed business rule like "distance-based standard time"). Document this decision clearly — it must be explainable in the demo.
4. Write `flag_delay_risk(predicted_eta, promised_eta, threshold_minutes=10)`: returns `"At Risk"` if `predicted_eta > promised_eta + threshold_minutes`, else `"On Track"`. Make the threshold a configurable constant.
5. Apply this pipeline to the entire cleaned dataset (simulating "today's orders" for demo purposes) and save the result — including original features, predicted ETA, promised ETA, and risk flag — to `data/processed/predictions.csv`.
6. Compute and store the top contributing factors for at-risk orders (using feature importances from Day 4) to support the Executive Summary's "top delay causes" feature.
7. Spot-check 10 random rows manually to confirm the risk flag logic makes sense (e.g. orders with predicted ETA far exceeding promised time are indeed flagged).

### 📂 Files and folders to create or modify
- `src/predict.py` (new)
- `data/processed/predictions.csv` (new — this is what the Streamlit app will read on Day 6+)

### 🔗 APIs, libraries, services, or tools to integrate
- joblib (loading saved model)
- pandas (data manipulation)

### 🧪 Testing tasks
- Confirm `predictions.csv` has no nulls in predicted ETA or risk flag columns.
- Confirm the percentage of orders flagged "At Risk" is reasonable (e.g. not 0% or 100% — if so, the threshold or SLA definition needs adjusting).
- Unit-test `flag_delay_risk()` with a few hand-crafted edge cases (exactly at threshold, well below, well above).

### 🐞 Common issues and debugging tips
- **Feature mismatch error when loading model:** ensure the exact same column order and encoding used in training is replicated here — save the feature list from Day 4 and reuse it explicitly.
- **Every order flagged as at-risk (or none):** the SLA/promised-time definition or threshold is likely miscalibrated — revisit Step 3 and adjust to produce a believable, demo-worthy risk distribution (e.g. roughly 15-30% at-risk is a realistic and useful range).

### ✅ End-of-day checklist
- [ ] `predict.py` written with working `predict_eta()` and `flag_delay_risk()` functions
- [ ] `predictions.csv` generated for the full dataset
- [ ] Risk flag distribution manually sanity-checked
- [ ] Top delay factors computed and saved

### 📸 Expected project state and screenshots to capture
- Sample rows of `predictions.csv` showing predicted ETA, promised ETA, and risk flag side by side
- Console output showing % of orders flagged at-risk

### ➡️ Handoff notes for Day 6
State clearly: the exact schema (column names) of `predictions.csv`, the SLA/promised-time definition used, the risk threshold value, and where the top-delay-factors data lives. Day 6 builds the Streamlit Operations Dashboard reading directly from this file.

---

## Day 6 — Streamlit App: Operations Dashboard (Core UI)

### 🎯 Objective
Build the primary, hero screen of ShipSense: a functional Operations Dashboard that ops managers would actually want to use.

### 📖 What I'll learn
- Streamlit fundamentals: layout, widgets, dataframes, filtering
- How to structure a multi-page/multi-section Streamlit app
- Basic UI/UX thinking for a data-heavy operational tool

### 🛠 Features to build
- `app/app.py`: Streamlit entrypoint with page config and navigation.
- Operations Dashboard section: filterable table of deliveries with predicted ETA, promised ETA, and risk flag.

### 📝 Step-by-step implementation plan
1. Initialize `app/app.py` with `st.set_page_config(page_title="ShipSense", layout="wide")`.
2. Load `data/processed/predictions.csv` using `st.cache_data` to avoid reloading on every interaction.
3. Build a sidebar or top-of-page filter section: filter by area/zone (if available), by risk status (All / At Risk / On Track), and by time window (if hour/date data available).
4. Display the filtered results in a `st.dataframe()` with clear column headers (Order ID, Predicted ETA, Promised ETA, Risk Status, key contextual features).
5. Color-code or icon-flag the risk status column for quick visual scanning (e.g. 🔴 At Risk / 🟢 On Track using a formatted column or `st.dataframe` styling).
6. Add a small summary strip at the top of the dashboard: total orders shown, count/percentage at-risk within current filter.
7. Add basic page navigation (sidebar radio or `st.tabs`) with placeholders for "Operations", "Demand Heatmap", and "Executive Summary" — the latter two get built out on Day 7.
8. Test the app locally with `streamlit run app/app.py`.

### 📂 Files and folders to create or modify
- `app/app.py` (new — core file)
- Possibly `app/utils.py` if filtering logic gets reused across pages

### 🔗 APIs, libraries, services, or tools to integrate
- Streamlit (`st.dataframe`, `st.sidebar`, `st.tabs`, `st.cache_data`, `st.metric`)

### 🧪 Testing tasks
- Test each filter combination individually and in combination (area + risk status + time window).
- Confirm the dashboard doesn't crash with an empty filter result (e.g. show a friendly "no results" message).
- Test on a smaller browser window to check basic responsiveness.

### 🐞 Common issues and debugging tips
- **App reloads slowly on every filter change:** ensure `st.cache_data` is applied to the data-loading function, not recomputed every time.
- **Dataframe too wide / columns cut off:** use `layout="wide"` in page config and consider hiding less critical columns behind an "Advanced" expander.
- **Color coding not rendering:** Streamlit's native dataframe styling has limits — a simple emoji-prefixed text column (e.g. "🔴 At Risk") is a reliable, low-risk approach vs. complex conditional styling.

### ✅ End-of-day checklist
- [ ] `app.py` runs locally without errors
- [ ] Filters work correctly individually and combined
- [ ] Risk status is visually distinguishable at a glance
- [ ] Summary strip shows correct counts

### 📸 Expected project state and screenshots to capture
- Screenshot of the full Operations Dashboard with filters applied
- Screenshot of the "no results" empty state

### ➡️ Handoff notes for Day 7
State clearly: the current file structure of `app/app.py`, the filter widget variable names, and the placeholder locations for Demand Heatmap and Executive Summary tabs/pages. Day 7 fills in those two remaining views — do not restructure the Operations Dashboard unless a bug was found.

---

## Day 7 — Streamlit App: Demand Heatmap & Executive Summary

### 🎯 Objective
Complete the remaining two views: the Demand Heatmap (aggregation-based) and the Executive Summary (KPI-based), making ShipSense feature-complete.

### 📖 What I'll learn
- Data aggregation and grouping in pandas for visualization
- Building simple, effective charts in Streamlit (bar/heatmap-style)
- Translating model outputs into executive-level KPIs

### 🛠 Features to build
- Demand Heatmap view: order density by area and hour.
- Executive Summary view: KPI cards + top delay factors.

### 📝 Step-by-step implementation plan
1. **Demand Heatmap:** Using `predictions.csv` (or the cleaned dataset), group orders by hour-of-day and area/zone (or just hour-of-day and day-of-week if no geographic zone column exists). Build a heatmap-style visualization — `st.bar_chart` for a simple version, or a Plotly heatmap (`plotly.express.imshow` or `density_heatmap`) for a richer visual.
2. Add a brief caption explaining what the heatmap shows and why it matters (e.g. "Peak order density by hour helps anticipate staffing needs").
3. **Executive Summary:** Build KPI cards using `st.metric()` for: total orders, average predicted ETA, % at-risk deliveries, and average delay (predicted minus promised, for at-risk orders only).
4. Add a small bar chart of "Top Factors Correlated with Delay" using the feature importances saved on Day 5.
5. Add a short narrative text block (2-3 sentences, auto-generated from the computed stats) summarizing today's "delivery health" — this adds a nice executive-facing touch.
6. Wire both views into the navigation structure built on Day 6 (tabs or sidebar pages).
7. Full end-to-end manual walkthrough: navigate between all three views, confirm no errors, confirm layout looks intentional (not cramped or empty).

### 📂 Files and folders to create or modify
- `app/app.py` (extend with the two new views/sections)
- Possibly `app/charts.py` if chart-building logic is broken out for cleanliness

### 🔗 APIs, libraries, services, or tools to integrate
- Plotly Express (`plotly.express`) for heatmap/bar charts, or Streamlit-native charts if simplicity is preferred
- pandas `.groupby()` for aggregation

### 🧪 Testing tasks
- Confirm heatmap renders correctly with realistic-looking density patterns (not flat/uniform, which would suggest a grouping bug).
- Confirm KPI numbers update correctly if filters from the Operations Dashboard are (optionally) shared across views, or confirm they're intentionally independent/global — document whichever approach was chosen.
- Cross-check one KPI manually against a pandas calculation to confirm correctness (e.g. manually compute % at-risk and compare to what's displayed).

### 🐞 Common issues and debugging tips
- **Heatmap looks flat/uninteresting:** check the groupby granularity — too coarse (e.g. just day-of-week) may lose the interesting hourly pattern; too fine may fragment the data too much.
- **KPI cards showing NaN:** likely a division-by-zero when the filtered dataset is empty — add a guard clause showing a friendly message instead.
- **Layout feels unbalanced:** use `st.columns()` to arrange KPI cards side by side rather than stacked vertically.

### ✅ End-of-day checklist
- [ ] Demand Heatmap renders with realistic patterns
- [ ] Executive Summary KPIs are correct and cross-checked manually
- [ ] Top delay factors chart displays correctly
- [ ] All three views (Operations, Heatmap, Executive Summary) navigate cleanly with no errors

### 📸 Expected project state and screenshots to capture
- Screenshot of Demand Heatmap view
- Screenshot of Executive Summary view with KPI cards

### ➡️ Handoff notes for Day 8
State clearly: the app is now feature-complete locally. Confirm the app runs with `streamlit run app/app.py` with zero errors across all views. Day 8 focuses entirely on deployment — no new features should be added on Day 8 unless a bug is discovered.

---

## Day 8 — Deployment (Streamlit Community Cloud)

### 🎯 Objective
Deploy ShipSense publicly so it's accessible via a shareable link, with buffer days remaining for polish.

### 📖 What I'll learn
- How to prepare a Python app for cloud deployment
- Git/GitHub basics for deployment workflows
- Common deployment troubleshooting (dependency/version issues)

### 🛠 Features to build
- None — this is a deployment/infra day, not a feature day.

### 📝 Step-by-step implementation plan
1. Ensure `requirements.txt` is complete and pinned (test in a fresh virtual environment locally one more time before deploying).
2. Ensure large files (raw datasets, if bulky) are excluded via `.gitignore` — the app should only need `data/processed/predictions.csv` (and any other small processed files) at runtime, not the full raw dataset.
3. Initialize a Git repository (if not already done) and push the project to a public GitHub repository.
4. **Manual step (guided):** Go to Streamlit Community Cloud, sign in with GitHub, click "New app," select the repository, branch, and set the main file path to `app/app.py`, then deploy.
5. Watch the build logs for dependency errors; fix and redeploy as needed (this can take 2-3 iterations — expected, not a failure).
6. Once live, test the public URL on both desktop and mobile browser to confirm it renders acceptably.
7. Add the live URL to `README.md` and to the top of the Streamlit app itself (e.g. a small caption).

### 📂 Files and folders to create or modify
- `.gitignore` (finalize — exclude raw data, virtual environment folders, `.pkl` files only if too large for GitHub's limits, in which case regenerate them via a startup script instead)
- `README.md` (add live URL, deployment instructions)

### 🔗 APIs, libraries, services, or tools to integrate
- GitHub (source control + deployment source)
- Streamlit Community Cloud (free hosting)

### 🧪 Testing tasks
- Test the live public link in an incognito/private browser window (simulates a first-time visitor with no cached state).
- Test on a mobile device or narrow browser window for basic responsiveness.
- Confirm the app doesn't silently fail if a file path assumption (e.g. relative path) breaks in the cloud environment — use paths relative to the script's location, not the local machine's absolute paths.

### 🐞 Common issues and debugging tips
- **"File not found" errors in the cloud but not locally:** almost always a relative-vs-absolute path issue; use `os.path.dirname(__file__)`-based paths.
- **Dependency version conflicts:** pin exact versions in `requirements.txt` that match what was tested locally.
- **Large model or data file exceeds GitHub's size limits:** consider retraining the model at app startup with a cached decorator, or use Git LFS, or reduce the dataset subset size used in `predictions.csv`.
- **App works but looks broken (blank charts):** check Plotly/Streamlit version compatibility between local and cloud requirements.

### ✅ End-of-day checklist
- [ ] Code pushed to a public GitHub repository
- [ ] App successfully deployed and live on Streamlit Community Cloud
- [ ] Public URL tested in incognito mode and on mobile
- [ ] README updated with live link

### 📸 Expected project state and screenshots to capture
- Screenshot of the Streamlit Cloud deployment dashboard showing "Running" status
- Screenshot of the live public app in a browser

### ➡️ Handoff notes for Day 9
State clearly: the live public URL, and confirm whether deployment is fully stable or if any known minor issues remain. Day 9 is polish and testing — no major new features, only bug fixes and UX refinement.

---

## Day 9 — Testing, Polish & Documentation

### 🎯 Objective
Harden the deployed app, fix any remaining rough edges, and finalize all documentation so the project reads as complete and professional.

### 📖 What I'll learn
- Structured manual QA/testing approach for a small data app
- How to write clear technical documentation for a portfolio project
- UX polish techniques (empty states, loading states, copy clarity)

### 🛠 Features to build
- No new core features — only bug fixes, UX polish, and edge-case handling.

### 📝 Step-by-step implementation plan
1. Do a full manual QA pass across all three views using the checklist below (Testing Tasks).
2. Fix any bugs found — prioritize anything that would break during a live demo (crashes, blank screens, broken filters).
3. Polish UX details: add a short intro/description at the top of the app explaining what ShipSense does in 1-2 sentences; add helpful captions under charts; ensure consistent number formatting (e.g. ETA in minutes, percentages to 1 decimal place).
4. Add a simple loading spinner or `st.spinner()` around any slower operations (e.g. initial data load).
5. Finalize `README.md`: project description, problem statement, tech stack, how to run locally, live demo link, model performance summary, and screenshots.
6. Re-verify the model's MAE/RMSE numbers are correctly and consistently reported everywhere they're mentioned (README, app, and — tomorrow — the pitch deck).
7. Do a final "fresh eyes" walkthrough: pretend to be a first-time user with no context and see if anything is confusing.

### 📂 Files and folders to create or modify
- `app/app.py` (bug fixes, polish)
- `README.md` (finalized, complete version)

### 🔗 APIs, libraries, services, or tools to integrate
- None new — this day is refinement only.

### 🧪 Testing tasks
- [ ] Every filter combination tested on the live deployed app (not just locally)
- [ ] Empty-state and edge-case behavior confirmed (e.g. all filters selected to an impossible combination)
- [ ] All numbers displayed (KPIs, ETAs, percentages) manually cross-checked for correctness
- [ ] App tested after a hard refresh / in a new incognito session
- [ ] Spelling/grammar check on all visible text and captions

### 🐞 Common issues and debugging tips
- **Small UI inconsistencies (e.g. inconsistent decimal places):** create a single formatting helper function and use it everywhere numbers are displayed, rather than fixing each instance separately.
- **App feels "empty" or under-explained:** a first-time visitor with zero context should be able to understand the app's purpose within 5 seconds — if not, add a clearer header/intro.
- **Forgot to update README after Day 8 deployment changes:** treat README as a living document updated at the end of every remaining day, not just today.

### ✅ End-of-day checklist
- [ ] All known bugs fixed and re-tested on the live deployment
- [ ] README fully finalized with live link, setup instructions, and model performance
- [ ] UX polish pass complete (intro text, consistent formatting, captions)
- [ ] Fresh-eyes walkthrough completed with no major confusion points

### 📸 Expected project state and screenshots to capture
- Screenshot of the finalized README (rendered on GitHub)
- Screenshots of the polished app (all three views, final state)

### ➡️ Handoff notes for Day 10
State clearly: the app is stable, deployed, and documented. Day 10 is entirely about final presentation — reviewing the pitch deck, rehearsing the demo narrative, and doing a last end-to-end sanity check. No code changes should be needed unless something breaks unexpectedly.

---

## Day 10 — Final Review, Demo Rehearsal & Submission

### 🎯 Objective
Confirm the entire product, documentation, and pitch deck are complete, coherent, and ready to present. Ship it.

### 📖 What I'll learn
- How to present a technical project persuasively to a non-technical or mixed audience
- How to structure a confident product demo narrative
- Final QA discipline before a deadline

### 🛠 Features to build
- None — this is a review, rehearsal, and submission day.

### 📝 Step-by-step implementation plan
1. Do one final end-to-end test of the live deployed app — confirm it's still running (Streamlit Cloud apps can occasionally sleep after inactivity; wake it up and confirm it loads correctly).
2. Review the Pitch Deck (generated Day 1) against the actual finished product — update any slide where the plan changed during the build (e.g. actual MAE/RMSE numbers, actual dataset used, any scope adjustments).
3. Rehearse a 3-5 minute demo narrative: Problem → who it's for → live demo walkthrough (Operations Dashboard → Heatmap → Executive Summary) → technical approach (one sentence on the model) → future scope (one sentence).
4. Prepare for likely questions: "How accurate is the model?", "Why this dataset?", "What would you build next?", "What was the hardest part?"
5. Do a final review of the GitHub repository: clean, no leftover scratch files, README renders well, no exposed secrets or credentials.
6. Package/organize final submission materials as required by the AB Talks Challenge (confirm exact submission format/requirements separately, outside this technical blueprint).
7. Take final screenshots/recording of the live app for submission or LinkedIn documentation, consistent with the "documenting the journey" content style already established.

### 📂 Files and folders to create or modify
- Pitch deck (final edits only, based on Day 1 draft)
- `README.md` (final read-through, no changes expected unless an error is found)

### 🔗 APIs, libraries, services, or tools to integrate
- None new.

### 🧪 Testing tasks
- Full live-app walkthrough exactly as it will be demoed, timed to fit the presentation window.
- Confirm the public link works from a completely different network/device if possible (e.g. phone on mobile data) to rule out local-network-only issues.

### 🐞 Common issues and debugging tips
- **App is "asleep" on Streamlit Cloud after inactivity:** visit the link a few minutes before any live demo to wake it up in advance.
- **Pitch deck numbers don't match the live app:** this is the most common last-day inconsistency — do a side-by-side check of every number mentioned in both places.

### ✅ End-of-day checklist
- [ ] Live app confirmed working end-to-end
- [ ] Pitch deck updated to match final, actual product details
- [ ] Demo narrative rehearsed at least once, timed
- [ ] GitHub repo clean and presentable
- [ ] Final screenshots/recording captured

### 📸 Expected project state and screenshots to capture
- Final screenshots of all three app views for submission/portfolio use
- Screenshot of the GitHub repository main page

### ➡️ Handoff notes
This is the final day — no further handoff needed. ShipSense v1.0 is complete, deployed, documented, and demo-ready.

---

## Appendix: Standing Rules for Every Remaining Day

- Do not reintroduce any feature listed as "Out of Scope" in the PRD (Section 5.2) without an explicit, deliberate scope-change conversation acknowledging the time trade-off.
- Every day should end with the app in a **working state** — never leave it broken overnight if avoidable within the day's time budget.
- If a day's plan turns out to be too ambitious for the actual time available, cut polish, not correctness — a plain but working feature beats a broken polished one.
- Keep `README.md` updated incrementally each day rather than all at once on Day 9 — it also doubles as a running log for the "documenting the journey" content style.
