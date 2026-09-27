# ShipSense — 30-Day Growth Plan

A realistic, one-milestone-per-day roadmap taking ShipSense from MVP to a significantly more complete product. Each day builds on the previous — do them in order. Pulled directly from `future-scope.md`'s prioritization.

## Week 1: Trustworthy Core (Days 1–7)
**Theme: Fix the honesty gaps before adding anything new.**

- **Day 1:** Audit every "explainable" claim in the app against what's actually computed. Document any gap between what the UI implies and what the code does.
- **Day 2:** Research real-world SLA/promised-time data sources or conventions for last-mile delivery; design a better SLA model than the regional-75th-percentile proxy.
- **Day 3:** Implement SHAP (or a simpler per-row feature-contribution method) for the Random Forest model.
- **Day 4:** Replace the global "top delay factor" with per-order SHAP-based explanations in `predict.py`.
- **Day 5:** Update the Operations Dashboard to show per-order explanations instead of the single global factor.
- **Day 6:** Write tests for `data_prep.py`, `train_model.py`, and `predict.py` (unit tests for the function contracts in `docs/API.md`).
- **Day 7:** Review week 1 end-to-end; update `docs/SCHEMA.md` and `docs/API.md` to reflect the new SHAP-based fields.

## Week 2: Multi-City & Monitoring (Days 8–14)
**Theme: Prove the model generalizes; know when it stops working.**

- **Day 8:** Download a second city from LaDe-D (e.g. Jilin); run the existing `data_prep.py` pipeline against it unmodified.
- **Day 9:** Train a second model on the new city; compare MAE/RMSE to the Yantai model.
- **Day 10:** Decide: one model per city, or one combined model with city as a feature? Document the decision and why.
- **Day 11:** Implement the chosen multi-city approach in `train_model.py`.
- **Day 12:** Add a simple drift-detection check: compare live prediction error (if ground truth becomes available) against the training-time MAE baseline.
- **Day 13:** Add a "model health" section to the Executive Summary showing current vs. baseline accuracy.
- **Day 14:** Full regression test across all cities; update README with multi-city support.

## Week 3: Proactive Alerts & Recommendations (Days 15–21)
**Theme: Move from "you have to check" to "it tells you."**

- **Day 15:** Design the alerting logic — what conditions trigger a notification (e.g. new at-risk order, risk % crossing a threshold).
- **Day 16:** Set up a free email or Slack webhook integration (e.g. Slack Incoming Webhooks, free tier).
- **Day 17:** Implement the alert-triggering logic as a scheduled check (e.g. via GitHub Actions cron, free tier).
- **Day 18:** Test alerts end-to-end with real data; tune thresholds to avoid alert fatigue.
- **Day 19:** Design the "nearby courier with capacity" recommendation logic (simple proximity + workload heuristic, not full optimization).
- **Day 20:** Implement the recommendation as a new column/panel in the Operations Dashboard.
- **Day 21:** User-test the alerting + recommendation flow; document in README.

## Week 4: Toward Multi-Tenant (Days 22–30)
**Theme: Make this deployable by someone who isn't you.**

- **Day 22:** Research free-tier auth options compatible with Streamlit (e.g. `streamlit-authenticator`).
- **Day 23:** Implement basic username/password auth as a new architectural layer (update `docs/ARCHITECTURE.md` to reflect this change).
- **Day 24:** Design a minimal multi-tenant data model — how would a second company's data stay separate from Yantai's?
- **Day 25:** Implement config-driven dataset selection so a new "company" can point to their own CSV.
- **Day 26:** Add a simple onboarding flow: upload your own delivery CSV, auto-run the pipeline.
- **Day 27:** Test the onboarding flow with a second, different dataset end-to-end.
- **Day 28:** Security review: confirm no cross-tenant data leakage, no secrets in code, auth properly gates all views.
- **Day 29:** Update all documentation (`README.md`, `ARCHITECTURE.md`, `SCHEMA.md`) to reflect the multi-tenant, authenticated version.
- **Day 30:** Full end-to-end demo with two "companies," write a retrospective on what changed since v1.0.0, tag `v2.0.0`.

---

**By Day 30:** ShipSense goes from a single-city, single-tenant demo to a multi-city, alerting, multi-tenant product with per-order explainability — the natural next chapter after this capstone's v1.0.0.
