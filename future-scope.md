# ShipSense — Future Scope

This roadmap builds on the deliberate scope cuts made in the PRD (Section 5.2) — each was cut for a good reason (time, complexity), not because it lacked value. This is the order I'd actually build them in, and why.

## Next 3 Months: Make the Core Loop Trustworthy at Scale

**1. Real, per-order SLA data (not regional 75th-percentile proxy)**
Right now "promised time" is inferred from historical regional patterns. Real deployment needs actual promised-delivery-time data per order. This is the single highest-value fix — everything else depends on the risk flag being trustworthy.

**2. Per-order explainability (SHAP values)**
Currently "top delay factor" is the model's *global* top feature applied to every at-risk order. Real per-row SHAP values would tell an ops manager *why this specific order* is at risk — much more actionable.

**3. Multi-city support**
Currently trained only on Yantai (smallest city in LaDe-D). Extending to the other cities in the dataset would test whether the model generalizes, and is a natural next dataset step already sitting there unused.

**4. Model monitoring / drift detection**
Add a simple mechanism to flag when live prediction error starts drifting from the reported 70.85 min MAE — the first sign a production model needs retraining.

## Next 6 Months: From Dashboard to Decision Support

**5. Live weather & traffic integration**
The biggest deliberately-cut feature from v1.0. Would require a live API layer (see Architecture Section 5's "future" note) and directly improve the model's biggest weak point — external conditions aren't in the data at all right now.

**6. Smart rider reallocation suggestions**
Not full optimization (too complex), but a simpler v1: "these 3 at-risk orders are near Courier X, who has capacity" — a recommendation, not an automated action.

**7. Alerting (Slack/email) for at-risk orders**
Right now someone has to open the dashboard. Push notifications for newly-flagged at-risk orders would make this a proactive tool instead of a passive one.

**8. Authentication + multi-company support**
Turns this from a single-dataset demo into something an actual company could deploy privately with their own data.

## Next 12 Months: Toward a Real Product

**9. Real route optimization engine**
The most technically ambitious cut feature. Would need a proper routing library (e.g. OR-Tools) and dedicated infrastructure — a genuine v2.0 scope, not an incremental add.

**10. Dynamic pricing engine**
Depends on #9 and real business input on pricing strategy — this is a product/business decision as much as a technical one, so it's rightly last.

**11. Mobile app for couriers**
A courier-facing view (not just ops-facing) showing their own risk-flagged deliveries — extends the product to a second user type entirely.

**12. Real-time order ingestion**
The final architectural shift: from "batch-scored historical data" (today's correct choice) to genuinely live order streams — would require revisiting the no-database decision in `ARCHITECTURE.md` Section 7, at the point this becomes justified.

---

**Why this order:** each phase makes the *existing* core more trustworthy before adding new surface area. #1-4 fix honesty/accuracy gaps in what already exists. #5-8 make the tool proactive instead of passive. #9-12 are the genuinely new capabilities that were correctly out of scope for a 10-day build.
