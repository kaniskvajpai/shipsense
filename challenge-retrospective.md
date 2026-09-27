# ShipSense — Challenge Retrospective

*A day-by-day account of building ShipSense, Days 1–10 of the AB Talks 60-Day Claude AI Challenge capstone.*

## The Arc

ShipSense started as a 9-feature "AI Smart Logistics Platform" idea — ETA prediction, delay prediction, rider allocation, route optimization, demand forecasting, weather analysis, dynamic pricing, heatmaps, and three separate dashboards. It ended as one honest, well-built ML model wrapped in a focused three-screen product, live on the public internet. The story of these 10 days is mostly the story of that narrowing — and everything that had to go right (and wrong) to make the narrower version real.

## Day 1 — Discovery

The interview process surfaced the core tension immediately: the original idea was really eight or nine separate products bundled together. The key reframe was combining ETA prediction and delay-risk into *one* model with a derived business rule, rather than two separate ML systems — this single decision is what made the whole project buildable in 10 days. Named the project **ShipSense**.

## Day 2 — Design

Locked in the "no database, no backend server" architecture — a deliberate simplicity choice justified by the static, historical nature of the data. Adapted the "API Design" deliverable from REST endpoints to internal function contracts, since the approved architecture had no HTTP boundary. This document (`docs/API.md`) turned out to matter enormously later — every function signature built in Days 3–5 matched it exactly, with zero redesign needed mid-build.

## Day 3 — Foundation

Environment setup surfaced the first real technical lesson: **Python 3.14 (the newest version) had no pre-built wheels for pandas/scikit-learn**, forcing a switch to Python 3.12. Also discovered a Windows + OneDrive quirk — two different "Desktop" folders, one tracked by Git and one not — that caused confusing "file not found" issues for the rest of the project until caught and documented.

## Day 4 — Real Data, Real Problems

The planned dataset (`Cainiao-AI/LaDe`'s generic loader) failed with a schema-mismatch error across cities. Pivoted to downloading a single city (Yantai, the smallest) as a direct Parquet file — a clean fix that didn't compromise data quality. Cleaned 206,431 raw rows down to 197,524. The EDA delivered the project's first genuine surprise: **distance had almost zero linear correlation with delivery duration** — a finding that shaped every subsequent day's expectations.

## Day 5 — Model Training

Random Forest beat the Linear Regression baseline by 40% (MAE 70.85 min vs. 117.19 min) — direct confirmation that the relationships in the data were non-linear, exactly as Day 4's EDA had hinted. Feature importances then delivered the project's second surprise: **distance was actually the #1 feature (36.6%)** once non-linear interactions were captured — refining, not contradicting, the earlier finding. Also hit a real Git lesson: a 133MB trained model file exceeded GitHub's 100MB limit, resolved by excluding it via `.gitignore` since the deployed app never actually needs the raw model file.

## Day 6 — Inference & a Real Bug Caught Before It Mattered

Built the delay-risk logic. First version used each region's *median* historical duration as the "promised time" — which mathematically guarantees ~50% of orders exceed it. Caught this before shipping it: switched to the 75th percentile, landing at a realistic 14.4% at-risk rate. This is the project's clearest example of "the model was right, but the business logic wrapping it was wrong" — a distinction that matters in real ML products.

## Day 7 — The Operations Dashboard

Built the first real, functional screen. Also the day repeated Notepad file-save failures started becoming a real productivity cost — multiple silent truncations that looked like bugs in the code but were actually editor/save issues. Learned to verify every edit with `findstr` before assuming it worked.

## Day 8 — QA Pass

A deliberate, honest engineering review of what existed (not a full-app review, since 2 of 3 tabs were still placeholders) — found and fixed real gaps: no error handling on data load, no loading state, a fragile SLA edge case. The Notepad truncation problem escalated seriously this day (a multi-hour debugging saga), ultimately resolved by switching to VS Code for large file edits — a tooling decision that fixed a whole category of recurring problems at once.

## Day 9 — MVP Complete & Live

Finished the Demand Heatmap and Executive Summary, completing the 3-tab MVP. Deployment hit a genuine platform-level issue: Streamlit Community Cloud's `runtime.txt` didn't reliably pin the Python version (a documented platform quirk, confirmed via community reports, not a mistake in the code). Fixed by deleting and redeploying with Python 3.12 selected explicitly in Advanced Settings. **ShipSense went live.**

## Day 10 — Release

Final polish, documentation, and v1.0.0 release — turning a working app into a presentable, portfolio-ready project.

## Skills Demonstrated

- **Product scoping under real constraints** — narrowing 9 features to 1 defensible core loop
- **Data science honesty** — reporting a surprising/inconvenient finding (weak linear correlation) rather than hiding it, then correctly reconciling it with a later finding
- **ML engineering fundamentals** — baseline comparison, proper train/test methodology, feature importance interpretation
- **Debugging discipline** — isolating whether a failure was code, environment, editor, or platform (all four categories showed up across 10 days)
- **Systems thinking** — an architecture decision (no DB, no backend) made on Day 2 held up, unmodified, through Day 9's deployment
- **Real deployment experience** — hitting and resolving actual platform-specific issues (GitHub file limits, Streamlit Cloud Python pinning), not just "happy path" tutorial deployment

## Lessons Learned

1. **The newest tool version isn't always the right one** — Python 3.14 vs. 3.12 cost real time on Day 3.
2. **A statistically "correct" business rule can still be a broken product** — the median-based SLA on Day 6.
3. **When something fails in a way that doesn't make sense, stop patching and reset** — the Day 8 Notepad saga, solved by switching tools entirely rather than continuing to debug the symptom.
4. **Weak correlation ≠ no relationship** — the distance finding taught more about non-linear modeling than any tutorial could have.
5. **Documentation written early (Day 2) pays off late** — the API contracts written before any code existed meant Days 3–5 had zero design ambiguity.

## Farewell

Kanishk — from Day 1's "nine-feature platform" to today's live, working ShipSense: this was a real narrowing, and it's the reason this project actually shipped. Most of the hard moments in these 10 days weren't about the ML — the model came together fast once the data was clean. The hard moments were the unglamorous ones: figuring out why a file kept truncating, why a percentage looked wrong, why a deployment kept failing for a reason that had nothing to do with your code. That's what real engineering work actually is, and you did it without flinching or skipping the debugging to get to something that "looked" done.

The distance-correlation finding on Day 4, and catching your own SLA bug on Day 6 before it shipped — those are the two moments I'd point to if someone asked whether you actually understand what you built, versus just running code someone else wrote. You do.

Sixty days ago this started with AI fundamentals. It ends with a live URL, a trained model with honest numbers, and a GitHub history that tells a true story of how it got built. That's worth being proud of. Go build the next thing.
