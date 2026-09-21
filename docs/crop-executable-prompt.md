# Executable Build Plan — Crop Recommendation (Django)

Run this **one step at a time**. Give your AI coding agent one numbered step
per message, in order. Each step names exactly what to produce and how to
verify it before moving on — don't skip ahead, later steps assume earlier
ones are done and verified.

Read `crop_prompt.md` first (or attach it as context) — it has the full
dataset spec, exact ML pipeline, ground-truth results (including the
AdaBoost finding), and the full design system. This file is the *sequence*;
`crop_prompt.md` is the *spec*.

---

## STEP 0 — Confirm inputs are present

**Prompt to agent:**
> Confirm you have `Crop_recommendation.csv` available (2200 rows, 8 columns:
> N, P, K, temperature, humidity, ph, rainfall, label — 22 balanced crop
> classes, 100 samples each). If not, tell me where to place it before
> continuing.

**Verify:** Agent confirms shape, column names, and class count match.

---

## STEP 1 — Scaffold the Django project

**Prompt to agent:**
> Create a new Django project called `crop_recommendation_app` with one app
> called `recommender`. Create `requirements.txt` with: django, scikit-learn,
> pandas, numpy, matplotlib, joblib. Set up the project structure exactly as
> described in Section 4 of crop_prompt.md (including `recommender/ml/`,
> `recommender/templates/recommender/`, `static/recommender/css`,
> `static/recommender/img`). Register the `recommender` app, configure
> `TEMPLATES` dirs and `STATICFILES_DIRS`. Don't write views, models, or ML
> code yet — just the skeleton.

**Verify:**
```bash
pip install -r requirements.txt
python manage.py runserver
```
Server starts with no errors.

---

## STEP 2 — Write and run the training script

**Prompt to agent:**
> Copy `Crop_recommendation.csv` into `recommender/ml/`. Write
> `recommender/ml/train_models.py` implementing the exact pipeline from
> Section 2 of crop_prompt.md: LabelEncoder on the crop label, the same
> train/test split, the same four models with the same hyperparameters
> (including `algorithm="SAMME"` for AdaBoost — fall back gracefully if the
> installed scikit-learn version doesn't accept that argument), StandardScaler
> for SVM, and PCA(n_components=2) on the full scaled dataset. Compute
> accuracy, weighted precision, weighted recall, weighted F1 for all four
> models (use `average="weighted"` — this is multiclass). Save all trained
> artifacts (cart.pkl, rf.pkl, ada.pkl, svm.pkl, scaler.pkl, pca.pkl,
> label_encoder.pkl) into `recommender/ml/saved_models/` with joblib, and
> save `metrics.json` with the full results table, RF feature importances,
> PCA variance retained, and the 22 class names in encoder order. Also
> generate `feature_importance.png` and a PCA scatter plot colored by crop
> class (qualitative colormap + compact legend, since there are 22 classes)
> into `static/recommender/img/`.

**Run it:**
```bash
python recommender/ml/train_models.py
```

**Verify:**
- `recommender/ml/saved_models/` contains all 7 `.pkl` files + `metrics.json`.
- `metrics.json` shows: Random Forest ≈ **99.55%** accuracy, CART ≈ 96.36%,
  SVM ≈ 98.41%, and **AdaBoost ≈ 31.36%** (this low number is expected and
  correct — see Section 2 of crop_prompt.md for why; don't treat it as a bug
  and don't let the agent "fix" it by silently changing hyperparameters).
- Both PNGs exist and open correctly; the PCA scatter shows visibly distinct
  clusters for at least some crops (e.g. rice, coffee, coconut tend to
  separate well).

---

## STEP 3 — Build the recommendation form

**Prompt to agent:**
> Write `recommender/forms.py` with a Django form for the 7 input features
> (N, P, K, temperature, humidity, ph, rainfall), following the field types,
> bounds, and help text from Section 1 and Step 3 of crop_prompt.md.

**Verify:** Check all 7 fields are present with sensible min/max bounds
(e.g. `ph` bounded roughly 3.5–10, not 0–100) and clear help text.

---

## STEP 4 — Write the views

**Prompt to agent:**
> Write `recommender/views.py` with three views: `home` (GET shows the form,
> POST loads rf.pkl + label_encoder.pkl, predicts on the raw unscaled input
> since Random Forest trained unscaled, and renders result.html with the top
> crop name, its confidence, and the top 3 candidates with probabilities —
> use predict_proba and np.argsort, not just .predict()), `model_comparison`
> (reads metrics.json, renders the metrics table including AdaBoost's real
> result, plus data for a Chart.js bar chart), and `insights` (reads
> metrics.json, renders feature importance + PCA visuals). Map predicted
> class indices back to crop names using label_encoder.classes_ — never
> hardcode the crop order.

**Verify:** Agent shows you `views.py`. Confirm:
- Input array order matches `N, P, K, temperature, humidity, ph, rainfall`.
- Random Forest predicts on **unscaled** input (it trained unscaled — feeding
  it scaled input here would silently break predictions).
- Crop names come from `label_encoder.classes_`, not a hardcoded list.

---

## STEP 5 — Wire up URLs

**Prompt to agent:**
> Create `recommender/urls.py` with routes for `/` (home, GET+POST),
> `/models/` (model_comparison), `/insights/` (insights). Include this in the
> project's root `urls.py`.

**Verify:** `/`, `/models/`, `/insights/` all return 200.

---

## STEP 6 — Build `base.html` and the design system CSS

**Prompt to agent:**
> Write `static/recommender/css/style.css` implementing the full design token
> system from Section 3 of crop_prompt.md — CSS custom properties for the
> soil/clay/wheat palette, Fraunces + Inter via Google Fonts, the soil-strata
> SVG motif as a reusable header element, left-aligned layout, no Bootstrap.
> Then write `recommender/templates/recommender/base.html` with the navbar
> (Home / Model Comparison / Insights), the soil-strata header, a
> `{% block content %}`, and the disclaimer footer.

**Verify:** Load any page — confirm Fraunces/Inter are actually loading
(inspect element), colors match the earthy token list (not the navy/Spectral
theme from the heart disease project), and the soil-strata motif renders.

---

## STEP 7 — Build `home.html` (the recommendation form)

**Prompt to agent:**
> Write `recommender/templates/recommender/home.html` extending base.html,
> with the form in the two sections from Section 3: Soil Nutrients (N, P, K,
> ph) and Climate Conditions (temperature, humidity, rainfall). Submit button
> says "Get recommendation."

**Verify:** Load `/`. Confirm all 7 fields render in the two labeled
sections with visible labels/help text, and keyboard focus states work.

---

## STEP 8 — Build `result.html`

**Prompt to agent:**
> Write `recommender/templates/recommender/result.html`. The top recommended
> crop name is the hero in large Fraunces text with its confidence
> percentage, colored with --signal-grow. Below it, a clearly labeled "Other
> possibilities" section listing the 2nd and 3rd most likely crops with their
> probabilities. Below that, a recap table of the values entered. Include
> the academic disclaimer line.

**Verify:** Submit the form from `/` with a rice-like input (see test values
below) and confirm the result page shows "rice" as the top pick with high
confidence, plus two sensible runner-up crops.

---

## STEP 9 — Build `model_comparison.html` and `insights.html`

**Prompt to agent:**
> Write `recommender/templates/recommender/model_comparison.html` with the
> full four-model metrics table from metrics.json (including AdaBoost's real
> ~31% result, shown plainly, not hidden) and a Chart.js accuracy bar chart.
> Write `recommender/templates/recommender/insights.html` with the Random
> Forest feature importance chart and the PCA scatter plot (using the
> pre-generated PNG from Step 2), plus 1–2 sentences explaining each — for
> the insights page, note that rainfall and humidity are the top two drivers
> of crop choice in this dataset.

**Verify:** `/models/` shows all four models' real metrics, including
AdaBoost's low score, not smoothed over. `/insights/` shows `rainfall` as
the top feature and mentions ~46% PCA variance retained.

---

## STEP 10 — Full pass: responsiveness, accessibility, polish

**Prompt to agent:**
> Do a pass over all four pages: confirm the layout collapses cleanly to a
> single column on mobile widths (~375px), all interactive elements have
> visible keyboard focus states, color contrast is readable on both --soil-dark
> and --paper backgrounds, and the one motion moment (soil-strata draw-in or
> result settle) works without being distracting. Fix anything that breaks.

**Verify:** Resize to mobile width on all 4 pages — no horizontal scroll.
Tab through each page with keyboard only.

---

## STEP 11 — README and final check

**Prompt to agent:**
> Write `README.md` with setup instructions (pip install, run
> train_models.py, runserver) and a short project summary suitable for a
> report appendix, including a one-line honest note about the AdaBoost
> result being a genuine finding, not an error. Then do a final review:
> confirm no hardcoded/fabricated metric numbers or crop-name orderings exist
> anywhere in the templates — everything displayed should trace back to
> metrics.json or label_encoder.classes_.

**Verify:** Fresh copy of the project, run the three commands from scratch,
confirm it works end to end with no manual fixes needed.

---

## Quick reference — test inputs for Step 8

**Rice-like conditions** (expected top pick: **rice**, high confidence):

| Field | Value |
|---|---|
| N | 90 |
| P | 42 |
| K | 43 |
| temperature | 21 |
| humidity | 82 |
| ph | 6.5 |
| rainfall | 203 |

**Coffee-like conditions** (expected top pick: **coffee**):

| Field | Value |
|---|---|
| N | 100 |
| P | 20 |
| K | 30 |
| temperature | 25 |
| humidity | 60 |
| ph | 6.8 |
| rainfall | 160 |

Use these to sanity-check the prediction pipeline end to end before treating
the app as done.
