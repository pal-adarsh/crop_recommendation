# Project Build Prompt: Crop Recommendation Web App (Django)

## Context & Intent

This is a BE Machine Learning Lab Mini Project for your friend. The topic is
**crop recommendation**: given soil nutrient levels and climate conditions,
recommend which of 22 crops is best suited to grow. This is a **multiclass
classification** problem (22 classes), which is a meaningfully different
flavor from a binary classification project — good, since your other friend's
heart disease project is binary. Don't let the two look like copies of each
other.

Build this as **Python + Django only** — same reasoning as before: the ML
model is Python/scikit-learn, so keeping the backend in Python avoids an
unnecessary API layer. A single Django project with server-rendered templates
is the right scope for an academic project that needs to run with one command
and be explainable in a five-minute viva.

Do not change the dataset, preprocessing, or hyperparameters below. Every
number in this document came from an actual run against the provided CSV —
treat it as ground truth, not a placeholder.

---

## 1. Dataset & Ground Truth (fixed — do not alter)

- Source file: `Crop_recommendation.csv` — **2200 rows, 8 columns**
  (7 features + `label`), **0 missing values, 0 duplicates**.
- Columns, in this exact order: `N, P, K, temperature, humidity, ph, rainfall, label`
- **Perfectly balanced**: 22 crop classes, exactly 100 samples each.
- Crop classes (alphabetical): `apple, banana, blackgram, chickpea, coconut,
  coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans,
  mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice,
  watermelon`

| Feature | Meaning | Range in data |
|---|---|---|
| N | Nitrogen content in soil (kg/ha) | 0–140 |
| P | Phosphorus content in soil (kg/ha) | 5–145 |
| K | Potassium content in soil (kg/ha) | 5–205 |
| temperature | Temperature (°C) | 8.83–43.68 |
| humidity | Relative humidity (%) | 14.26–99.98 |
| ph | Soil pH | 3.50–9.94 |
| rainfall | Rainfall (mm) | 20.21–298.56 |

- Target `label` is a **string crop name** — encode it with `LabelEncoder`
  before training, and save the fitted encoder so predictions can be mapped
  back to a readable crop name.

## 2. ML Pipeline (replicate exactly)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df["label"])   # 22 classes -> integers 0-21
X = df.drop("label", axis=1)        # N, P, K, temperature, humidity, ph, rainfall

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
cart_model = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
rf_model   = RandomForestClassifier(n_estimators=100, random_state=42)
ada_model  = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME")
svm_model  = SVC(kernel="rbf", random_state=42, probability=True)
```

- CART, Random Forest, AdaBoost train on **unscaled** `X_train`.
- SVM trains on **scaled** `X_train_scaled`.
- `algorithm="SAMME"` is required for AdaBoost on newer scikit-learn versions
  when doing multiclass classification — without it you'll get an error or a
  deprecation warning depending on version. If the installed scikit-learn
  version doesn't accept the `algorithm` argument, omit it (older versions
  default to SAMME already).
- Use `average="weighted"` for precision/recall/F1 — this is a 22-class
  problem, not binary, so plain `precision_score(y_test, pred)` will error
  without an averaging strategy.
- PCA (insights page only): `PCA(n_components=2)` fit on
  `scaler.fit_transform(X)` — the full dataset, not just train.

### Actual results (source of truth — do not invent different numbers)

| Model | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) |
|---|---|---|---|---|
| CART | 96.36% | 97.18% | 96.36% | 96.39% |
| **Random Forest** | **99.55%** | **99.57%** | **99.55%** | **99.55%** |
| AdaBoost | 31.36% | 17.66% | 31.36% | 20.49% |
| SVM | 98.41% | 98.56% | 98.41% | 98.40% |

- **Best model: Random Forest** (highest accuracy, and the clear winner here)
  — this is the model that powers the live recommendation form.
- **Important finding to report honestly, not hide**: AdaBoost performs
  poorly (~31%) on this dataset. This is a real, expected result — standard
  AdaBoost with weak decision-stump learners struggles on 22-class problems
  with subtle feature overlap between similar crops (e.g. mungbean vs
  blackgram vs lentil have close N-P-K profiles). Do not tune AdaBoost to
  artificially inflate this number or quietly drop it from the comparison —
  report it as-is and note in the UI/report that it demonstrates a real
  limitation of boosting weak learners on many-class problems. This is a
  legitimate, interesting viva talking point, not a bug to fix.
- Random Forest feature importance, highest to lowest:
  `rainfall (0.230) > humidity (0.224) > K (0.175) > P (0.151) > N (0.096) > temperature (0.072) > ph (0.051)`
- PCA: 7 features → 2 components, **46.07% variance retained** (27.59% + 18.48%).

---

## 3. Design Direction — read this before writing any HTML/CSS

Don't reuse your other friend's clinical/navy theme verbatim — this is a
different subject (agriculture, not medicine) and should look and feel
different, even though the underlying Django structure is similar. Avoid the
generic AI-template defaults: no warm-cream-background-plus-terracotta-accent
combo, no rounded SaaS card grids with soft drop shadows, no centered
gradient hero, no emoji as icons.

Think of this as **a field agronomist's instrument panel** — practical,
earthy, precise, built for someone standing in a field checking soil numbers
against real thresholds, not a consumer gardening app.

### Design tokens

**Color** (define as CSS custom properties on `:root`):
```css
--soil-dark:    #2B2417;   /* primary background, deep soil brown-black */
--paper:        #F5F1E8;   /* light surface / card background, unbleached paper */
--paper-dim:    #EBE4D4;   /* subtle section divider background */
--signal-grow:  #4C7C3F;   /* confirmed recommendation — leaf green */
--accent-clay:  #B5652D;   /* data highlights, active nav, chart accents — terracotta clay, not decorative orange */
--accent-wheat: #D4A94C;   /* secondary accent for charts, muted gold-wheat */
--ink-soft:     #A69C87;   /* secondary/muted text on dark */
--slate-600:    #5C5443;   /* secondary text on light */
--line:         rgba(245,241,232,0.14); /* hairline dividers on dark */
```
This palette is deliberately earthy (soil brown, leaf green, clay) rather than
generic SaaS colors — it's grounded in the actual subject matter (soil,
crops, farmland), not decoration for its own sake.

**Typography**:
- Headlines, section titles, and all numeric data (predicted crop name,
  confidence %, metric tables) → **`Fraunces`** (serif, Google Fonts, has an
  agricultural-almanac/seed-catalog character at higher optical sizes),
  weight 500–600.
- Body copy, form labels, navigation, buttons → **`Inter`** (Google Fonts),
  weights 400/500/600 — clean and legible for dense form labels.
- Don't mix both fonts on one line. No all-caps labels — normal case with
  letter-spacing ≤ 0.02em if emphasis is needed.

**Layout & structural motif**:
- Left-aligned throughout — no centered hero, no centered form.
- Use a recurring **soil-strata motif**: a thin horizontal band of layered
  lines (like a soil cross-section diagram — 3–4 stacked horizontal strokes
  of decreasing opacity in `--accent-clay`/`--accent-wheat`/`--ink-soft`) as
  a section-divider/header accent. This is the one bold recurring device —
  use it once or twice per page, not on every element.
- Group the recommendation form into two labeled sections, mirroring how a
  soil test report actually reads:
  1. **Soil Nutrients** — N, P, K, ph
  2. **Climate Conditions** — temperature, humidity, rainfall
- Result page: the recommended crop name is the hero — large `Fraunces` text
  naming the crop (e.g. "Recommended: Rice") with confidence percentage,
  colored with `--signal-grow`. Below it, show the **top 3 candidate crops**
  with their probabilities (this is a 22-class problem — showing just the
  top pick without runners-up loses useful information a real
  recommendation tool would show). No celebratory tone — plain and
  informative.
- Model comparison & insights pages: real tables and a Chart.js bar chart
  (via CDN), not icon stat-cards. The AdaBoost result should be shown plainly
  alongside the others, not hidden or footnoted apologetically.

**Motion**: one deliberate moment only — e.g. the soil-strata bands drawing
in on page load, or the result crop name/confidence settling into place. No
hover-lift on every card, no staggered per-section fade-ins.

**Copy voice**: plain and practical. Buttons say what they do
("Get recommendation", not "Submit"). The result explains itself in one
sentence. Every page carries a small, quiet disclaimer that this is an
academic project and not a substitute for professional agronomic advice.

Before building, if your environment supports it, sketch the token/layout
plan in a couple of sentences and check it doesn't collapse into the generic
templated look before writing code.

---

## 4. Project Structure

```
crop_recommendation_app/
├── manage.py
├── requirements.txt
├── README.md
├── crop_recommendation_app/       # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── recommender/                   # Django app
│   ├── ml/
│   │   ├── train_models.py        # trains all 4 models + PCA, saves artifacts
│   │   ├── Crop_recommendation.csv
│   │   └── saved_models/          # cart.pkl, rf.pkl, ada.pkl, svm.pkl,
│   │                                # scaler.pkl, pca.pkl, label_encoder.pkl,
│   │                                # metrics.json
│   ├── forms.py                   # Django form for N, P, K, temp, humidity, ph, rainfall
│   ├── views.py
│   ├── urls.py
│   └── templates/recommender/
│       ├── base.html              # nav, soil-strata header, footer disclaimer
│       ├── home.html              # sectioned recommendation form
│       ├── result.html            # top pick hero + top-3 candidates + input recap
│       ├── model_comparison.html  # metrics table + accuracy bar chart
│       └── insights.html          # feature importance + PCA scatter (colored by crop)
└── static/recommender/
    ├── css/style.css              # design tokens + all styling, no Bootstrap
    └── img/                       # pre-generated feature-importance & PCA PNGs (fallback)
```

Plain CSS with the token system above — no Bootstrap, no Tailwind. Chart.js
via CDN is fine for the accuracy bar chart.

---

## 5. Step-by-Step Build Instructions

### Step 1 — Environment & scaffolding
- `requirements.txt`: `django`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`.
- `django-admin startproject crop_recommendation_app .`
- `python manage.py startapp recommender`
- Register `recommender` in `INSTALLED_APPS`; set up `STATICFILES_DIRS` and
  `TEMPLATES` dirs.

### Step 2 — Training script (`recommender/ml/train_models.py`)
- Load `Crop_recommendation.csv`, run the exact pipeline in Section 2.
- `LabelEncoder` on `label` — save the fitted encoder as `label_encoder.pkl`
  (needed to turn predictions back into crop names).
- Train and evaluate all four models on the same test set; compute accuracy,
  weighted precision, weighted recall, weighted F1 for each (use
  `average="weighted"` — required for multiclass).
- Fit PCA and Random Forest feature importances.
- Save all trained artifacts with `joblib.dump()` into `saved_models/`:
  `cart.pkl`, `rf.pkl`, `ada.pkl`, `svm.pkl`, `scaler.pkl`, `pca.pkl`,
  `label_encoder.pkl`.
- Save `metrics.json` with: the results table (Section 2), RF feature
  importances, PCA variance retained, and the list of 22 crop class names in
  encoder order.
- Pre-generate the feature-importance bar chart and a PCA scatter plot
  (colored by crop class — with 22 classes, use a qualitative colormap, not
  a single accent color, and include a compact legend) as PNGs into
  `static/recommender/img/`, using the Section 3 color tokens where sensible.
- Run this script once manually before `runserver` — never on request.

### Step 3 — Django form (`recommender/forms.py`)
- One field per feature: `N`, `P`, `K` (IntegerField), `temperature`,
  `humidity`, `ph`, `rainfall` (FloatField) — with `min_value`/`max_value`
  bounds from the range table in Section 1, and `help_text` explaining each
  in plain language (e.g. "K — Potassium content in soil, kg/ha").

### Step 4 — Views (`recommender/views.py`)
- `home`: GET renders the sectioned form; POST loads `rf.pkl` +
  `label_encoder.pkl` (Random Forest doesn't need scaling — it trained on
  unscaled data), predicts, and renders `result.html` with the top predicted
  crop, its confidence (`predict_proba`), and the **top 3 candidates with
  probabilities** (use `np.argsort` on the probability array, not just
  `.predict()`).
- `model_comparison`: reads `metrics.json`, renders the four-model table +
  accuracy bar chart. Include the AdaBoost result plainly — don't omit or
  visually bury it.
- `insights`: reads `metrics.json`, renders RF feature importance chart and
  PCA scatter (pre-generated PNG, or Chart.js/SVG equivalent).

### Step 5 — Templates & styling
- `base.html` first: navbar (Home / Model Comparison / Insights), soil-strata
  SVG motif, disclaimer footer.
- `home.html`: two-section form (Soil Nutrients / Climate Conditions).
- `result.html`: crop-name hero + confidence, then a "Other possibilities"
  block listing candidates #2 and #3 with their probabilities, then the
  input recap.
- `model_comparison.html` and `insights.html` per Section 3.
- `static/recommender/css/style.css` implementing the full token system.

### Step 6 — URLs
- `recommender/urls.py`: `/`, `/models/`, `/insights/`.
- Wire into the project's root `urls.py`.

### Step 7 — Polish & verify
- Responsive to mobile, visible keyboard focus states.
- Confirm class-index-to-crop-name mapping is correct — verify against
  `label_encoder.classes_`, don't hardcode the 22-crop order manually
  anywhere (it must always come from the saved encoder).
- `README.md` with setup steps and a project summary for the report appendix.

---

## 6. What NOT to Do
- Don't change the dataset, hyperparameters, random_state, or train/test split.
- Don't swap the "best model" away from Random Forest without flagging it to
  your friend explicitly.
- Don't hide, retune, or apologize away the AdaBoost result — report it
  honestly as a real, explainable limitation.
- Don't add authentication, a database for storing predictions, or deployment
  config — this is a local-run academic demo.
- Don't reuse your other friend's exact navy/Spectral clinical theme — this
  project needs its own visual identity per Section 3.
- Don't hardcode the 22 crop names in prediction/display order anywhere other
  than reading them from `label_encoder.classes_` — a mismatch here silently
  produces wrong crop names for correct predictions.

---

## 7. Deliverable

A working Django project runnable locally with:
```
pip install -r requirements.txt
python recommender/ml/train_models.py
python manage.py runserver
```
...that lets a user enter soil nutrient and climate values through a
sectioned form, get a Random-Forest-based crop recommendation with confidence
and top-3 alternatives, styled in the earthy/soil-strata visual identity from
Section 3, and browse model comparison and feature-importance/PCA insight
pages — all numbers matching this document.
