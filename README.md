# Crop Recommendation Decision Support System (Django + ML)

An academic machine learning web application built with **Python & Django** for **multiclass crop recommendation** across **22 agricultural crop classes** based on soil chemistry and ambient environmental indicators.

---

## 1. Quickstart & Setup

Follow these three commands to run the project locally:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML pipeline, generate model artifacts, metrics, and visualization plots
python recommender/ml/train_models.py

# 3. Start the local Django development server
python manage.py runserver
```

Once running, navigate to `http://127.0.0.1:8000/` in your browser.

---

## 2. Dataset & Features

The model is trained on `Crop_recommendation.csv` containing **2,200 empirical field samples** with **0 missing values** and **0 duplicate entries**.

- **Target (`label`)**: 22 perfectly balanced crop classes (100 samples each):
  `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`.

- **Input Features (7 total)**:
  | Feature | Agronomic Meaning | Range in Dataset | Units |
  | :--- | :--- | :--- | :--- |
  | `N` | Nitrogen content in soil | 0 – 140 | kg/ha |
  | `P` | Phosphorus content in soil | 5 – 145 | kg/ha |
  | `K` | Potassium content in soil | 5 – 205 | kg/ha |
  | `temperature` | Ambient temperature | 8.83 – 43.68 | °C |
  | `humidity` | Relative humidity | 14.26 – 99.98 | % |
  | `ph` | Soil pH level | 3.50 – 9.94 | pH scale |
  | `rainfall` | Seasonal rainfall | 20.21 – 298.56 | mm |

---

## 3. Empirical Model Benchmark Results

Evaluated on a **20% stratified holdout test set** (440 samples, 20 per class, `random_state=42`) with sample-weighted averaging:

| Algorithm | Accuracy | Precision (wtd) | Recall (wtd) | F1-Score (wtd) | Implementation Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** (Best) | **99.55%** | **99.57%** | **99.55%** | **99.55%** | **Production model**: 100 estimators trained on unscaled data. |
| **SVM (RBF Kernel)** | **98.41%** | **98.56%** | **98.41%** | **98.40%** | Standardized inputs (`StandardScaler`); probability enabled. |
| **CART (Decision Tree)** | **96.36%** | **97.18%** | **96.36%** | **96.39%** | Gini criterion, `max_depth=10`. |
| **AdaBoost (SAMME)** | **31.36%** | **17.66%** | **31.36%** | **20.49%** | 100 decision stumps sequentially boosted. |

### Academic Note on the AdaBoost Result (~31.36%)
Standard AdaBoost utilizes single-split decision stumps as weak base learners. In a high-cardinality **22-class problem** with subtle continuous feature boundaries (e.g., legume varieties like *mungbean*, *blackgram*, and *lentil* sharing near-identical N-P-K profiles), orthogonal 1-split stumps struggle to partition complex multiclass decision spaces. This finding provides legitimate theoretical insight for viva discussions on why bagging (Random Forest) excels over boosting weak stumps on overlapping multiclass data.

---

## 4. Key Agronomic Insights

- **Feature Importance (Random Forest Gini Score)**:
  `rainfall (0.2302) > humidity (0.2242) > K (0.1754) > P (0.1508) > N (0.0964) > temperature (0.0724) > ph (0.0506)`
  - *Rainfall* and *Relative Humidity* are the primary drivers of crop selection in this dataset, accounting for over **45%** of model weight.
- **Principal Component Analysis (PCA)**:
  - 7 features reduced to 2 principal components retain **46.07% cumulative variance** (PC1: 27.59%, PC2: 18.48%).
  - 2D projection demonstrates distinct clusters for specialized crops (rice, coffee, coconut).

---

## 5. Design System & User Interface

The web interface is styled as a **field agronomist's instrument panel** using custom CSS tokens without Bootstrap or Tailwind:
- **Palette**: Soil dark (`#2B2417`), Paper (`#F5F1E8`), Terracotta Clay (`#B5652D`), Wheat Gold (`#D4A94C`), Leaf Green (`#4C7C3F`).
- **Typography**: Google Fonts [`Fraunces`](https://fonts.google.com/specimen/Fraunces) (serif for headlines and numbers) + [`Inter`](https://fonts.google.com/specimen/Inter) (sans-serif for labels and inputs).
- **Motifs**: Animated horizontal soil-strata SVG cross-section dividers.

---

## 6. Project Architecture

```
Crop-prediction/
├── manage.py
├── requirements.txt
├── README.md
├── crop_recommendation_app/       # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── recommender/                   # Main Django application
│   ├── ml/
│   │   ├── Crop_recommendation.csv
│   │   ├── train_models.py        # ML training and evaluation script
│   │   └── saved_models/          # rf.pkl, svm.pkl, cart.pkl, ada.pkl, scaler.pkl, pca.pkl, label_encoder.pkl, metrics.json
│   ├── forms.py                   # 7-field bounded Django form
│   ├── views.py                   # home, model_comparison, insights views
│   ├── urls.py                    # App routing
│   └── templates/recommender/
│       ├── base.html              # Layout, navbar, soil-strata motif, footer
│       ├── home.html              # Two-section input form with presets
│       ├── result.html            # Top pick hero + top-3 candidates + recap table
│       ├── model_comparison.html  # Full metrics table + Chart.js visual
│       └── insights.html          # Feature importance & PCA scatter plot
└── static/recommender/
    ├── css/style.css              # Earthy design tokens & responsive CSS
    └── img/
        ├── feature_importance.png # Pre-generated Gini importance plot
        └── pca_scatter.png        # Pre-generated 22-class PCA scatter plot
```

---

## 7. Verification & Sanity Checks

Test inputs to verify predictions:

- **Rice-like conditions** (`N=90, P=42, K=43, ph=6.5, temp=21.0, hum=82.0, rain=203.0`) -> Predicts **Rice** (~90% confidence).
- **Coffee-like conditions** (`N=100, P=20, K=30, ph=6.8, temp=25.0, hum=60.0, rain=160.0`) -> Predicts **Coffee** (100% confidence).
