import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render

from .forms import CropRecommendationForm

# Base directory for saved model artifacts
ML_DIR = Path(__file__).resolve().parent / "ml"
SAVED_MODELS_DIR = ML_DIR / "saved_models"

# Lazy-loaded model cache to avoid re-reading disk on every request
_MODEL_CACHE = {}


def _get_ml_artifacts():
    """Load and cache Random Forest model, label encoder, and metrics.json."""
    if "rf" not in _MODEL_CACHE:
        rf_path = SAVED_MODELS_DIR / "rf.pkl"
        le_path = SAVED_MODELS_DIR / "label_encoder.pkl"
        metrics_path = SAVED_MODELS_DIR / "metrics.json"

        if not rf_path.exists() or not le_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                "Trained ML artifacts not found. Please run `python recommender/ml/train_models.py` first."
            )

        _MODEL_CACHE["rf"] = joblib.load(rf_path)
        _MODEL_CACHE["le"] = joblib.load(le_path)

        with open(metrics_path, "r", encoding="utf-8") as f:
            _MODEL_CACHE["metrics"] = json.load(f)

    return _MODEL_CACHE["rf"], _MODEL_CACHE["le"], _MODEL_CACHE["metrics"]


def home(request):
    """
    Home view:
    - GET: Displays the two-section soil & climate input form.
    - POST: Runs Random Forest prediction on raw unscaled input and returns top 3 crop candidates.
    """
    if request.method == "POST":
        form = CropRecommendationForm(request.POST)
        if form.is_valid():
            # Extract cleaned values in exact feature order: N, P, K, temperature, humidity, ph, rainfall
            n = form.cleaned_data["N"]
            p = form.cleaned_data["P"]
            k = form.cleaned_data["K"]
            temperature = form.cleaned_data["temperature"]
            humidity = form.cleaned_data["humidity"]
            ph = form.cleaned_data["ph"]
            rainfall = form.cleaned_data["rainfall"]

            features_df = pd.DataFrame(
                [[n, p, k, temperature, humidity, ph, rainfall]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
            )

            # Load model and label encoder
            rf_model, label_encoder, _ = _get_ml_artifacts()

            # Random Forest predicts on raw (unscaled) input
            probabilities = rf_model.predict_proba(features_df)[0]

            # Top candidate indices sorted by descending probability
            top_indices = np.argsort(probabilities)[::-1]

            # Map indices dynamically from label_encoder.classes_
            classes = label_encoder.classes_
            
            top_candidates = []
            for idx in top_indices[:3]:
                top_candidates.append({
                    "crop": classes[idx],
                    "crop_display": classes[idx].replace("_", " ").title(),
                    "probability": round(float(probabilities[idx]) * 100, 2),
                    "probability_raw": float(probabilities[idx]),
                })

            top_pick = top_candidates[0]
            other_possibilities = top_candidates[1:3]

            input_recap = [
                {"name": "Nitrogen (N)", "value": f"{n} kg/ha", "section": "Soil Nutrients"},
                {"name": "Phosphorus (P)", "value": f"{p} kg/ha", "section": "Soil Nutrients"},
                {"name": "Potassium (K)", "value": f"{k} kg/ha", "section": "Soil Nutrients"},
                {"name": "Soil pH", "value": f"{ph:.2f}", "section": "Soil Nutrients"},
                {"name": "Temperature", "value": f"{temperature:.1f} °C", "section": "Climate Conditions"},
                {"name": "Relative Humidity", "value": f"{humidity:.1f} %", "section": "Climate Conditions"},
                {"name": "Rainfall", "value": f"{rainfall:.1f} mm", "section": "Climate Conditions"},
            ]

            context = {
                "top_pick": top_pick,
                "other_possibilities": other_possibilities,
                "top_candidates": top_candidates,
                "input_recap": input_recap,
                "raw_inputs": form.cleaned_data,
            }
            return render(request, "recommender/result.html", context)
    else:
        form = CropRecommendationForm()

    return render(request, "recommender/home.html", {"form": form})


def model_comparison(request):
    """
    Model Comparison view:
    - Reads metrics.json
    - Renders the 4-model evaluation table and data for Chart.js bar charts.
    """
    _, _, metrics = _get_ml_artifacts()
    models_data = metrics.get("models", {})

    # Structure data for Chart.js visualization
    model_names = list(models_data.keys())
    accuracies = [models_data[m]["accuracy"] for m in model_names]
    precisions = [models_data[m]["precision"] for m in model_names]
    recalls = [models_data[m]["recall"] for m in model_names]
    f1_scores = [models_data[m]["f1_score"] for m in model_names]

    chart_payload = {
        "labels": model_names,
        "datasets": [
            {
                "label": "Accuracy (%)",
                "data": accuracies,
                "backgroundColor": "#B5652D",
            },
            {
                "label": "F1 Score (weighted, %)",
                "data": f1_scores,
                "backgroundColor": "#D4A94C",
            },
        ],
    }

    context = {
        "models_data": models_data,
        "chart_payload_json": json.dumps(chart_payload),
    }
    return render(request, "recommender/model_comparison.html", context)


def insights(request):
    """
    Insights view:
    - Reads metrics.json
    - Displays Random Forest feature importance rankings and 2D PCA variance breakdown.
    """
    _, _, metrics = _get_ml_artifacts()
    
    feature_importances = metrics.get("feature_importances", {})
    pca_variance = metrics.get("pca_variance", {})

    # Format feature importances with percentage for UI bar rendering
    max_imp = max(feature_importances.values()) if feature_importances else 1.0
    formatted_importances = []
    for feat, val in feature_importances.items():
        formatted_importances.append({
            "feature": feat,
            "feature_label": feat.replace("_", " ").title(),
            "importance": val,
            "percentage": round(val * 100, 2),
            "bar_width": round((val / max_imp) * 100, 1),
        })

    context = {
        "feature_importances": formatted_importances,
        "pca_variance": pca_variance,
    }
    return render(request, "recommender/insights.html", context)
