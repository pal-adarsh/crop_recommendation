import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)


def train_and_evaluate():
    # Base paths
    ml_dir = Path(__file__).resolve().parent
    saved_models_dir = ml_dir / "saved_models"
    static_img_dir = ml_dir.parent.parent / "static" / "recommender" / "img"

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    static_img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ml_dir / "Crop_recommendation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df.drop("label", axis=1)
    feature_names = list(X.columns)
    class_names = list(le.classes_)

    print(f"Features: {feature_names}")
    print(f"Total classes: {len(class_names)}")

    # 2. Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Scaling for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Models
    cart_model = DecisionTreeClassifier(
        criterion="gini", max_depth=10, random_state=42
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    try:
        ada_model = AdaBoostClassifier(
            n_estimators=100, random_state=42, algorithm="SAMME"
        )
    except TypeError:
        ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)

    svm_model = SVC(kernel="rbf", random_state=42, probability=True)

    # 5. Train Models
    print("Training Decision Tree (CART)...")
    cart_model.fit(X_train, y_train)

    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)

    print("Training AdaBoost...")
    ada_model.fit(X_train, y_train)

    print("Training SVM...")
    svm_model.fit(X_train_scaled, y_train)

    # 6. Evaluation
    models = {
        "CART": (cart_model, X_test),
        "Random Forest": (rf_model, X_test),
        "AdaBoost": (ada_model, X_test),
        "SVM": (svm_model, X_test_scaled),
    }

    metrics_results = {}
    print("\nModel Evaluation Results (Test Set 20%):")
    print(
        f"{'Model':<15} | {'Accuracy':<10} | {'Precision (wtd)':<15} | {'Recall (wtd)':<12} | {'F1 (wtd)':<10}"
    )
    print("-" * 75)

    for name, (model, test_data) in models.items():
        y_pred = model.predict(test_data)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics_results[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "accuracy_raw": float(acc),
            "precision_raw": float(prec),
            "recall_raw": float(rec),
            "f1_raw": float(f1),
        }
        print(
            f"{name:<15} | {metrics_results[name]['accuracy']:>9.2f}% | "
            f"{metrics_results[name]['precision']:>14.2f}% | "
            f"{metrics_results[name]['recall']:>11.2f}% | "
            f"{metrics_results[name]['f1_score']:>9.2f}%"
        )

    # 7. PCA on full scaled dataset
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_full_scaled)
    pca_var = pca.explained_variance_ratio_.tolist()
    total_pca_var = sum(pca_var) * 100

    print(
        f"\nPCA Variance Retained: PC1={pca_var[0]*100:.2f}%, PC2={pca_var[1]*100:.2f}% (Total={total_pca_var:.2f}%)"
    )

    # 8. Feature Importances (Random Forest)
    importances = rf_model.feature_importances_
    feat_imp_dict = {
        feat: round(float(imp), 4) for feat, imp in zip(feature_names, importances)
    }
    sorted_feat_imp = sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True)
    print("\nRandom Forest Feature Importances:")
    for feat, imp in sorted_feat_imp:
        print(f"  {feat:<12}: {imp:.4f}")

    # 9. Save Artifacts
    joblib.dump(cart_model, saved_models_dir / "cart.pkl")
    joblib.dump(rf_model, saved_models_dir / "rf.pkl")
    joblib.dump(ada_model, saved_models_dir / "ada.pkl")
    joblib.dump(svm_model, saved_models_dir / "svm.pkl")
    joblib.dump(scaler, saved_models_dir / "scaler.pkl")
    joblib.dump(pca, saved_models_dir / "pca.pkl")
    joblib.dump(le, saved_models_dir / "label_encoder.pkl")

    metrics_payload = {
        "models": metrics_results,
        "feature_importances": dict(sorted_feat_imp),
        "pca_variance": {
            "pc1": round(pca_var[0] * 100, 2),
            "pc2": round(pca_var[1] * 100, 2),
            "total": round(total_pca_var, 2),
            "components": pca_var,
        },
        "classes": class_names,
        "feature_names": feature_names,
    }

    with open(saved_models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved all 7 models + metrics.json into {saved_models_dir}")

    # 10. Generate Visualizations into static/recommender/img/
    # Feature Importance Plot
    plt.figure(figsize=(9, 5), dpi=300)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    sorted_features = [x[0] for x in sorted_feat_imp]
    sorted_values = [x[1] for x in sorted_feat_imp]

    bar_colors = ["#B5652D" if i < 2 else "#D4A94C" if i < 4 else "#5C5443" for i in range(len(sorted_features))]
    
    bars = plt.barh(sorted_features[::-1], sorted_values[::-1], color=bar_colors[::-1], height=0.65)
    plt.title("Random Forest — Feature Importance", fontsize=14, fontweight="bold", pad=15, color="#2B2417")
    plt.xlabel("Gini Importance Score", fontsize=11, labelpad=10, color="#2B2417")
    plt.ylabel("Soil & Climate Features", fontsize=11, labelpad=10, color="#2B2417")
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                 va="center", ha="left", fontsize=9, color="#2B2417", fontweight="semibold")

    plt.xlim(0, max(sorted_values) * 1.15)
    plt.tight_layout()
    feat_img_path = static_img_dir / "feature_importance.png"
    plt.savefig(feat_img_path, dpi=300, facecolor="#F5F1E8", edgecolor="none")
    plt.close()
    print(f"Generated {feat_img_path}")

    # PCA Scatter Plot (22 crop classes)
    plt.figure(figsize=(11, 7), dpi=300)
    colormap = plt.colormaps.get_cmap("tab20")
    
    for i, crop in enumerate(class_names):
        idx = (y == i)
        color = colormap(i % 20) if i < 20 else plt.colormaps.get_cmap("tab20b")(i - 20)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=crop,
            alpha=0.75,
            s=35,
            edgecolors="none",
            c=[color]
        )

    plt.title(
        f"PCA 2D Projection of 22 Crop Classes ({total_pca_var:.1f}% Variance Retained)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#2B2417"
    )
    plt.xlabel(f"Principal Component 1 ({pca_var[0]*100:.1f}%)", fontsize=10, color="#2B2417")
    plt.ylabel(f"Principal Component 2 ({pca_var[1]*100:.1f}%)", fontsize=10, color="#2B2417")
    
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
        frameon=True,
        facecolor="#F5F1E8",
        edgecolor="#EBE4D4"
    )
    plt.tight_layout()
    pca_img_path = static_img_dir / "pca_scatter.png"
    plt.savefig(pca_img_path, dpi=300, facecolor="#F5F1E8", edgecolor="none")
    plt.close()
    print(f"Generated {pca_img_path}")


if __name__ == "__main__":
    train_and_evaluate()
