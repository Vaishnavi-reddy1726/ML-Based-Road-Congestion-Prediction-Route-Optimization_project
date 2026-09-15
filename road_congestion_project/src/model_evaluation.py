"""
model_evaluation.py
----------------------
Standalone evaluation utility: loads the saved model, evaluates it on a
held-out chronological test split, prints regression metrics, and saves
a feature-importance bar chart + a predicted-vs-actual scatter plot to
the outputs/ folder.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engineering import get_feature_matrix, FEATURE_COLUMNS
from model_training import chronological_split, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned_data.csv"), parse_dates=["timestamp"])
    _, test_df = chronological_split(df, test_frac=0.2)

    model = load_model()
    X_test, y_test, _ = get_feature_matrix(test_df)
    preds = model.predict(X_test)

    print("Regression metrics on held-out test set:")
    print(f"  MAE : {mean_absolute_error(y_test, preds):.4f}")
    print(f"  RMSE: {mean_squared_error(y_test, preds) ** 0.5:.4f}")
    print(f"  R2  : {r2_score(y_test, preds):.4f}")

    # --- Feature importance plot ---
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.title("Feature Importance - Congestion Prediction Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(OUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {fi_path}")

    # --- Predicted vs actual scatter ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.05, s=5)
    plt.plot([0, 1], [0, 1], "r--", linewidth=1)
    plt.xlabel("Actual congestion level")
    plt.ylabel("Predicted congestion level")
    plt.title("Predicted vs Actual Congestion")
    plt.tight_layout()
    pa_path = os.path.join(OUT_DIR, "predicted_vs_actual.png")
    plt.savefig(pa_path, dpi=150)
    plt.close()
    print(f"Saved predicted-vs-actual plot to {pa_path}")


if __name__ == "__main__":
    main()
