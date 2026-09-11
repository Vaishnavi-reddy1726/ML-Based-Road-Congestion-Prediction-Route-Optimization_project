"""
model_training.py
-------------------
Trains a regression model to predict road congestion level (0-1)
from weather + time features, using a chronological train/test split
(no shuffling, since this is time series-like data).
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import get_feature_matrix, FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def chronological_split(df: pd.DataFrame, test_frac=0.2):
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.iloc[split_idx]["timestamp"]
    train_df = df[df["timestamp"] < cutoff_time]
    test_df = df[df["timestamp"] >= cutoff_time]
    return train_df, test_df


def train_model(train_df: pd.DataFrame, model_type="random_forest"):
    X_train, y_train, _ = get_feature_matrix(train_df)

    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=80, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, test_df: pd.DataFrame):
    X_test, y_test, _ = get_feature_matrix(test_df)
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }
    return metrics, preds, y_test


def save_model(model, filename="congestion_model.joblib"):
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    return path


def load_model(filename="congestion_model.joblib"):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv"),
        parse_dates=["timestamp"]
    )

    train_df, test_df = chronological_split(df, test_frac=0.2)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    model = train_model(train_df, model_type="random_forest")
    metrics, preds, y_test = evaluate_model(model, test_df)

    print("\nModel evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    path = save_model(model)
    print(f"\nModel saved to {path}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8))
