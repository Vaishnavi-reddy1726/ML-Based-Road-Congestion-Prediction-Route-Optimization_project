"""
classification_model.py
-------------------------
Trains and evaluates a 3-level congestion classifier (Low / Medium / High)
on top of the same engineered features used by the regression model.
This is the model behind the "3-level congestion, X% accuracy" metric.

The regression model (model_training.py) is still what powers the live
graph-weight updates for route optimization, since a continuous travel-time
estimate is more useful there than a coarse 3-bucket label. This
classifier is a separate, complementary view of the same problem for
reporting a single, interpretable accuracy number.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_engineering import build_features, FEATURE_COLUMNS
from model_training import chronological_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_classification_matrix(df: pd.DataFrame):
    df = build_features(df)
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["congestion_class"].astype(str)
    return X, y


def train_classifier(train_df: pd.DataFrame):
    X_train, y_train = get_classification_matrix(train_df)
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, test_df: pd.DataFrame):
    X_test, y_test = get_classification_matrix(test_df)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=["Low", "Medium", "High"])

    return acc, report, cm


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_data.csv"),
        parse_dates=["timestamp"]
    )
    train_df, test_df = chronological_split(df, test_frac=0.2)

    print(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")
    clf = train_classifier(train_df)
    acc, report, cm = evaluate_classifier(clf, test_df)

    print(f"\nHeld-out test accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows=actual, cols=predicted) [Low, Medium, High]:")
    print(cm)

    path = os.path.join(MODEL_DIR, "congestion_classifier.joblib")
    joblib.dump(clf, path)
    print(f"\nClassifier saved to {path}")
