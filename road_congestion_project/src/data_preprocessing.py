"""
data_preprocessing.py
----------------------
Cleans and prepares the raw historical weather + traffic dataset:
    - handles missing values
    - fixes data types
    - removes outliers / invalid rows
    - encodes categorical weather condition
"""

import pandas as pd
import numpy as np


def load_raw_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Handle missing values ---
    numeric_cols = ["temperature_c", "precipitation_mm", "visibility_km",
                     "wind_speed_kmph", "congestion_level"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "weather_condition" in df.columns:
        df["weather_condition"] = df["weather_condition"].fillna("Clear")

    # --- Fix invalid / out-of-range values ---
    df["congestion_level"] = df["congestion_level"].clip(0.0, 1.0)
    df["visibility_km"] = df["visibility_km"].clip(lower=0.0)
    df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)
    df["wind_speed_kmph"] = df["wind_speed_kmph"].clip(lower=0.0)

    # --- Drop exact duplicates ---
    df = df.drop_duplicates()

    # --- Drop rows still missing critical fields ---
    df = df.dropna(subset=["road_id", "timestamp", "congestion_level"])

    # --- Simple outlier removal (z-score) on congestion_level per road ---
    df["z"] = df.groupby("road_id")["congestion_level"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
    )
    df = df[df["z"].abs() < 4].drop(columns=["z"])

    return df.reset_index(drop=True)


def one_hot_encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["weather_condition"], prefix="weather")


if __name__ == "__main__":
    raw = load_raw_data("data/historical_data.csv")
    cleaned = clean_data(raw)
    print(f"Raw rows: {len(raw)}  ->  Cleaned rows: {len(cleaned)}")
    cleaned.to_csv("data/cleaned_data.csv", index=False)
