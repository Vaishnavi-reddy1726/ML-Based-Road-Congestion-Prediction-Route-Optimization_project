"""
feature_engineering.py
------------------------
Builds model-ready features from cleaned weather + traffic data:
    - cyclical time encodings (hour, day of week)
    - rush hour / weekend flags
    - weather severity index
    - per-road rolling average congestion (lag feature)
    - one-hot encoded weather condition
"""

import numpy as np
import pandas as pd

WEATHER_SEVERITY = {"Clear": 0.0, "Rain": 0.4, "Fog": 0.5, "Storm": 0.8, "Snow": 0.9}

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_rush_hour",
    "temperature_c", "precipitation_mm", "visibility_km", "wind_speed_kmph",
    "weather_severity",
    "rolling_avg_congestion",
    "weather_Clear", "weather_Fog", "weather_Rain", "weather_Snow", "weather_Storm",
]

TARGET_COLUMN = "congestion_level"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df


def add_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "weather_condition" in df.columns:
        df["weather_severity"] = df["weather_condition"].map(WEATHER_SEVERITY).fillna(0.0)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-road rolling average of the last 3 hourly observations (lag feature).

    When historical `timestamp` ordering is available (training data), this
    computes a true trailing rolling average per road. For live, single-shot
    prediction batches (no `timestamp` column -- one row per road representing
    "right now"), the caller is expected to have already seeded
    `congestion_level` with the most recent known average for that road, so
    that value is used directly as the rolling average.
    """
    df = df.copy()

    if "timestamp" not in df.columns:
        df["rolling_avg_congestion"] = df["congestion_level"]
        return df

    df = df.sort_values(["road_id", "timestamp"])
    df["rolling_avg_congestion"] = (
        df.groupby("road_id")["congestion_level"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["rolling_avg_congestion"] = df["rolling_avg_congestion"].fillna(df["congestion_level"].mean())
    return df


def one_hot_encode_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["weather_condition"], prefix="weather")
    # Ensure all expected weather dummy columns exist (some may be absent in a given slice)
    for col in ["weather_Clear", "weather_Fog", "weather_Rain", "weather_Snow", "weather_Storm"]:
        if col not in df.columns:
            df[col] = 0
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_weather_severity(df)
    df = add_rolling_features(df)
    df = one_hot_encode_weather(df)
    return df


def get_feature_matrix(df: pd.DataFrame):
    df = build_features(df)
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(float)
    return X, y, df


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv", parse_dates=["timestamp"])
    X, y, full_df = get_feature_matrix(df)
    print("Feature matrix shape:", X.shape)
    print(X.head())
