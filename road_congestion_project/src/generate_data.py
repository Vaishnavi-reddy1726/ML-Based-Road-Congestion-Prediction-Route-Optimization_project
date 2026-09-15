"""
generate_data.py
-----------------
Generates a synthetic but realistic road network and historical
weather + traffic dataset, since no real dataset was supplied.

Outputs:
    data/road_network.csv       -> edges of the road graph
    data/historical_data.csv    -> historical weather + traffic records per road segment

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_road_network(n_nodes=130, avg_degree=4.3):
    """
    Creates a random, connected, directed road network.
    Each edge represents a road segment with a distance and a free-flow
    (uncongested) speed limit.
    """
    nodes = [f"N{i}" for i in range(n_nodes)]
    edges = []

    # Ensure connectivity with a random spanning structure first
    for i in range(1, n_nodes):
        j = RNG.integers(0, i)
        edges.append((nodes[j], nodes[i]))
        edges.append((nodes[i], nodes[j]))

    # Add extra random edges to reach desired average degree
    n_extra = max(0, int((avg_degree * n_nodes) // 2 - len(edges) // 2))
    for _ in range(n_extra):
        a, b = RNG.choice(nodes, size=2, replace=False)
        edges.append((a, b))
        edges.append((b, a))

    # Deduplicate
    edges = list(set(edges))

    rows = []
    for idx, (u, v) in enumerate(edges):
        distance_km = round(RNG.uniform(0.5, 8.0), 2)
        base_speed_kmph = int(RNG.choice([30, 40, 50, 60, 80]))
        rows.append({
            "road_id": f"R{idx}",
            "from_node": u,
            "to_node": v,
            "distance_km": distance_km,
            "base_speed_kmph": base_speed_kmph,
        })

    return pd.DataFrame(rows)


def generate_historical_data(road_df, days=60, records_per_day=24):
    """
    Simulates hourly weather + traffic observations for every road
    segment over a number of days. Congestion is generated from a
    rule-based latent process (rush hour, weekends, bad weather),
    plus noise -- this is what the ML model will later learn to predict.
    """
    weather_conditions = ["Clear", "Rain", "Fog", "Storm", "Snow"]
    weather_severity = {"Clear": 0.0, "Rain": 0.4, "Fog": 0.5, "Storm": 0.8, "Snow": 0.9}

    records = []
    start = pd.Timestamp("2024-01-01")

    for day in range(days):
        date = start + pd.Timedelta(days=day)
        day_of_week = date.dayofweek  # 0=Mon
        is_weekend = day_of_week >= 5

        # One weather condition per day per rough "zone" -- keep it simple: per day, global weather
        weather = RNG.choice(weather_conditions, p=[0.55, 0.2, 0.1, 0.1, 0.05])
        temperature_c = round(RNG.normal(22 if weather == "Clear" else 15, 5), 1)
        precipitation_mm = round(max(0, RNG.normal(weather_severity[weather] * 10, 2)), 1)
        visibility_km = round(max(0.2, 10 - weather_severity[weather] * 8 + RNG.normal(0, 1)), 1)
        wind_speed_kmph = round(max(0, RNG.normal(10 + weather_severity[weather] * 20, 5)), 1)

        for hour in range(records_per_day):
            # Rush hour effect
            is_rush = hour in (7, 8, 9, 17, 18, 19)
            rush_factor = 0.5 if is_rush else 0.0
            weekend_factor = -0.2 if is_weekend else 0.0
            weather_factor = weather_severity[weather]

            for _, road in road_df.iterrows():
                # Latent congestion score in [0, 1]
                base = 0.15
                noise = RNG.normal(0, 0.05)
                congestion = base + rush_factor + weekend_factor + 0.35 * weather_factor + noise
                congestion = float(np.clip(congestion, 0.0, 1.0))

                records.append({
                    "road_id": road["road_id"],
                    "timestamp": date + pd.Timedelta(hours=hour),
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_weekend),
                    "temperature_c": temperature_c,
                    "precipitation_mm": precipitation_mm,
                    "visibility_km": visibility_km,
                    "wind_speed_kmph": wind_speed_kmph,
                    "weather_condition": weather,
                    "congestion_level": round(congestion, 3),
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating synthetic road network...")
    road_df = generate_road_network(n_nodes=120, avg_degree=4.2)
    road_df.to_csv(os.path.join(OUT_DIR, "road_network.csv"), index=False)
    print(f"  -> {len(road_df)} road segments saved to data/road_network.csv")

    print("Generating synthetic historical weather + traffic data...")
    hist_df = generate_historical_data(road_df, days=60, records_per_day=24)
    hist_df.to_csv(os.path.join(OUT_DIR, "historical_data.csv"), index=False)
    print(f"  -> {len(hist_df)} records saved to data/historical_data.csv")
