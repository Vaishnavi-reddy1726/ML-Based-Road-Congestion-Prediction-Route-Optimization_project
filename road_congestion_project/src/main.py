"""
main.py
--------
End-to-end demo of the full pipeline:

    1. Generate synthetic road network + historical weather/traffic data
       (skipped automatically if data already exists).
    2. Clean the data.
    3. Engineer features.
    4. Train + evaluate the ML congestion-prediction model.
    5. Build the weighted road-network graph.
    6. Predict live congestion for a given weather scenario and run
       Dijkstra's algorithm to compute the optimal route.

Run from the project root:
    python src/main.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from generate_data import generate_road_network, generate_historical_data
from data_preprocessing import clean_data
from model_training import chronological_split, train_model, evaluate_model, save_model
from graph_network import RoadNetworkGraph
from route_optimizer import RouteOptimizer
from classification_model import train_classifier, evaluate_classifier

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")


def step1_get_data():
    road_path = os.path.join(DATA_DIR, "road_network.csv")
    hist_path = os.path.join(DATA_DIR, "historical_data.csv")

    if os.path.exists(road_path) and os.path.exists(hist_path):
        print("[1/6] Found existing data, loading from disk...")
        road_df = pd.read_csv(road_path)
        hist_df = pd.read_csv(hist_path, parse_dates=["timestamp"])
    else:
        print("[1/6] No existing data found -- generating synthetic dataset...")
        road_df = generate_road_network(n_nodes=130, avg_degree=4.3)
        road_df.to_csv(road_path, index=False)
        hist_df = generate_historical_data(road_df, days=60, records_per_day=24)
        hist_df.to_csv(hist_path, index=False)

    print(f"    Road segments: {len(road_df)} | Historical records: {len(hist_df)}")
    return road_df, hist_df


def step2_clean(hist_df):
    print("[2/6] Cleaning data...")
    cleaned = clean_data(hist_df)
    cleaned.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)
    print(f"    Cleaned rows: {len(cleaned)}")
    return cleaned


def step3_train(cleaned_df):
    print("[3/6] Splitting data chronologically and training model...")
    train_df, test_df = chronological_split(cleaned_df, test_frac=0.2)
    model = train_model(train_df, model_type="random_forest")
    print(f"    Trained on {len(train_df)} rows, testing on {len(test_df)} rows")
    return model, train_df, test_df


def step4_evaluate(model, train_df, test_df):
    print("[4/6] Evaluating model on held-out test set...")
    metrics, preds, y_test = evaluate_model(model, test_df)
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\n    Also training the 3-level (Low/Medium/High) congestion classifier...")
    clf = train_classifier(train_df)
    acc, report, cm = evaluate_classifier(clf, test_df)
    print(f"    3-level classification accuracy: {acc * 100:.2f}%")

    return metrics


def step5_save(model):
    print("[5/6] Saving trained model...")
    path = save_model(model)
    print(f"    Saved to {path}")


def step6_optimize_route(model, road_df, cleaned_df):
    print("[6/6] Building graph and optimizing an example route...")
    optimizer = RouteOptimizer(model, road_df, road_history_df=cleaned_df)

    nodes = sorted(optimizer.graph.nodes())
    origin, destination = nodes[0], nodes[-1]

    weather_scenario = {
        "weather_condition": "Rain",
        "temperature_c": 17.0,
        "precipitation_mm": 6.0,
        "visibility_km": 5.0,
        "wind_speed_kmph": 18.0,
    }

    result = optimizer.optimize_route(
        origin=origin,
        destination=destination,
        weather=weather_scenario,
        hour=8,           # 8 AM rush hour
        day_of_week=2,    # Wednesday
    )

    print(f"\n    Scenario: Rain, 8 AM Wednesday rush hour")
    print(f"    Route: {origin} -> {destination}")
    if result["total_travel_time_min"] is None:
        print("    No path found between these nodes.")
    else:
        print(f"    Estimated travel time: {result['total_travel_time_min']} minutes")
        print(f"    Path: {' -> '.join(result['path_nodes'])}")
        print("    Segment breakdown:")
        for seg in result["segments"]:
            print(f"      {seg['from']} -> {seg['to']} | "
                  f"{seg['distance_km']} km | "
                  f"congestion={seg['predicted_congestion']} | "
                  f"time={seg['travel_time_min']} min")

    return result


if __name__ == "__main__":
    road_df, hist_df = step1_get_data()
    cleaned_df = step2_clean(hist_df)
    model, train_df, test_df = step3_train(cleaned_df)
    step4_evaluate(model, train_df, test_df)
    step5_save(model)
    step6_optimize_route(model, road_df, cleaned_df)

    print("\nPipeline complete.")
