"""
route_optimizer.py
--------------------
Ties everything together:
    1. Takes current/forecast weather conditions + time.
    2. Builds the feature vector for every road segment.
    3. Uses the trained ML model to predict congestion per segment.
    4. Updates the weighted road-network graph with those predictions.
    5. Runs Dijkstra's algorithm to compute the optimal (fastest) route.
"""

import os
import pandas as pd

from feature_engineering import build_features, FEATURE_COLUMNS, WEATHER_SEVERITY
from graph_network import RoadNetworkGraph
from dijkstra import dijkstra_shortest_path


class RouteOptimizer:
    def __init__(self, model, road_df: pd.DataFrame, road_history_df: pd.DataFrame = None):
        """
        model: trained congestion-prediction model (sklearn-compatible, .predict)
        road_df: road network table (road_id, from_node, to_node, distance_km, base_speed_kmph)
        road_history_df: optional recent history per road, used to compute the
                          rolling_avg_congestion lag feature. If not supplied,
                          a neutral default is used.
        """
        self.model = model
        self.road_df = road_df
        self.graph = RoadNetworkGraph(road_df)
        self.road_history_df = road_history_df

    def _recent_avg_congestion(self, road_id, default=0.3):
        if self.road_history_df is None:
            return default
        recent = self.road_history_df[self.road_history_df["road_id"] == road_id]
        if recent.empty:
            return default
        return recent.sort_values("timestamp")["congestion_level"].tail(3).mean()

    def predict_congestion_for_all_roads(self, weather: dict, hour: int, day_of_week: int):
        """
        weather: {
            "weather_condition": "Rain",
            "temperature_c": 18.0,
            "precipitation_mm": 4.0,
            "visibility_km": 6.0,
            "wind_speed_kmph": 15.0,
        }
        Returns {road_id: predicted_congestion_level}
        """
        rows = []
        road_ids = list(self.road_df["road_id"])

        for road_id in road_ids:
            rows.append({
                "road_id": road_id,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5),
                "temperature_c": weather.get("temperature_c", 20.0),
                "precipitation_mm": weather.get("precipitation_mm", 0.0),
                "visibility_km": weather.get("visibility_km", 10.0),
                "wind_speed_kmph": weather.get("wind_speed_kmph", 5.0),
                "weather_condition": weather.get("weather_condition", "Clear"),
                "congestion_level": self._recent_avg_congestion(road_id),  # used only to seed rolling avg
            })

        batch_df = pd.DataFrame(rows)
        batch_df = build_features(batch_df)

        X = batch_df[FEATURE_COLUMNS].astype(float)
        preds = self.model.predict(X)

        return dict(zip(road_ids, preds))

    def optimize_route(self, origin: str, destination: str, weather: dict,
                        hour: int, day_of_week: int):
        """
        Full pipeline: predict congestion -> update graph -> shortest path.
        Returns a dict with the path, estimated travel time, and per-segment detail.
        """
        congestion_preds = self.predict_congestion_for_all_roads(weather, hour, day_of_week)
        self.graph.update_weights_with_predictions(congestion_preds)

        adjacency = self.graph.get_adjacency_with_weights()
        total_time, path_nodes, path_edges = dijkstra_shortest_path(adjacency, origin, destination)

        segment_details = []
        for road_id in path_edges:
            row = self.road_df[self.road_df["road_id"] == road_id].iloc[0]
            segment_details.append({
                "road_id": road_id,
                "from": row["from_node"],
                "to": row["to_node"],
                "distance_km": row["distance_km"],
                "predicted_congestion": round(congestion_preds.get(road_id, 0.0), 3),
                "travel_time_min": round(self.graph.edge_weight[road_id], 2),
            })

        return {
            "origin": origin,
            "destination": destination,
            "total_travel_time_min": round(total_time, 2) if total_time != float("inf") else None,
            "path_nodes": path_nodes,
            "segments": segment_details,
        }
