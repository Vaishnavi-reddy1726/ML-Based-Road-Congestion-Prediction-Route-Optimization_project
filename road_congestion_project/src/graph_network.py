"""
graph_network.py
------------------
Builds a weighted, directed road-network graph from the road segment
table, and provides a method to dynamically refresh edge weights
(travel times) using ML-predicted congestion levels.

Edge weight = travel time in minutes, computed as:

    effective_speed = base_speed_kmph * (1 - congestion_level * SLOWDOWN_FACTOR)
    travel_time_min = (distance_km / effective_speed) * 60

Higher congestion -> lower effective speed -> higher travel time.
"""

from collections import defaultdict
import pandas as pd

SLOWDOWN_FACTOR = 0.85  # at congestion_level = 1.0, speed drops by 85%
MIN_SPEED_KMPH = 5.0    # floor so travel time never becomes infinite


class RoadNetworkGraph:
    def __init__(self, road_df: pd.DataFrame):
        """
        road_df must contain: road_id, from_node, to_node, distance_km, base_speed_kmph
        """
        self.road_df = road_df.set_index("road_id")
        self.adjacency = defaultdict(list)   # node -> list of (neighbor, road_id)
        self.edge_weight = {}                # road_id -> current travel time (minutes)
        self._build_topology()
        self.reset_to_free_flow()

    def _build_topology(self):
        for road_id, row in self.road_df.iterrows():
            self.adjacency[row["from_node"]].append((row["to_node"], road_id))

    def reset_to_free_flow(self):
        """Initialize all edge weights assuming zero congestion."""
        for road_id, row in self.road_df.iterrows():
            travel_time = (row["distance_km"] / row["base_speed_kmph"]) * 60
            self.edge_weight[road_id] = travel_time

    def update_weights_with_predictions(self, congestion_by_road: dict):
        """
        congestion_by_road: {road_id: predicted_congestion_level in [0,1]}
        Recomputes travel time for every road segment using the ML predictions.
        """
        for road_id, row in self.road_df.iterrows():
            congestion = congestion_by_road.get(road_id, 0.0)
            congestion = max(0.0, min(1.0, congestion))

            effective_speed = row["base_speed_kmph"] * (1 - congestion * SLOWDOWN_FACTOR)
            effective_speed = max(effective_speed, MIN_SPEED_KMPH)

            travel_time_min = (row["distance_km"] / effective_speed) * 60
            self.edge_weight[road_id] = travel_time_min

    def get_adjacency_with_weights(self):
        """
        Returns graph in the form expected by the Dijkstra solver:
            {node: [(neighbor_node, weight, road_id), ...], ...}
        """
        graph = defaultdict(list)
        for u, neighbors in self.adjacency.items():
            for v, road_id in neighbors:
                graph[u].append((v, self.edge_weight[road_id], road_id))
        return graph

    def nodes(self):
        return set(self.road_df["from_node"]).union(set(self.road_df["to_node"]))


if __name__ == "__main__":
    road_df = pd.read_csv("data/road_network.csv")
    g = RoadNetworkGraph(road_df)
    print(f"Nodes: {len(g.nodes())}  Edges: {len(g.edge_weight)}")
    sample_graph = g.get_adjacency_with_weights()
    first_node = next(iter(sample_graph))
    print(f"Sample adjacency for {first_node}: {sample_graph[first_node]}")
