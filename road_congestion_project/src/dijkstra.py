"""
dijkstra.py
------------
Classic Dijkstra's shortest-path algorithm implemented with a binary
heap priority queue (heapq). Operates on a graph of the form:

    graph = {
        node: [(neighbor_node, weight, edge_id), ...],
        ...
    }

weight is the (ML-predicted, congestion-aware) travel time in minutes.
"""

import heapq


def dijkstra_shortest_path(graph: dict, source: str, target: str):
    """
    Returns (total_cost, path_nodes, path_edges) for the shortest path
    from source to target. If unreachable, returns (float('inf'), [], []).
    """
    # dist[node] = best known cost from source
    dist = {source: 0.0}
    prev = {}          # node -> (previous_node, edge_id_used)
    visited = set()

    # priority queue of (cost, node)
    pq = [(0.0, source)]

    while pq:
        cost, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == target:
            break

        for v, weight, edge_id in graph.get(u, []):
            if v in visited:
                continue
            new_cost = cost + weight
            if new_cost < dist.get(v, float("inf")):
                dist[v] = new_cost
                prev[v] = (u, edge_id)
                heapq.heappush(pq, (new_cost, v))

    if target not in dist:
        return float("inf"), [], []

    # Reconstruct path
    path_nodes = [target]
    path_edges = []
    node = target
    while node != source:
        prev_node, edge_id = prev[node]
        path_edges.append(edge_id)
        node = prev_node
        path_nodes.append(node)

    path_nodes.reverse()
    path_edges.reverse()

    return dist[target], path_nodes, path_edges


if __name__ == "__main__":
    # Tiny smoke test
    demo_graph = {
        "A": [("B", 2, "e1"), ("C", 5, "e2")],
        "B": [("C", 1, "e3"), ("D", 4, "e4")],
        "C": [("D", 1, "e5")],
        "D": [],
    }
    cost, nodes, edges = dijkstra_shortest_path(demo_graph, "A", "D")
    print(f"Shortest cost A->D: {cost}, path: {nodes}, edges: {edges}")
