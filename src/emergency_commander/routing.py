from __future__ import annotations

import heapq
from collections import defaultdict
from math import inf
from typing import Any


class NoRouteError(RuntimeError):
    """Raised when no traversable path exists between two nodes."""


def road_risk(road: dict[str, Any], risk_weights: dict[str, float]) -> float:
    risk = road["risk"]
    return max(
        0.0,
        min(
            1.0,
            risk_weights["fire"] * risk["fire"]
            + risk_weights["damage"] * risk["damage"]
            + risk_weights["congestion"] * risk["congestion"]
            + risk_weights["secondary"] * risk["secondary_disaster"],
        ),
    )


def risk_aware_astar(
    roads: list[dict[str, Any]],
    *,
    start: str,
    goal: str,
    speed: float,
    risk_weights: dict[str, float],
    unit_type: str = "rescue_car",
    max_fire_risk: float | None = None,
) -> dict[str, Any]:
    """Find the minimum time-and-risk route.

    The input contract has no node coordinates, so the admissible heuristic is
    zero. This is still A* and is equivalent to Dijkstra for this data shape.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    if start == goal:
        return {"path": [start], "road_ids": [], "eta": 0.0, "path_risk": 0.0, "total_cost": 0.0}

    adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for road in roads:
        if road.get("status", "open") == "blocked" and unit_type != "drone":
            continue
        if (
            unit_type != "drone"
            and max_fire_risk is not None
            and road["risk"]["fire"] > max_fire_risk
        ):
            continue
        adjacency[road["from"]].append((road["to"], road))
        if road.get("bidirectional", True):
            adjacency[road["to"]].append((road["from"], road))

    frontier: list[tuple[float, str]] = [(0.0, start)]
    best_cost = {start: 0.0}
    previous: dict[str, tuple[str, dict[str, Any]]] = {}

    while frontier:
        current_cost, node = heapq.heappop(frontier)
        if current_cost > best_cost.get(node, inf):
            continue
        if node == goal:
            break
        for neighbor, road in adjacency.get(node, []):
            base_time = float(road["travel_time_base"]) / speed
            risk = road_risk(road, risk_weights)
            if unit_type == "drone":
                risk *= 0.25
            traversal_cost = base_time * (1.0 + risk)
            candidate_cost = current_cost + traversal_cost
            if candidate_cost < best_cost.get(neighbor, inf):
                best_cost[neighbor] = candidate_cost
                previous[neighbor] = (node, road)
                heapq.heappush(frontier, (candidate_cost, neighbor))

    if goal not in best_cost:
        raise NoRouteError(f"no route from {start} to {goal}")

    nodes = [goal]
    selected_roads: list[dict[str, Any]] = []
    cursor = goal
    while cursor != start:
        prior_node, road = previous[cursor]
        nodes.append(prior_node)
        selected_roads.append(road)
        cursor = prior_node
    nodes.reverse()
    selected_roads.reverse()

    eta = sum(float(road["travel_time_base"]) / speed for road in selected_roads)
    if eta:
        weighted_risk = sum(
            road_risk(road, risk_weights) * (float(road["travel_time_base"]) / speed)
            for road in selected_roads
        ) / eta
    else:
        weighted_risk = 0.0
    if unit_type == "drone":
        weighted_risk *= 0.25

    return {
        "path": nodes,
        "road_ids": [road["road_id"] for road in selected_roads],
        "eta": round(eta, 6),
        "path_risk": round(weighted_risk, 6),
        "total_cost": round(best_cost[goal], 6),
    }
