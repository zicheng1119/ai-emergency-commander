from __future__ import annotations

from itertools import product
from typing import Any

from emergency_commander.routing import NoRouteError, risk_aware_astar


def _zone_lookup(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {zone["zone_id"]: zone for zone in scenario["zones"]}


def build_utility_matrix(
    scenario: dict[str, Any], assessments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build explainable unit-zone candidates, including rejected options."""
    utility_weights = scenario["config"]["weights"]["utility"]
    risk_weights = scenario["config"]["weights"]["astar_risk"]
    zones = _zone_lookup(scenario)
    candidates: list[dict[str, Any]] = []

    for unit in scenario["units"]:
        is_drone = unit["type"] == "drone"
        for assessment in assessments:
            zone = zones[assessment["zone_id"]]
            constraints = unit.get("constraints", {})
            feasible = True
            reason = "feasible"
            if not is_drone and assessment["passability_prob"] < constraints.get("min_passability", 0.45):
                feasible = False
                reason = "passability_below_vehicle_minimum"
            elif not is_drone and zone["observations"]["fire"] > constraints.get("max_fire_risk", 1.0):
                feasible = False
                reason = "fire_risk_above_vehicle_maximum"

            route = None
            if feasible:
                try:
                    route = risk_aware_astar(
                        scenario["roads"],
                        start=unit["start_node"],
                        goal=assessment["node_id"],
                        speed=float(unit["speed"]),
                        risk_weights=risk_weights,
                        unit_type=unit["type"],
                        max_fire_risk=constraints.get("max_fire_risk") if not is_drone else None,
                    )
                except NoRouteError:
                    feasible = False
                    reason = "no_route"

            expected_utility = None
            if feasible and route is not None:
                accessibility_value = (
                    1.0 - assessment["passability_prob"] if is_drone else assessment["passability_prob"]
                )
                arrival_time_normalized = min(route["eta"] / 20.0, 1.0)
                expected_utility = (
                    utility_weights["alpha"] * assessment["trapped_prob"]
                    + utility_weights["beta"] * assessment["life_risk"]
                    + utility_weights["gamma"] * accessibility_value
                    - utility_weights["delta"] * arrival_time_normalized
                    - utility_weights["epsilon"] * route["path_risk"]
                )

            candidates.append(
                {
                    "unit_id": unit["unit_id"],
                    "unit_type": unit["type"],
                    "target_zone": assessment["zone_id"],
                    "mission_type": "reconnaissance" if is_drone else "rescue",
                    "feasible": feasible,
                    "reason": reason,
                    "expected_utility": round(expected_utility, 6) if expected_utility is not None else None,
                    "route": route,
                }
            )

    return candidates


def allocate_tasks(
    scenario: dict[str, Any], utility_matrix: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Maximize total utility by enumerating the small demo assignment space."""
    by_unit: dict[str, list[dict[str, Any] | None]] = {}
    for unit in scenario["units"]:
        unit_candidates = [
            item
            for item in utility_matrix
            if item["unit_id"] == unit["unit_id"] and item["feasible"]
        ]
        by_unit[unit["unit_id"]] = [None, *unit_candidates]

    unit_ids = [unit["unit_id"] for unit in scenario["units"]]
    best_total = float("-inf")
    best_combination: tuple[dict[str, Any] | None, ...] | None = None
    for combination in product(*(by_unit[unit_id] for unit_id in unit_ids)):
        zones = [item["target_zone"] for item in combination if item is not None]
        if len(zones) != len(set(zones)):
            continue
        total = sum(item["expected_utility"] for item in combination if item is not None)
        if total > best_total:
            best_total = total
            best_combination = combination

    assignments: list[dict[str, Any]] = []
    if best_combination is None:
        return assignments
    for item in best_combination:
        if item is None:
            continue
        assignments.append(
            {
                "unit_id": item["unit_id"],
                "target_zone": item["target_zone"],
                "mission_type": item["mission_type"],
                "expected_utility": item["expected_utility"],
                "reason": item["reason"],
                "route": item["route"],
            }
        )
    return assignments
