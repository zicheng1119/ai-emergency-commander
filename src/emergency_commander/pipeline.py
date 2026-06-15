from __future__ import annotations

from copy import deepcopy
from typing import Any

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.inference import assess_zones
from emergency_commander.input_adapter import normalize_scenario
from emergency_commander.replanning import apply_event


def _public_plan(
    assessments: list[dict[str, Any]], assignments_with_routes: list[dict[str, Any]]
) -> dict[str, Any]:
    assignments = []
    routes = []
    for assignment in assignments_with_routes:
        assignments.append(
            {
                "unit_id": assignment["unit_id"],
                "target_zone": assignment["target_zone"],
                "mission_type": assignment["mission_type"],
                "expected_utility": assignment["expected_utility"],
                "reason": assignment["reason"],
            }
        )
        route = assignment["route"]
        routes.append(
            {
                "unit_id": assignment["unit_id"],
                "target_zone": assignment["target_zone"],
                "path": route["path"],
                "road_ids": route["road_ids"],
                "eta": route["eta"],
                "path_risk": route["path_risk"],
            }
        )
    return {
        "zone_assessment": assessments,
        "assignments": assignments,
        "routes": routes,
    }


def _calculate_plan(scenario: dict[str, Any]) -> dict[str, Any]:
    assessments = assess_zones(scenario)
    matrix = build_utility_matrix(scenario, assessments)
    assignments = allocate_tasks(scenario, matrix)
    return _public_plan(assessments, assignments)


def run_pipeline(raw_scenario: dict[str, Any], *, process_events: bool = True) -> dict[str, Any]:
    """Run inference, allocation, routing, and optional event replanning."""
    scenario = normalize_scenario(raw_scenario)
    current_plan = _calculate_plan(scenario)
    replan_log: list[dict[str, Any]] = []

    if process_events:
        for event in sorted(scenario["events"], key=lambda item: item.get("trigger_step", 0)):
            old_plan = deepcopy(current_plan)
            scenario = normalize_scenario(apply_event(scenario, event))
            current_plan = _calculate_plan(scenario)
            replan_log.append(
                {
                    "trigger_event": deepcopy(event),
                    "old_plan": old_plan,
                    "new_plan": deepcopy(current_plan),
                    "reason": event.get("description", event["event_type"]),
                }
            )

    return {
        "scenario_id": scenario["scenario_id"],
        "run_mode": scenario["run_mode"],
        "weights_used": deepcopy(scenario["config"]["weights"]),
        **current_plan,
        "replan_log": replan_log,
    }

