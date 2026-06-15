from __future__ import annotations

from copy import deepcopy
from typing import Any

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.bayesian_network import DiscreteBayesianNetwork
from emergency_commander.contracts import validate_decision_output
from emergency_commander.inference import assess_zones
from emergency_commander.input_adapter import normalize_scenario
from emergency_commander.replanning import apply_event
from emergency_commander.simulation import (
    advance_unit_states,
    initialize_unit_states,
    start_assignments,
)


def _public_plan(
    assessments: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    utility_matrix: list[dict[str, Any]],
) -> dict[str, Any]:
    assignments = []
    routes = []
    for state in states.values():
        task = state.get("current_task")
        if not task:
            continue
        route = task["route"]
        if task.get("target_zone") is not None:
            assignments.append(
                {
                    "unit_id": state["unit_id"],
                    "target_zone": task["target_zone"],
                    "mission_type": task["mission_type"],
                    "expected_utility": task.get("expected_utility"),
                    "estimated_people": task.get("estimated_people", 0),
                    "reason": task.get("reason", "active_mission"),
                    "feasibility_reason": task.get("feasibility_reason", "feasible"),
                    "explanation": task.get("explanation", task.get("reason", "active_mission")),
                    "resource_cost": task.get("resource_cost", state.get("resource_cost", 0.0)),
                    "utility_inputs": deepcopy(task.get("utility_inputs")),
                    "utility_breakdown": deepcopy(task.get("utility_breakdown")),
                    "unit_status": state["status"],
                }
            )
        routes.append(
            {
                "unit_id": state["unit_id"],
                "target_zone": task.get("target_zone"),
                "mission_type": task["mission_type"],
                "path": route["path"],
                "road_ids": route["road_ids"],
                "eta": route["eta"],
                "remaining_eta": state["remaining_travel"],
                "path_risk": route["path_risk"],
                "route_layer": route.get("route_layer", "ground"),
                "heuristic": route.get("heuristic", "zero_fallback"),
                "expanded_nodes": route.get("expanded_nodes", 0),
            }
        )
    return {
        "zone_assessment": assessments,
        "assignments": sorted(assignments, key=lambda item: item["unit_id"]),
        "routes": sorted(routes, key=lambda item: item["unit_id"]),
        "utility_matrix": deepcopy(utility_matrix),
    }


def _plan_idle_units(
    scenario: dict[str, Any],
    states: dict[str, dict[str, Any]],
    network: DiscreteBayesianNetwork | None,
    model_name: str,
) -> dict[str, Any]:
    assessments = assess_zones(scenario, network, model_name=model_name)
    active_zones = {
        state["current_task"]["target_zone"]
        for state in states.values()
        if state.get("current_task") and state["current_task"].get("target_zone")
    }
    planning = deepcopy(scenario)
    planning["units"] = []
    for unit in scenario["units"]:
        state = states[unit["unit_id"]]
        if state["status"] != "idle":
            continue
        available = deepcopy(unit)
        available["start_node"] = state["current_node"]
        planning["units"].append(available)
    eligible_assessments = [
        assessment for assessment in assessments if assessment["zone_id"] not in active_zones
    ]
    matrix = (
        build_utility_matrix(planning, eligible_assessments)
        if planning["units"] and eligible_assessments
        else []
    )
    matrix = [
        candidate
        for candidate in matrix
        if candidate["target_zone"]
        not in states[candidate["unit_id"]].get("completed_targets", [])
    ]
    new_assignments = allocate_tasks(planning, matrix) if matrix else []
    assessment_by_zone = {item["zone_id"]: item for item in assessments}
    units = {unit["unit_id"]: unit for unit in scenario["units"]}
    for assignment in new_assignments:
        unit = units[assignment["unit_id"]]
        trapped = assessment_by_zone[assignment["target_zone"]]["trapped_prob"]
        capacity = int(unit.get("capacity", 4 if unit["type"] == "rescue_car" else 0))
        assignment["estimated_people"] = (
            max(1, round(trapped * capacity))
            if assignment["mission_type"] == "rescue" and trapped >= 0.5
            else 0
        )
    start_assignments(states, new_assignments, scenario)
    return _public_plan(assessments, states, matrix)


def _invalidate_affected_missions(
    states: dict[str, dict[str, Any]], event: dict[str, Any]
) -> None:
    if event["event_type"] != "road_collapse":
        return
    collapsed_route = event["target_id"]
    for state in states.values():
        task = state.get("current_task")
        if (
            state["status"] == "en_route"
            and task
            and task["route"].get("route_layer", "ground") == "ground"
            and collapsed_route in task["route"]["road_ids"]
        ):
            state["status"] = "idle"
            state["current_task"] = None
            state["remaining_travel"] = 0.0
            state["remaining_service"] = 0.0


def run_pipeline(
    raw_scenario: dict[str, Any],
    *,
    process_events: bool = True,
    network: DiscreteBayesianNetwork | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run Bayesian inference, stateful dispatch, routing, and event replanning."""
    scenario = normalize_scenario(raw_scenario)
    model_name = model_name or (
        "learned_cpt" if scenario["run_mode"] == "learned" else "expert_cpt"
    )
    states = initialize_unit_states(scenario)
    simulation_clock = 0.0
    current_plan = _plan_idle_units(scenario, states, network, model_name)
    timeline = [
        {
            "step": 0,
            "clock_minutes": simulation_clock,
            "event": None,
            "plan": deepcopy(current_plan),
            "unit_states": deepcopy(states),
            "scenario_state": deepcopy(scenario),
        }
    ]
    replan_log: list[dict[str, Any]] = []

    if process_events:
        for step, event in enumerate(
            sorted(scenario["events"], key=lambda item: item.get("trigger_step", 0)),
            start=1,
        ):
            old_plan = deepcopy(current_plan)
            elapsed = float(event.get("elapsed_minutes", 0.0))
            advance_unit_states(states, elapsed, scenario)
            simulation_clock += elapsed
            scenario = normalize_scenario(apply_event(scenario, event))
            _invalidate_affected_missions(states, event)
            current_plan = _plan_idle_units(scenario, states, network, model_name)
            snapshot = {
                "step": step,
                "clock_minutes": round(simulation_clock, 6),
                "event": deepcopy(event),
                "plan": deepcopy(current_plan),
                "unit_states": deepcopy(states),
                "scenario_state": deepcopy(scenario),
            }
            timeline.append(snapshot)
            replan_log.append(
                {
                    "trigger_event": deepcopy(event),
                    "clock_minutes": round(simulation_clock, 6),
                    "old_plan": old_plan,
                    "new_plan": deepcopy(current_plan),
                    "reason": event.get("description", event["event_type"]),
                }
            )

    output = {
        "scenario_id": scenario["scenario_id"],
        "run_mode": scenario["run_mode"],
        "bayesian_model": model_name,
        "weights_used": deepcopy(scenario["config"]["weights"]),
        **current_plan,
        "unit_states": states,
        "simulation_clock": round(simulation_clock, 6),
        "timeline": timeline,
        "replan_log": replan_log,
    }
    validate_decision_output(output)
    return output
