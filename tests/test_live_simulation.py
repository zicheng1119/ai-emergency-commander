from copy import deepcopy

import pytest

from emergency_commander.live_simulation import LiveSimulation
from emergency_commander.random_scenario import generate_random_scenario


def _advance_to_phase(session, phase, limit=30):
    for _ in range(limit):
        if session.phase == phase:
            return
        session.step()
    raise AssertionError(f"session did not reach phase {phase}")


def test_live_session_starts_without_precomputing_and_advances_phases():
    session = LiveSimulation.create(generate_random_scenario(11), seed=11)

    assert session.phase == "validate"
    assert session.clock_minutes == 0
    assert session.timeline == []
    assert session.current_plan == {}
    assert session.calculation_history == []

    observed = []
    for _ in range(7):
        observed.append(session.phase)
        session.step()

    assert observed == [
        "validate",
        "infer",
        "prioritize",
        "route",
        "utility",
        "allocate",
        "execute",
    ]
    assert session.current_plan["assignments"]
    assert [entry["phase"] for entry in session.algorithm_log[:6]] == observed[:6]
    assert [entry["phase"] for entry in session.calculation_history] == observed
    assert set(session.calculation_history[0]) == {
        "index",
        "phase",
        "title",
        "clock_minutes",
        "summary",
        "focus",
        "inputs",
        "operations",
        "outputs",
    }
    assert session.calculation_history[1]["outputs"]["zones"]
    route_record = session.calculation_history[3]
    assert route_record["outputs"]["candidates"]
    first_feasible = next(
        candidate
        for candidate in route_record["outputs"]["candidates"]
        if candidate.get("route")
    )
    assert route_record["focus"]["roads"] == first_feasible["route"]["road_ids"]
    assert route_record["focus"]["zones"] == [first_feasible["target_zone"]]


def test_execute_step_changes_clock_and_unit_position():
    session = LiveSimulation.create(generate_random_scenario(12), seed=12)
    _advance_to_phase(session, "execute")
    before = {
        unit_id: dict(state["position"])
        for unit_id, state in session.unit_states.items()
    }

    session.step(execution_minutes=1.0)

    assert session.clock_minutes == 1.0
    assert any(
        state["position"] != before[unit_id]
        for unit_id, state in session.unit_states.items()
    )
    assert session.timeline[-1]["clock_minutes"] == 1.0


def test_execute_can_advance_directly_to_next_unit_transition():
    session = LiveSimulation.create(generate_random_scenario(120), seed=120)
    _advance_to_phase(session, "execute")
    expected = min(
        state["remaining_travel"]
        for state in session.unit_states.values()
        if state["status"] in {"en_route", "returning"}
        and state["remaining_travel"] > 0
    )

    session.step(to_next_transition=True)

    assert session.clock_minutes == pytest.approx(expected)
    record = session.calculation_history[-1]
    assert record["phase"] == "execute"
    assert record["inputs"]["elapsed_minutes"] == pytest.approx(expected)
    assert any(
        before["status"] != after["status"]
        for before, after in zip(
            record["operations"]["before"].values(),
            record["operations"]["after"].values(),
        )
    )


def test_live_session_round_trips_without_recomputing():
    session = LiveSimulation.create(generate_random_scenario(13), seed=13)
    _advance_to_phase(session, "execute")
    payload = session.to_dict()

    restored = LiveSimulation.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.phase == "execute"
    assert restored.calculation_history == session.calculation_history


@pytest.mark.parametrize(
    "event_type", ["road_collapse", "fire_spread", "new_sos", "drone_update"]
)
def test_supported_event_interrupts_and_replans(event_type):
    session = LiveSimulation.create(generate_random_scenario(21), seed=21)
    _advance_to_phase(session, "execute")
    old_plan = deepcopy(session.current_plan)

    event = session.inject_event(event_type)

    assert event["event_type"] == event_type
    assert session.phase == "replan"
    assert len(session.event_log) == 1

    session.step()
    assert session.phase == "infer"
    _advance_to_phase(session, "execute")
    assert session.replan_log[-1]["old_plan"] == old_plan
    assert session.replan_log[-1]["new_plan"] == session.current_plan
    assert session.status == "running"


def test_manual_event_target_is_used_and_duplicate_event_is_not_reapplied():
    session = LiveSimulation.create(generate_random_scenario(22), seed=22)
    _advance_to_phase(session, "execute")
    target = session.available_event_targets("fire_spread")[0]

    event = session.inject_event("fire_spread", target_id=target)
    fire = next(
        zone for zone in session.scenario["zones"] if zone["zone_id"] == target
    )["observations"]["fire"]

    with pytest.raises(ValueError, match="already applied"):
        session.inject_event_payload(event)
    assert next(
        zone for zone in session.scenario["zones"] if zone["zone_id"] == target
    )["observations"]["fire"] == fire


def test_road_collapse_invalidates_only_tasks_using_that_road():
    session = LiveSimulation.create(generate_random_scenario(23), seed=23)
    _advance_to_phase(session, "execute")
    active_ground = {
        unit_id: deepcopy(state["current_task"])
        for unit_id, state in session.unit_states.items()
        if state.get("current_task")
        and state["current_task"]["route"]["route_layer"] == "ground"
    }
    assert len(active_ground) >= 2
    affected_unit, affected_task = next(iter(active_ground.items()))
    target_road = affected_task["route"]["road_ids"][0]

    session.inject_event("road_collapse", target_id=target_road)

    assert session.unit_states[affected_unit]["current_task"] is None
    for unit_id, task in active_ground.items():
        if target_road not in task["route"]["road_ids"]:
            assert session.unit_states[unit_id]["current_task"] == task


def test_paused_session_does_not_advance_and_can_resume():
    session = LiveSimulation.create(generate_random_scenario(31), seed=31)
    _advance_to_phase(session, "execute")
    session.pause()
    frozen = session.to_dict()

    session.step()

    assert session.to_dict() == frozen
    session.resume()
    session.step()
    assert session.clock_minutes == 1.0


def test_event_can_be_queued_while_paused_then_replans_after_resume():
    session = LiveSimulation.create(generate_random_scenario(32), seed=32)
    _advance_to_phase(session, "execute")
    session.pause()

    session.inject_event("fire_spread")

    assert session.status == "paused"
    assert session.phase == "replan"
    session.resume()
    session.step()
    assert session.phase == "infer"


def test_final_report_counts_delivered_rescues_not_drone_recon():
    session = LiveSimulation.create(generate_random_scenario(33), seed=33)
    for _ in range(500):
        if session.status == "completed":
            break
        session.step()
    assert session.status == "completed"

    report = session.build_result()
    delivered = {
        zone_id
        for state in session.unit_states.values()
        for zone_id in state.get("delivered_targets", [])
    }

    assert set(report["completed_zones"]) == delivered
    assert report["rescued_people"] == sum(
        state.get("rescued_people", 0) for state in session.unit_states.values()
    )
    assert report["end_reason"] in {
        "all_rescues_complete",
        "no_feasible_tasks",
        "timeout",
    }
    assert report["algorithm_log"]
    assert report["timeline"]
    assert report["seed"] == 33
