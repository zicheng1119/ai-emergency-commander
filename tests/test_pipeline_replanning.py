from emergency_commander.pipeline import run_pipeline
from emergency_commander.replanning import apply_event


def scenario_with_collapse():
    return {
        "scenario_id": "collapse_case",
        "generated_at": "2026-06-15T20:00:00+08:00",
        "mode": "fixed",
        "command_center": {"node_id": "HQ"},
        "hospital": {"node_id": "HOSPITAL"},
        "nodes": {
            "HQ": {"x": 0.0, "y": 0.0},
            "X": {"x": 2.0, "y": 2.0},
            "ZONE_A": {"x": 4.0, "y": 0.0},
            "HOSPITAL": {"x": -2.0, "y": 0.0}
        },
        "config": {
            "weights": {
                "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
                "passability": {"road_damage": 0.45, "fire_risk": 0.35, "congestion": 0.20, "drone_confidence": 0.0},
                "life_risk": {"fire": 0.40, "trapped_prob": 0.35, "time_urgency": 0.25},
                "priority": {"trapped_prob": 0.40, "life_risk": 0.30, "time_urgency": 0.20, "accessibility": 0.10},
                "utility": {"alpha": 0.30, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "epsilon": 0.10},
                "astar_risk": {"fire": 0.35, "damage": 0.25, "congestion": 0.20, "secondary": 0.20},
            },
            "thresholds": {"car_min_passability": 0.45, "drone_recon_priority_risk": 0.70},
        },
        "zones": [
            {
                "zone_id": "A",
                "node_id": "ZONE_A",
                "observations": {
                    "sos_signal": 0.90,
                    "building_collapse": 0.80,
                    "smoke": 0.30,
                    "fire": 0.20,
                    "road_damage": 0.10,
                    "human_activity": 0.80,
                    "congestion": 0.10,
                    "time_urgency": 0.90,
                    "drone_confidence": 0.0,
                },
            }
        ],
        "roads": [
            {
                "road_id": "direct",
                "from": "HQ",
                "to": "ZONE_A",
                "distance": 2.0,
                "travel_time_base": 2.0,
                "status": "open",
                "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
            },
            {
                "road_id": "detour_1",
                "from": "HQ",
                "to": "X",
                "distance": 2.0,
                "travel_time_base": 2.0,
                "status": "open",
                "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
            },
            {
                "road_id": "detour_2",
                "from": "X",
                "to": "ZONE_A",
                "distance": 2.0,
                "travel_time_base": 2.0,
                "status": "open",
                "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
            },
            {
                "road_id": "hospital_hq",
                "from": "HOSPITAL",
                "to": "HQ",
                "distance": 2.0,
                "travel_time_base": 2.0,
                "status": "open",
                "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
            },
        ],
        "air_routes": [],
        "units": [
            {
                "unit_id": "RescueCar-1",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 1.0,
                "can_transport": True,
                "capacity": 4,
                "service_time": 1.0,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            }
        ],
        "events": [
            {
                "event_id": "EVT_01",
                "event_type": "road_collapse",
                "trigger_step": 1,
                "elapsed_minutes": 1.0,
                "target_id": "direct",
                "changes": {"status": "blocked", "risk.damage": 1.0},
                "description": "direct road collapsed",
            }
        ],
    }


def test_pipeline_replans_after_road_collapse():
    output = run_pipeline(scenario_with_collapse(), process_events=True)

    assert output["run_mode"] == "fixed"
    assert output["routes"][0]["path"] == ["HQ", "X", "ZONE_A"]
    assert output["replan_log"][0]["trigger_event"]["event_type"] == "road_collapse"
    assert output["replan_log"][0]["old_plan"]["routes"][0]["path"] == ["HQ", "ZONE_A"]
    assert output["replan_log"][0]["new_plan"]["routes"][0]["path"] == ["HQ", "X", "ZONE_A"]
    assert output["simulation_clock"] == 1.0
    assert len(output["timeline"]) == 2
    assert output["timeline"][0]["unit_states"]["RescueCar-1"]["status"] == "en_route"
    assert output["timeline"][1]["unit_states"]["RescueCar-1"]["current_node"] == "HQ"


def test_completed_drone_target_is_not_immediately_assigned_again():
    scenario = scenario_with_collapse()
    scenario["units"] = [
        {
            "unit_id": "Drone-1",
            "type": "drone",
            "start_node": "HQ",
            "speed": 2.0,
            "can_transport": False,
            "capacity": 0,
            "service_time": 0.0,
            "constraints": {},
        }
    ]
    scenario["air_routes"] = [
        {
            "road_id": "air_a",
            "from": "HQ",
            "to": "ZONE_A",
            "distance": 2.0,
            "travel_time_base": 1.0,
            "status": "open",
            "risk": {"fire": 0.0, "damage": 0.0, "congestion": 0.0, "secondary_disaster": 0.0},
        }
    ]
    scenario["events"] = [
        {
            "event_id": "drone_done",
            "event_type": "drone_update",
            "trigger_step": 1,
            "elapsed_minutes": 2.0,
            "target_id": "A",
            "changes": {"observations.drone_confidence": 1.0},
            "description": "drone finished A",
        }
    ]

    output = run_pipeline(scenario, process_events=True)

    drone = output["unit_states"]["Drone-1"]
    assert drone["status"] == "idle"
    assert drone["completed_targets"] == ["A"]
    assert output["assignments"] == []


def test_all_supported_zone_events_apply_dot_path_changes():
    for event_type, field, value in [
        ("drone_update", "observations.drone_confidence", 0.95),
        ("new_sos", "observations.sos_signal", 1.0),
        ("fire_spread", "observations.fire", 0.90),
    ]:
        scenario = scenario_with_collapse()
        event = {
            "event_id": f"event_{event_type}",
            "event_type": event_type,
            "target_id": "A",
            "changes": {field: value},
            "description": event_type,
        }

        updated = apply_event(scenario, event)

        leaf = field.split(".")[-1]
        assert updated["zones"][0]["observations"][leaf] == value
        assert scenario["zones"][0]["observations"][leaf] != value
