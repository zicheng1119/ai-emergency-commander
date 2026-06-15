import pytest

from emergency_commander.allocation import allocate_tasks, build_utility_matrix
from emergency_commander.inference import assess_zones
from emergency_commander.routing import NoRouteError, risk_aware_astar


RISK_WEIGHTS = {
    "fire": 0.35,
    "damage": 0.25,
    "congestion": 0.20,
    "secondary": 0.20,
}


def road(road_id, start, end, travel_time, *, fire=0.0, damage=0.0, status="open"):
    return {
        "road_id": road_id,
        "from": start,
        "to": end,
        "distance": travel_time,
        "travel_time_base": travel_time,
        "status": status,
        "bidirectional": True,
        "risk": {
            "fire": fire,
            "damage": damage,
            "congestion": 0.0,
            "secondary_disaster": 0.0,
        },
    }


def test_risk_aware_astar_chooses_safer_detour():
    roads = [
        road("direct", "HQ", "ZONE_A", 2.0, fire=1.0),
        road("detour_1", "HQ", "X", 1.2),
        road("detour_2", "X", "ZONE_A", 1.2),
    ]

    route = risk_aware_astar(
        roads,
        start="HQ",
        goal="ZONE_A",
        speed=1.0,
        risk_weights=RISK_WEIGHTS,
    )

    assert route["path"] == ["HQ", "X", "ZONE_A"]
    assert route["eta"] == pytest.approx(2.4)
    assert route["path_risk"] == pytest.approx(0.0)


def test_risk_aware_astar_rejects_blocked_only_route():
    roads = [road("collapsed", "HQ", "ZONE_A", 2.0, status="blocked")]

    with pytest.raises(NoRouteError, match="HQ.*ZONE_A"):
        risk_aware_astar(
            roads,
            start="HQ",
            goal="ZONE_A",
            speed=1.0,
            risk_weights=RISK_WEIGHTS,
        )


def test_rescue_route_rejects_road_above_unit_fire_limit():
    roads = [road("burning", "HQ", "ZONE_A", 2.0, fire=0.9)]

    with pytest.raises(NoRouteError, match="HQ.*ZONE_A"):
        risk_aware_astar(
            roads,
            start="HQ",
            goal="ZONE_A",
            speed=1.0,
            risk_weights=RISK_WEIGHTS,
            max_fire_risk=0.7,
        )


def allocation_scenario():
    observations = {
        "A": (0.95, 0.85, 0.55, 0.50, 0.20, 0.80),
        "B": (0.30, 0.20, 0.20, 0.10, 0.10, 0.35),
        "C": (0.80, 0.70, 0.90, 0.85, 0.30, 0.90),
    }
    zones = []
    for zone_id, (sos, collapse, fire, damage, congestion, urgency) in observations.items():
        zones.append(
            {
                "zone_id": zone_id,
                "node_id": f"ZONE_{zone_id}",
                "observations": {
                    "sos_signal": sos,
                    "building_collapse": collapse,
                    "smoke": fire,
                    "fire": fire,
                    "road_damage": damage,
                    "human_activity": sos * 0.7,
                    "congestion": congestion,
                    "time_urgency": urgency,
                    "drone_confidence": 0.0,
                },
            }
        )

    return {
        "config": {
            "weights": {
                "trapped": {"sos": 0.35, "collapse": 0.30, "human_activity": 0.20, "smoke": 0.15},
                "passability": {"road_damage": 0.45, "fire_risk": 0.35, "congestion": 0.20, "drone_confidence": 0.0},
                "life_risk": {"fire": 0.40, "trapped_prob": 0.35, "time_urgency": 0.25},
                "priority": {"trapped_prob": 0.40, "life_risk": 0.30, "time_urgency": 0.20, "accessibility": 0.10},
                "utility": {"alpha": 0.30, "beta": 0.25, "gamma": 0.20, "delta": 0.15, "epsilon": 0.10},
                "astar_risk": RISK_WEIGHTS,
            },
            "thresholds": {"car_min_passability": 0.45, "drone_recon_priority_risk": 0.70},
        },
        "zones": zones,
        "roads": [
            road("to_a", "HQ", "ZONE_A", 5.0, fire=0.2),
            road("to_b", "HQ", "ZONE_B", 4.0),
            road("to_c", "HQ", "ZONE_C", 3.0, fire=0.8, damage=0.6),
        ],
        "units": [
            {
                "unit_id": "RescueCar-1",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 1.2,
                "can_transport": True,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
            {
                "unit_id": "RescueCar-2",
                "type": "rescue_car",
                "start_node": "HQ",
                "speed": 1.0,
                "can_transport": True,
                "constraints": {"max_fire_risk": 0.70, "min_passability": 0.45},
            },
            {
                "unit_id": "Drone-1",
                "type": "drone",
                "start_node": "HQ",
                "speed": 2.0,
                "can_transport": False,
                "constraints": {},
            },
        ],
    }


def test_allocator_sends_drone_to_high_risk_low_accessibility_zone():
    scenario = allocation_scenario()
    assessments = assess_zones(scenario)

    matrix = build_utility_matrix(scenario, assessments)
    assignments = allocate_tasks(scenario, matrix)

    by_unit = {item["unit_id"]: item for item in assignments}
    assert by_unit["Drone-1"]["target_zone"] == "C"
    assert by_unit["Drone-1"]["mission_type"] == "reconnaissance"
    assert {by_unit["RescueCar-1"]["target_zone"], by_unit["RescueCar-2"]["target_zone"]} == {"A", "B"}
    assert all(item["target_zone"] != "C" for item in assignments if item["mission_type"] == "rescue")
