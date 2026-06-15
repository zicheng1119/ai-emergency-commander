from copy import deepcopy
from collections import deque

from emergency_commander.contracts import validate_scenario
from emergency_commander.random_scenario import generate_random_scenario


def _reachable(roads, start, goal):
    graph = {}
    for road in roads:
        if road["status"] != "open":
            continue
        graph.setdefault(road["from"], set()).add(road["to"])
        if road.get("bidirectional", True):
            graph.setdefault(road["to"], set()).add(road["from"])
    queue = deque([start])
    visited = {start}
    while queue:
        node = queue.popleft()
        if node == goal:
            return True
        for neighbor in graph.get(node, set()) - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def test_random_scenario_is_reproducible_and_contract_valid():
    first = generate_random_scenario(20260616)
    second = generate_random_scenario(20260616)
    other = generate_random_scenario(20260617)

    assert first == second
    assert first != other
    validate_scenario(first)
    assert first["events"] == []
    assert len(first["zones"]) == 6
    assert len(first["nodes"]) >= 27
    assert 30 <= len(first["roads"]) <= 40
    assert 6 <= len(first["air_routes"]) <= 9
    unit_types = [unit["type"] for unit in first["units"]]
    assert unit_types.count("rescue_car") == 3
    assert unit_types.count("drone") == 2


def test_every_single_ground_road_failure_keeps_zones_reachable():
    scenario = generate_random_scenario(7)
    for failed in scenario["roads"]:
        if failed["road_id"].startswith("R_HOSPITAL"):
            continue
        roads = deepcopy(scenario["roads"])
        target = next(
            road for road in roads if road["road_id"] == failed["road_id"]
        )
        target["status"] = "blocked"
        for zone in scenario["zones"]:
            assert _reachable(roads, "HQ", zone["node_id"]), failed["road_id"]


def test_air_routes_reach_every_zone():
    scenario = generate_random_scenario(9)
    for zone in scenario["zones"]:
        assert _reachable(scenario["air_routes"], "HQ", zone["node_id"])
