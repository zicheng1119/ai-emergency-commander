from __future__ import annotations

from typing import Any


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def assess_zones(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Calculate explainable probability, risk, and priority values per zone."""
    weights = scenario["config"]["weights"]
    trapped_weights = weights["trapped"]
    passability_weights = weights["passability"]
    life_weights = weights["life_risk"]
    priority_weights = weights["priority"]
    assessments: list[dict[str, Any]] = []

    for zone in scenario["zones"]:
        obs = zone["observations"]
        trapped_prob = _clip(
            trapped_weights["sos"] * obs["sos_signal"]
            + trapped_weights["collapse"] * obs["building_collapse"]
            + trapped_weights["human_activity"] * obs["human_activity"]
            + trapped_weights["smoke"] * obs["smoke"]
        )
        passability_prob = _clip(
            1.0
            - passability_weights["road_damage"] * obs["road_damage"]
            - passability_weights["fire_risk"] * obs["fire"]
            - passability_weights["congestion"] * obs["congestion"]
            + passability_weights.get("drone_confidence", 0.0) * obs["drone_confidence"]
        )
        life_risk = _clip(
            life_weights["fire"] * obs["fire"]
            + life_weights["trapped_prob"] * trapped_prob
            + life_weights["time_urgency"] * obs["time_urgency"]
        )
        priority_score = _clip(
            priority_weights["trapped_prob"] * trapped_prob
            + priority_weights["life_risk"] * life_risk
            + priority_weights["time_urgency"] * obs["time_urgency"]
            + priority_weights["accessibility"] * passability_prob
        )
        assessments.append(
            {
                "zone_id": zone["zone_id"],
                "node_id": zone["node_id"],
                "trapped_prob": round(trapped_prob, 6),
                "passability_prob": round(passability_prob, 6),
                "life_risk": round(life_risk, 6),
                "priority_score": round(priority_score, 6),
            }
        )

    return sorted(assessments, key=lambda item: item["priority_score"], reverse=True)

