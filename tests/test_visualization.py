import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from emergency_commander.visualization import (
    build_calibration_figure,
    build_map_figure,
    build_metrics_frame,
    build_probability_frame,
    build_utility_contribution_figure,
    build_utility_frame,
)
from emergency_commander.pipeline import run_pipeline
from emergency_commander.random_scenario import generate_random_scenario


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_visualization_builders_expose_map_metrics_and_calibration_layers():
    scenario = load_json("examples/scenario_input.json")
    output = run_pipeline(scenario)
    metrics = load_json("artifacts/full_bayesian_experiment/experiment_metrics.json")
    snapshot = output["timeline"][1]

    map_figure = build_map_figure(snapshot["scenario_state"], snapshot)
    probability_frame = build_probability_frame(snapshot["plan"])
    metrics_frame = build_metrics_frame(metrics)
    calibration = build_calibration_figure(metrics, "trapped_people")
    initial_matrix = output["timeline"][0]["plan"]["utility_matrix"]
    utility_frame = build_utility_frame(initial_matrix)
    feasible = next(item for item in initial_matrix if item["feasible"])
    utility_figure = build_utility_contribution_figure(feasible)

    trace_names = {trace.name for trace in map_figure.data}
    assert {"Air corridors", "Units", "Disaster zones", "Ground · Blocked"} <= trace_names
    assert any(name.startswith("Ground · ") for name in trace_names)
    assert set(probability_frame["区域"]) == {"A", "B", "C"}
    assert {"Expert CPT", "Learned CPT"} == set(metrics_frame["模型"])
    assert {trace.name for trace in calibration.data} >= {"Perfect calibration", "Expert CPT", "Learned CPT"}
    assert {"单位", "区域", "可行", "总效用", "资源成本", "原因"} <= set(utility_frame.columns)
    assert {trace.name for trace in utility_figure.data} == {"效用贡献"}


def test_map_renders_complex_scenario_before_inference_and_highlights_focus():
    scenario = generate_random_scenario(20260616)
    snapshot = {
        "plan": {
            "zone_assessment": [],
            "assignments": [],
            "routes": [],
            "utility_matrix": [],
        },
        "unit_states": {},
        "scenario_state": scenario,
    }
    focus = {
        "roads": [scenario["roads"][0]["road_id"]],
        "zones": ["A"],
        "units": ["RescueCar-1"],
    }

    figure = build_map_figure(scenario, snapshot, focus=focus)

    trace_names = [trace.name for trace in figure.data]
    assert "Disaster zones" in trace_names
    assert "Units" in trace_names
    assert "Calculation focus · roads" in trace_names
    assert "Calculation focus · zones" in trace_names
    assert "Calculation focus · units" in trace_names
    disaster_trace = next(trace for trace in figure.data if trace.name == "Disaster zones")
    assert len(disaster_trace.x) == 6
    unit_trace = next(trace for trace in figure.data if trace.name == "Units")
    assert len(unit_trace.x) == 5


def test_streamlit_demo_starts_without_runtime_exception():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("AI Emergency Commander" in title.value for title in app.title)
    labels = {button.label for button in app.button}
    assert "生成复杂地图" in labels
    assert {"道路坍塌", "火势蔓延", "新增求救", "无人机情报"} <= labels
    assert any("等待生成复杂救援地图" in item.value for item in app.markdown)


def test_generation_creates_a_manual_session_that_advances_one_phase_per_click():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    next(
        button for button in app.button if button.label == "生成复杂地图"
    ).click().run()

    assert not app.exception
    session = app.session_state["live_simulation"]
    assert session["status"] == "running"
    assert session["seed"]
    assert session["scenario"]["nodes"]
    assert len(session["scenario"]["zones"]) == 6
    assert session["phase"] == "validate"
    assert session["calculation_history"] == []

    next(
        button for button in app.button if button.label == "执行下一算法步骤"
    ).click().run()

    session = app.session_state["live_simulation"]
    assert session["phase"] == "infer"
    assert len(session["calculation_history"]) == 1
    assert session["calculation_history"][0]["phase"] == "validate"
