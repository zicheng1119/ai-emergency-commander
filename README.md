# AI Emergency Commander Algorithms

This repository implements the algorithm owner's fixed and learned variants from `计划.md`.
Both modes share the same explainable pipeline:

`JSON input -> probability inference -> life risk/priority -> expected utility -> constrained assignment -> risk-aware A* -> dynamic replanning`

Only the probability-layer weights are learned. Utility, assignment constraints, and A* structure remain fixed and inspectable.

## Quick Start

```bash
python3 -m pytest -q

python3 run.py generate-data \
  --output data/synthetic_training.json --samples 300 --seed 42

python3 run.py train \
  --data data/synthetic_training.json \
  --output artifacts/learned_weights.json \
  --metrics artifacts/training_metrics.json

python3 run.py run \
  --scenario examples/scenario_input.json \
  --mode fixed \
  --output examples/decision_output_fixed.json

python3 run.py run \
  --scenario examples/scenario_input.json \
  --mode learned \
  --weights artifacts/learned_weights.json \
  --output examples/decision_output_learned.json

python3 run.py compare \
  --data data/showcase_cases.json \
  --weights artifacts/learned_weights.json \
  --output artifacts/model_comparison.json
```

The package uses a `src` layout. `run.py` makes the CLI available without installation. For an editable installation and the `emergency-commander` console command, use `python3 -m pip install -e .`.

## JSON Contract

- Input example: `examples/scenario_input.json`
- Input schema: `schemas/scenario.schema.json`
- Output schema: `schemas/decision_output.schema.json`
- `mode` / `run_mode`: `fixed` or `learned`
- All observation and risk values are normalized to `[0, 1]`.
- Event `changes` use dot paths such as `risk.damage` or `observations.sos_signal`.
- Supported events: `road_collapse`, `drone_update`, `new_sos`, `fire_spread`.

## Integration Boundary

Qwen or the frontend supplies `scenario_input.json`. The algorithm service returns the same output structure in both modes. The frontend and report generator should consume only:

- `zone_assessment`
- `assignments`
- `routes`
- `replan_log`
- `weights_used`

See `HANDOFF.md` for current status and continuation notes.
