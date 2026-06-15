# Agent Handoff

Last updated: 2026-06-15

## Objective

Implement `计划.md`: a fixed-weight and learned-weight emergency rescue algorithm pipeline with a shared JSON interface.

## Current Status

- Phase 1 complete: project scaffold, input validation, and fixed inference.
- Phase 2 complete: risk-aware A*, expected utility matrix, and constrained global assignment.
- Phase 3 complete: four supported events, dynamic replanning, and unified output.
- Phase 4 complete: deterministic synthetic data and supervised probability-weight learning.
- Phase 5 complete: CLI, JSON Schemas, standard scenario, generated artifacts, and dual-mode outputs.
- Verification: 16 tests pass; compile, CLI, JSON parsing, and independent acceptance assertions pass.
- The workspace is not a Git repository, so there is no branch or worktree state to preserve.
- The authoritative `计划.md` learns probability weights only. Utility and A* weights remain configurable but fixed.

## Module Map

- `src/emergency_commander/input_adapter.py`: runtime validation and normalization of Qwen/preset JSON.
- `src/emergency_commander/inference.py`: trapped probability, passability, life risk, and priority.
- `src/emergency_commander/routing.py`: risk-aware A* with blocked-road and vehicle fire constraints.
- `src/emergency_commander/allocation.py`: utility candidates and small-scale global assignment enumeration.
- `src/emergency_commander/replanning.py`: event application for collapse, drone updates, SOS, and fire spread.
- `src/emergency_commander/pipeline.py`: shared fixed/learned decision output and replan history.
- `src/emergency_commander/training.py`: synthetic data, supervised fitting, Brier metrics, and comparisons.
- `src/emergency_commander/cli.py`: generate, train, run, and compare commands.

## Deliverables

- Standard input: `examples/scenario_input.json`
- Fixed output: `examples/decision_output_fixed.json` and root `decision_output.json`
- Learned output: `examples/decision_output_learned.json`
- Input/output contracts: `schemas/*.schema.json`
- Training data: `data/synthetic_training.json` (300 samples)
- Showcase data: `data/showcase_cases.json` (6 reviewed cases)
- Learned parameters: `artifacts/learned_weights.json`
- Validation metrics: `artifacts/training_metrics.json`
- Case comparison: `artifacts/model_comparison.json`

## Verified Results

- Validation mean Brier: fixed `0.22148899`, learned `0.21290494`.
- Showcase mean Brier: fixed `0.12266042`, learned `0.0741002`; 6/6 cases improve.
- Initial fixed assignment: Car-1 -> A, Car-2 -> B, Drone-1 -> C.
- Road collapse replans Car-1 from direct route to `HQ -> X -> ZONE_A`.
- Drone update makes C reachable and reassigns Car-1 to C.

## Commands

```bash
python3 -m pytest -q
python3 run.py --help
python3 run.py run --scenario examples/scenario_input.json --mode fixed --output decision_output.json
```

## Next Step

The plan is complete. A future agent can integrate `run_pipeline()` into Streamlit/Qwen adapters without changing the JSON contract. If the accepted scope changes to learn utility or A* coefficients, add separate training artifacts rather than changing the existing probability-weight file.
