from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from emergency_commander.pipeline import run_pipeline
from emergency_commander.training import (
    DEFAULT_PROBABILITY_WEIGHTS,
    apply_probability_weights,
    brier_scores,
    compare_weight_sets,
    generate_synthetic_dataset,
    train_probability_weights,
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Emergency Commander algorithm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="generate semi-synthetic labeled data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--samples", type=int, default=200)
    generate.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="fit learnable probability weights")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--metrics", required=True)
    train.add_argument("--validation-ratio", type=float, default=0.2)

    run = subparsers.add_parser("run", help="run the complete decision pipeline")
    run.add_argument("--scenario", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--mode", choices=("fixed", "learned"), required=True)
    run.add_argument("--weights", help="learned_weights.json, required in learned mode")
    run.add_argument("--no-events", action="store_true", help="skip dynamic event processing")

    compare = subparsers.add_parser("compare", help="compare fixed and learned weights per case")
    compare.add_argument("--data", required=True)
    compare.add_argument("--weights", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--limit", type=int)
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    dataset = generate_synthetic_dataset(sample_count=args.samples, seed=args.seed)
    _write_json(args.output, dataset)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    if not 0.05 <= args.validation_ratio <= 0.5:
        raise ValueError("validation-ratio must be between 0.05 and 0.5")
    dataset = _read_json(args.data)
    samples = dataset["samples"]
    split_index = int(len(samples) * (1.0 - args.validation_ratio))
    train_samples = samples[:split_index]
    validation_samples = samples[split_index:]
    learned_weights = train_probability_weights(train_samples)
    metrics = {
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "fixed_validation_brier": brier_scores(validation_samples, DEFAULT_PROBABILITY_WEIGHTS),
        "learned_validation_brier": brier_scores(validation_samples, learned_weights),
    }
    _write_json(args.output, learned_weights)
    _write_json(args.metrics, metrics)
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    scenario = _read_json(args.scenario)
    if args.mode == "learned":
        if not args.weights:
            raise ValueError("--weights is required when --mode learned")
        scenario = apply_probability_weights(scenario, _read_json(args.weights))
    else:
        scenario["mode"] = "fixed"
        scenario["run_mode"] = "fixed"
    output = run_pipeline(scenario, process_events=not args.no_events)
    _write_json(args.output, output)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    samples = _read_json(args.data)["samples"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        samples = samples[: args.limit]
    learned_weights = _read_json(args.weights)
    comparison = compare_weight_sets(samples, DEFAULT_PROBABILITY_WEIGHTS, learned_weights)
    _write_json(args.output, comparison)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate-data":
        return _run_generate(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "run":
        return _run_pipeline(args)
    if args.command == "compare":
        return _run_compare(args)
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
