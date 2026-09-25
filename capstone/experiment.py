"""Experiment orchestration, checkpointing, and result serialization."""

import datetime as dt
import importlib.metadata
import json
import platform
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import validate_args
from .models import BaseModel, build_model
from .statistics import (
    summarize_mitigation_effect,
    summarize_primary_endpoints,
    summarize_results,
)
from .tasks import load_examples, make_messages, score_prediction, serialized_reference
from .typos import compute_idf, introduce_typos, word_spans


@dataclass(frozen=True)
class Condition:
    typo_type: str
    location: str
    severity: int
    difficulty: str
    mitigation: str
    replicate: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def installed_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def evaluate_one(
    model: BaseModel,
    example: Dict,
    question: str,
    mitigation: str,
    max_new_tokens: int,
) -> Dict:
    """Generate and score one response without terminating a long run on error."""
    started = time.perf_counter()
    messages = make_messages(example, question, mitigation)
    try:
        prediction = model.generate(messages, max_new_tokens=max_new_tokens)
    except Exception as error:
        return {
            "prediction": "",
            "accuracy": np.nan,
            "em": np.nan,
            "f1": np.nan,
            "semantic_correct": np.nan,
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }

    return {
        "prediction": prediction,
        **score_prediction(example, prediction),
        "latency_seconds": time.perf_counter() - started,
        "error": None,
    }


def base_row(args, example: Dict) -> Dict:
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "task": args.task,
        "example_id": example["id"],
        "keyword_strategy": args.keyword_strategy,
        "clean_question": example["question"],
        "context": example.get("context"),
        "reference": serialized_reference(example),
        "question_word_count": len(word_spans(example["question"])),
    }


def corruption_seed(args, example_index: int, condition: Condition) -> int:
    """Stable across severity, difficulty, and mitigation for paired conditions."""
    typo_component = (
        0 if args.match_typo_targets
        else sum(ord(char) for char in condition.typo_type) * 101
    )
    return (
        args.seed * 1_000_003
        + example_index * 10_007
        + typo_component
        + sum(ord(char) for char in condition.location) * 17
        + condition.replicate * 1_000_000_007
    )


def experiment_conditions(args):
    return [
        Condition(typo, location, severity, difficulty, mitigation, replicate)
        for typo in args.typo_types
        for location in args.locations
        for severity in args.severities
        for difficulty in args.typo_difficulties
        for mitigation in args.mitigations
        for replicate in range(args.corruption_replicates)
    ]


def write_checkpoint(rows, output_path: str) -> None:
    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_metadata(
    args, examples, frame: pd.DataFrame, output_path: str, model: BaseModel
) -> str:
    metadata_path = output_path.rsplit(".", 1)[0] + "_metadata.json"
    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arguments": vars(args),
        "n_loaded_examples": len(examples),
        "n_rows": len(frame),
        "n_generation_errors": int(frame["error"].notna().sum()),
        "severity_definition": "number of distinct corrupted words",
        "nested_severity": True,
        "matched_typo_targets": args.match_typo_targets,
        "corruption_replicates": args.corruption_replicates,
        "independent_unit_for_inference": "example_id",
        "decoding": {
            "strategy": (
                "greedy" if args.backend == "hf" else "hosted API response"
            ),
            "do_sample": False if args.backend == "hf" else None,
            "max_new_tokens": args.max_new_tokens,
            "openai_temperature": args.openai_temperature,
        },
        "location_fallback_allowed": False,
        "keyword_definition": (
            "highest smoothed corpus IDF among non-stopwords, length tie-break"
            if args.keyword_strategy == "idf"
            else "longest non-stopword"
        ),
        "environment": {
            "python": platform.python_version(),
            "packages": installed_versions([
                "torch", "transformers", "datasets", "accelerate",
                "bitsandbytes", "numpy", "pandas", "scipy", "openai",
                "tiktoken",
            ]),
        },
        "model_runtime": model.metadata(),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def run_experiment(args) -> None:
    validate_args(args)
    seed_everything(args.seed)
    examples = load_examples(
        args.task,
        args.n_samples,
        args.seed,
        revision=args.dataset_revision,
        sample_offset=args.sample_offset,
    )
    keyword_scores = (
        compute_idf([example["question"] for example in examples])
        if args.keyword_strategy == "idf" else None
    )
    conditions = experiment_conditions(args)
    total_generations = len(examples) * (len(args.mitigations) + len(conditions))
    print(
        f"Design: {len(conditions)} corrupted conditions + "
        f"{len(args.mitigations)} matched clean condition(s); "
        f"{total_generations:,} total generations."
    )

    model = build_model(
        args.backend,
        args.model,
        args.load_in_4bit,
        revision=args.model_revision,
        openai_temperature=args.openai_temperature,
    )
    rows = []

    for mitigation in args.mitigations:
        print(
            f"Running clean baseline | mitigation={mitigation} on "
            f"{len(examples)} {args.task} examples..."
        )
        for example in tqdm(examples):
            result = evaluate_one(
                model,
                example,
                example["question"],
                mitigation,
                args.max_new_tokens,
            )
            rows.append({
                **base_row(args, example),
                "condition": "clean",
                "typo_type": "none",
                "location": "none",
                "severity": 0,
                "actual_severity": 0,
                "severity_achieved": True,
                "typo_difficulty": "none",
                "mitigation": mitigation,
                "input_question": example["question"],
                "corruption_seed": np.nan,
                "edits": "[]",
                "corruption_replicate": np.nan,
                "actual_char_operations": 0,
                **result,
            })
        write_checkpoint(rows, args.output)

    for condition in conditions:
        print(
            f"Running {condition.typo_type} | {condition.location} | "
            f"severity={condition.severity} | difficulty={condition.difficulty} | "
            f"mitigation={condition.mitigation} | "
            f"replicate={condition.replicate + 1}/{args.corruption_replicates}"
        )
        for index, example in enumerate(tqdm(examples, leave=False)):
            seed = corruption_seed(args, index, condition)
            corrupted, edits = introduce_typos(
                example["question"],
                typo_type=condition.typo_type,
                location=condition.location,
                severity=condition.severity,
                seed=seed,
                difficulty=condition.difficulty,
                keyword_scores=keyword_scores,
                target_typo_types=(
                    args.typo_types if args.match_typo_targets else None
                ),
                required_operations=(
                    2 if "hard" in args.typo_difficulties else 1
                ),
            )
            result = evaluate_one(
                model,
                example,
                corrupted,
                condition.mitigation,
                args.max_new_tokens,
            )
            rows.append({
                **base_row(args, example),
                "condition": "corrupted",
                "typo_type": condition.typo_type,
                "location": condition.location,
                "severity": condition.severity,
                "actual_severity": len(edits),
                "severity_achieved": len(edits) == condition.severity,
                "typo_difficulty": condition.difficulty,
                "mitigation": condition.mitigation,
                "input_question": corrupted,
                "corruption_seed": seed,
                "corruption_replicate": condition.replicate,
                "actual_char_operations": sum(
                    int(edit.get("operation_count", 0)) for edit in edits
                ),
                "edits": json.dumps(edits),
                **result,
            })
            if args.checkpoint_every and len(rows) % args.checkpoint_every == 0:
                write_checkpoint(rows, args.output)
        write_checkpoint(rows, args.output)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    summary = summarize_results(
        frame, n_boot=args.bootstrap_samples, seed=args.seed
    )
    summary_path = args.output.rsplit(".", 1)[0] + "_summary.csv"
    summary.to_csv(summary_path, index=False)

    primary_summary = summarize_primary_endpoints(
        frame, n_boot=args.bootstrap_samples, seed=args.seed
    )
    primary_path = args.output.rsplit(".", 1)[0] + "_primary_summary.csv"
    primary_summary.to_csv(primary_path, index=False)

    mitigation_path = None
    if {"none", "self_correct"}.issubset(set(args.mitigations)):
        mitigation = summarize_mitigation_effect(
            frame, n_boot=args.bootstrap_samples, seed=args.seed
        )
        mitigation_path = args.output.rsplit(".", 1)[0] + "_mitigation_summary.csv"
        mitigation.to_csv(mitigation_path, index=False)

    metadata_path = write_metadata(args, examples, frame, args.output, model)
    print("\n=== PRIMARY SUMMARY ===")
    print(primary_summary.to_string(index=False))
    print(
        "\nThe full condition-level summary was written to disk; use it for "
        "secondary typo-type/location analyses."
    )
    print(f"\nRaw results: {args.output}")
    print(f"Summary:     {summary_path}")
    print(f"Primary:     {primary_path}")
    if mitigation_path:
        print(f"Mitigation:  {mitigation_path}")
    print(f"Metadata:    {metadata_path}")
