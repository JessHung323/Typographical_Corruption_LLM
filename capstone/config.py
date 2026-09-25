"""Command-line configuration for the experiment runner."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM robustness to controlled typographical errors."
    )
    parser.add_argument("--task", choices=["squad", "gsm8k"], required=True)
    parser.add_argument("--backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_revision")
    parser.add_argument(
        "--dataset_revision",
        help="optional Hugging Face dataset git revision/commit for reproducibility",
    )
    parser.add_argument("--openai_temperature", type=float)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="skip this many examples after deterministic shuffling",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--typo_types",
        nargs="+",
        default=["keyboard", "deletion", "transposition"],
        choices=["keyboard", "deletion", "transposition"],
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["beginning", "middle", "end", "keyword"],
        choices=["beginning", "middle", "end", "keyword", "any"],
    )
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--match_typo_targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "select words eligible for every requested typo type so typo-type "
            "comparisons use the same targets (default: enabled)"
        ),
    )
    parser.add_argument(
        "--corruption_replicates",
        type=int,
        default=1,
        help=(
            "independent corruption realizations per example and condition; "
            "replicates are averaged within example for inference"
        ),
    )
    parser.add_argument(
        "--typo_difficulties",
        "--typo_difficulty",
        dest="typo_difficulties",
        nargs="+",
        choices=["standard", "hard"],
        default=["hard"],
        help=(
            "standard applies one operation per word; hard targets words that "
            "support two non-conflicting operations"
        ),
    )
    parser.add_argument(
        "--keyword_strategy", choices=["idf", "longest"], default="idf"
    )
    parser.add_argument(
        "--mitigations",
        nargs="+",
        default=["none"],
        choices=["none", "self_correct"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if any(severity < 1 for severity in args.severities):
        raise ValueError("All --severities values must be at least 1")
    if args.n_samples < 1:
        raise ValueError("--n_samples must be at least 1")
    if args.sample_offset < 0:
        raise ValueError("--sample_offset cannot be negative")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100")
    if args.corruption_replicates < 1:
        raise ValueError("--corruption_replicates must be at least 1")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint_every cannot be negative")
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Choose a new name or pass --overwrite."
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
