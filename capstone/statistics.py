"""Paired statistical summaries for robustness and mitigation effects."""

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def paired_bootstrap_difference(
    first_scores: Sequence[float],
    second_scores: Sequence[float],
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap the paired mean ``first_scores - second_scores``."""
    first = np.asarray(first_scores, dtype=float)
    second = np.asarray(second_scores, dtype=float)
    if len(first) != len(second):
        raise ValueError("Paired score arrays must have equal length")
    if not len(first):
        raise ValueError("Paired bootstrap requires at least one observation")

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot)
    for iteration in range(n_boot):
        indices = rng.integers(0, len(first), size=len(first))
        differences[iteration] = np.mean(first[indices] - second[indices])

    return {
        "mean_difference": float(np.mean(first - second)),
        "ci_2.5": float(np.percentile(differences, 2.5)),
        "ci_97.5": float(np.percentile(differences, 97.5)),
        "bootstrap_probability_le_zero": float(np.mean(differences <= 0)),
    }


def summarize_results(
    frame: pd.DataFrame, n_boot: int = 5000, seed: int = 42
) -> pd.DataFrame:
    metric = "f1" if frame["task"].iloc[0] == "squad" else "accuracy"
    group_columns = [
        "model", "task", "typo_type", "location", "severity",
        "typo_difficulty", "keyword_strategy", "mitigation",
    ]
    clean = frame[frame["condition"] == "clean"][
        ["example_id", "mitigation", metric]
    ].rename(columns={metric: "clean_example_score"})

    rows = []
    for keys, group in frame[frame["condition"] == "corrupted"].groupby(
        group_columns, dropna=False
    ):
        row = dict(zip(group_columns, keys))
        paired = group.merge(clean, on=["example_id", "mitigation"], how="inner")
        # Corruption replicates reduce Monte Carlo noise, but examples remain the
        # independent unit. Average replicates before bootstrapping.
        paired = (
            paired.groupby("example_id", as_index=False)
            .agg(
                corrupted_example_score=(metric, "mean"),
                clean_example_score=("clean_example_score", "first"),
                usable_replicates=(metric, "count"),
                requested_replicates=(metric, "size"),
            )
        )
        paired = paired[
            paired["usable_replicates"] == paired["requested_replicates"]
        ].dropna(subset=["corrupted_example_score", "clean_example_score"])
        scores = paired["corrupted_example_score"].to_numpy(dtype=float)
        clean_scores = paired["clean_example_score"].to_numpy(dtype=float)
        row.update({
            "score": float(np.mean(scores)) if len(scores) else np.nan,
            "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
            "n": int(len(scores)),
            "requested_n": int(group["example_id"].nunique()),
            "requested_rows": int(len(group)),
            "excluded_examples_due_to_errors": int(
                group["example_id"].nunique() - len(paired)
            ),
            "corruption_replicates": int(
                group["corruption_replicate"].nunique()
                if "corruption_replicate" in group else 1
            ),
            "error_rate": float(group["error"].notna().mean()),
            "mean_actual_severity": float(group["actual_severity"].mean()),
            "mean_actual_char_operations": float(
                group["actual_char_operations"].mean()
                if "actual_char_operations" in group
                else np.nan
            ),
            "severity_completion_rate": float(group["severity_achieved"].mean()),
            "clean_score": float(np.mean(clean_scores)) if len(clean_scores) else np.nan,
        })
        if len(scores):
            bootstrap = paired_bootstrap_difference(
                clean_scores, scores, n_boot=n_boot, seed=seed
            )
            row.update({
                "absolute_drop": bootstrap["mean_difference"],
                "drop_ci_2.5": bootstrap["ci_2.5"],
                "drop_ci_97.5": bootstrap["ci_97.5"],
                "bootstrap_probability_drop_le_zero": bootstrap[
                    "bootstrap_probability_le_zero"
                ],
                "relative_degradation": (
                    bootstrap["mean_difference"] / row["clean_score"]
                    if row["clean_score"] != 0 else np.nan
                ),
            })
        else:
            row.update({
                "absolute_drop": np.nan,
                "drop_ci_2.5": np.nan,
                "drop_ci_97.5": np.nan,
                "bootstrap_probability_drop_le_zero": np.nan,
                "relative_degradation": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_primary_endpoints(
    frame: pd.DataFrame, n_boot: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """Pre-specified severity effects balanced across typo types and locations.

    Each example contributes one value per severity/difficulty/mitigation after
    averaging its corruption replicates and factorial conditions. This compact
    table should be used for the main claims; the full condition summary is
    secondary/exploratory.
    """
    metric = "f1" if frame["task"].iloc[0] == "squad" else "accuracy"
    clean = (
        frame[frame["condition"] == "clean"]
        .groupby(["example_id", "mitigation"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "clean_score"})
    )
    corrupt = frame[frame["condition"] == "corrupted"].copy()
    group_columns = [
        "model", "task", "severity", "typo_difficulty", "mitigation"
    ]
    rows = []
    for group_number, (keys, group) in enumerate(
        corrupt.groupby(group_columns, dropna=False)
    ):
        row = dict(zip(group_columns, keys))
        balanced = (
            group.groupby("example_id", as_index=False)
            .agg(
                corrupted_score=(metric, "mean"),
                usable_condition_rows=(metric, "count"),
                requested_condition_rows=(metric, "size"),
            )
            .merge(
                clean[clean["mitigation"] == row["mitigation"]][
                    ["example_id", "clean_score"]
                ],
                on="example_id",
                how="inner",
            )
        )
        balanced = balanced[
            balanced["usable_condition_rows"]
            == balanced["requested_condition_rows"]
        ].dropna(subset=["corrupted_score", "clean_score"])
        requested_examples = int(group["example_id"].nunique())
        if balanced.empty:
            row.update({
                "n": 0, "clean_score": np.nan, "score": np.nan,
                "absolute_drop": np.nan, "drop_ci_2.5": np.nan,
                "drop_ci_97.5": np.nan,
                "bootstrap_probability_drop_le_zero": np.nan,
                "requested_n": requested_examples,
                "excluded_examples_due_to_errors": requested_examples,
            })
        else:
            bootstrap = paired_bootstrap_difference(
                balanced["clean_score"],
                balanced["corrupted_score"],
                n_boot=n_boot,
                seed=seed + group_number,
            )
            row.update({
                "n": int(len(balanced)),
                "requested_n": requested_examples,
                "excluded_examples_due_to_errors": int(
                    requested_examples - len(balanced)
                ),
                "clean_score": float(balanced["clean_score"].mean()),
                "score": float(balanced["corrupted_score"].mean()),
                "absolute_drop": bootstrap["mean_difference"],
                "drop_ci_2.5": bootstrap["ci_2.5"],
                "drop_ci_97.5": bootstrap["ci_97.5"],
                "bootstrap_probability_drop_le_zero": bootstrap[
                    "bootstrap_probability_le_zero"
                ],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mitigation_effect(
    frame: pd.DataFrame, n_boot: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """Compare self-correction and no mitigation on identical corruptions."""
    metric = "f1" if frame["task"].iloc[0] == "squad" else "accuracy"
    group_columns = [
        "model", "task", "typo_type", "location", "severity",
        "typo_difficulty", "keyword_strategy",
    ]
    rows = []
    corrupt = frame[frame["condition"] == "corrupted"]
    for keys, group in corrupt.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        index = ["example_id"]
        if "corruption_replicate" in group:
            index.append("corruption_replicate")
        pivot = group.pivot_table(
            index=index, columns="mitigation", values=metric, aggfunc="mean"
        )
        if not {"none", "self_correct"}.issubset(pivot.columns):
            row["n"] = 0
            rows.append(row)
            continue
        pivot = pivot.dropna(subset=["none", "self_correct"])
        if "corruption_replicate" in index and len(pivot):
            pivot = pivot.groupby(level="example_id")[["none", "self_correct"]].mean()
        if len(pivot):
            bootstrap = paired_bootstrap_difference(
                pivot["self_correct"], pivot["none"], n_boot=n_boot, seed=seed
            )
            row.update({
                "n": int(len(pivot)),
                "unmitigated_score": float(pivot["none"].mean()),
                "self_correct_score": float(pivot["self_correct"].mean()),
                "mean_improvement": bootstrap["mean_difference"],
                "improvement_ci_2.5": bootstrap["ci_2.5"],
                "improvement_ci_97.5": bootstrap["ci_97.5"],
                "bootstrap_probability_improvement_le_zero": bootstrap[
                    "bootstrap_probability_le_zero"
                ],
            })
        rows.append(row)
    return pd.DataFrame(rows)
