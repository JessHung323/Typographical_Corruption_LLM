"""Paired, example-level comparisons of typo robustness across model runs."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


TYPO_ORDER = ["keyboard", "deletion", "transposition"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
CONDITION_KEYS = ["example_id", "typo_type", "location", "severity"]


def primary_metric(df: pd.DataFrame) -> str:
    if "f1" in df and df["f1"].notna().any():
        return "f1"
    if "accuracy" in df and df["accuracy"].notna().any():
        return "accuracy"
    raise ValueError("No usable f1 or accuracy column")


def mean_ci(values: Iterable[float], n_boot: int, seed: int) -> dict:
    """Return a percentile bootstrap CI for a mean over independent examples."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": np.nan, "ci_2.5": np.nan, "ci_97.5": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    boot = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_2.5": float(np.percentile(boot, 2.5)),
        "ci_97.5": float(np.percentile(boot, 97.5)),
        "n": int(len(array)),
    }


def load_runs(
    paths: List[str], mitigation: str, difficulty: str
) -> tuple[pd.DataFrame, str]:
    """Load runs and retain only the example IDs shared by every input run."""
    frames = []
    metric = None
    example_sets = {}
    used_labels = set()
    for run_number, path_string in enumerate(paths, start=1):
        path = Path(path_string)
        frame = pd.read_csv(path)
        run_metric = primary_metric(frame)
        if metric is not None and run_metric != metric:
            raise ValueError("All inputs must use the same task metric")
        metric = run_metric
        frame = frame[frame["mitigation"] == mitigation].copy()
        frame = frame[
            (frame["condition"] == "clean")
            | (frame["typo_difficulty"] == difficulty)
        ]
        if frame.empty:
            raise ValueError(f"No matching rows in {path}")
        model = str(frame["model"].dropna().iloc[0])
        label = model.split("/")[-1]
        if label in used_labels:
            label = f"{label} ({run_number})"
        used_labels.add(label)
        run_id = f"run_{run_number}"
        example_sets[run_id] = set(
            frame[frame["condition"] == "clean"]["example_id"].astype(str)
        )
        frame["example_id"] = frame["example_id"].astype(str)
        frame["source_file"] = path.name
        frame["run_id"] = run_id
        frame["model_label"] = label
        frames.append(frame)

    tasks = {str(frame["task"].iloc[0]) for frame in frames}
    if len(tasks) != 1:
        raise ValueError("All inputs must use the same task")
    common = set.intersection(*example_sets.values())
    if not common:
        raise ValueError("The model runs have no shared example IDs")
    for run_id, ids in example_sets.items():
        if ids != common:
            print(
                f"Warning: {run_id} has {len(ids)} clean examples; "
                f"the paired comparison uses the {len(common)} shared examples."
            )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["example_id"].isin(common)].copy()
    return combined, metric


def example_level_drops(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Create one score-drop row per example and experimental condition.

    Multiple corruption replicates are averaged within an example before any
    inferential calculation, so replicates are never treated as independent.
    """
    clean = (
        df[df["condition"] == "clean"]
        .groupby(["run_id", "model", "model_label", "task", "example_id"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "clean_score"})
    )
    corrupt = df[df["condition"] == "corrupted"].copy()
    group_keys = [
        "run_id", "model", "model_label", "task", *CONDITION_KEYS,
        "typo_difficulty", "mitigation",
    ]
    aggregations = {
        "corrupted_score": (metric, "mean"),
        "usable_replicates": (metric, "count"),
        "requested_replicates": (metric, "size"),
    }
    if "error" in corrupt:
        corrupt["generation_failed"] = corrupt["error"].notna().astype(float)
        aggregations["error_rate"] = ("generation_failed", "mean")
    if "severity_achieved" in corrupt:
        aggregations["severity_completion_rate"] = ("severity_achieved", "mean")
    if "corruption_replicate" in corrupt:
        aggregations["n_corruption_replicates"] = ("corruption_replicate", "nunique")
    else:
        corrupt["_one"] = 1
        aggregations["n_corruption_replicates"] = ("_one", "sum")
    corrupt = corrupt.groupby(group_keys, dropna=False, as_index=False).agg(**aggregations)
    corrupt = corrupt[
        corrupt["usable_replicates"] == corrupt["requested_replicates"]
    ].copy()
    paired = corrupt.merge(
        clean,
        on=["run_id", "model", "model_label", "task", "example_id"],
        how="inner",
        validate="many_to_one",
    )
    paired["absolute_drop"] = paired["clean_score"] - paired["corrupted_score"]
    return paired


def condition_summary(drops: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    groups = [
        "run_id", "model", "model_label", "task", "typo_type", "location",
        "severity", "typo_difficulty", "mitigation",
    ]
    rows = []
    for group_number, (keys, group) in enumerate(drops.groupby(groups, dropna=False)):
        row = dict(zip(groups, keys))
        stats = mean_ci(group["absolute_drop"], n_boot, seed + group_number)
        row.update({
            "score": float(group["corrupted_score"].mean()),
            "clean_score": float(group["clean_score"].mean()),
            "absolute_drop": stats["mean"],
            "drop_ci_2.5": stats["ci_2.5"],
            "drop_ci_97.5": stats["ci_97.5"],
            "n_examples": stats["n"],
            "mean_replicates_per_example": float(group["n_corruption_replicates"].mean()),
            "relative_degradation": (
                stats["mean"] / group["clean_score"].mean()
                if group["clean_score"].mean() != 0 else np.nan
            ),
        })
        if "error_rate" in group:
            row["error_rate"] = float(group["error_rate"].mean())
        if "severity_completion_rate" in group:
            row["severity_completion_rate"] = float(
                group["severity_completion_rate"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_example_effects(
    drops: pd.DataFrame, group_columns: List[str]
) -> pd.DataFrame:
    """Balance conditions by averaging them within each independent example."""
    effects = (
        drops.groupby(["run_id", "model_label", "example_id", *group_columns], as_index=False)
        .agg(
            absolute_drop=("absolute_drop", "mean"),
            observed_conditions=("absolute_drop", "size"),
        )
    )
    balance_groups = ["run_id", "model_label", *group_columns]
    effects["expected_conditions"] = effects.groupby(balance_groups)[
        "observed_conditions"
    ].transform("max")
    return effects[
        effects["observed_conditions"] == effects["expected_conditions"]
    ].copy()


def bootstrap_group_table(
    example_effects: pd.DataFrame,
    group_columns: List[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_number, (keys, group) in enumerate(
        example_effects.groupby(group_columns, dropna=False)
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(mean_ci(group["absolute_drop"], n_boot, seed + group_number))
        rows.append(row)
    return pd.DataFrame(rows)


def model_gap_tables(
    drops: pd.DataFrame, n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate paired differences in degradation for every pair of model runs."""
    run_labels = drops[["run_id", "model_label"]].drop_duplicates()
    condition_rows = []
    severity_rows = []
    for pair_number, ((run_a, label_a), (run_b, label_b)) in enumerate(
        itertools.combinations(run_labels.itertuples(index=False, name=None), 2)
    ):
        left = drops[drops["run_id"] == run_a][CONDITION_KEYS + ["absolute_drop"]]
        right = drops[drops["run_id"] == run_b][CONDITION_KEYS + ["absolute_drop"]]
        paired = left.merge(
            right,
            on=CONDITION_KEYS,
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            continue
        paired["model_gap"] = paired["absolute_drop_a"] - paired["absolute_drop_b"]
        condition_groups = ["typo_type", "location", "severity"]
        for group_number, (keys, group) in enumerate(paired.groupby(condition_groups)):
            stats = mean_ci(
                group["model_gap"], n_boot,
                seed + pair_number * 1000 + group_number,
            )
            condition_rows.append({
                "model_a": label_a,
                "model_b": label_b,
                "gap_definition": "drop_model_a_minus_drop_model_b",
                **dict(zip(condition_groups, keys)),
                "mean_gap": stats["mean"],
                "gap_ci_2.5": stats["ci_2.5"],
                "gap_ci_97.5": stats["ci_97.5"],
                "n_examples": stats["n"],
            })
        balanced = (
            paired.groupby(["example_id", "severity"], as_index=False)
            .agg(
                model_gap=("model_gap", "mean"),
                observed_conditions=("model_gap", "size"),
            )
        )
        balanced["expected_conditions"] = balanced.groupby("severity")[
            "observed_conditions"
        ].transform("max")
        balanced = balanced[
            balanced["observed_conditions"] == balanced["expected_conditions"]
        ]
        for group_number, (severity, group) in enumerate(balanced.groupby("severity")):
            stats = mean_ci(
                group["model_gap"], n_boot,
                seed + 50_000 + pair_number * 100 + group_number,
            )
            severity_rows.append({
                "model_a": label_a,
                "model_b": label_b,
                "gap_definition": "drop_model_a_minus_drop_model_b",
                "severity": severity,
                "mean_gap": stats["mean"],
                "gap_ci_2.5": stats["ci_2.5"],
                "gap_ci_97.5": stats["ci_97.5"],
                "n_examples": stats["n"],
            })
    return pd.DataFrame(condition_rows), pd.DataFrame(severity_rows)


def save_overall_curves(
    drops: pd.DataFrame, output_dir: Path, n_boot: int, seed: int
) -> pd.DataFrame:
    effects = aggregate_example_effects(drops, ["severity"])
    table = bootstrap_group_table(
        effects, ["model_label", "severity"], n_boot=n_boot, seed=seed
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    for index, (model, group) in enumerate(table.groupby("model_label", sort=False)):
        group = group.sort_values("severity")
        y = 100 * group["mean"]
        yerr = np.vstack((y - 100 * group["ci_2.5"], 100 * group["ci_97.5"] - y))
        axis.errorbar(
            group["severity"], y, yerr=yerr, marker="o", linewidth=2,
            capsize=3, color=COLORS[index % len(COLORS)], label=model,
        )
    axis.set_xlabel("Corrupted words (severity)")
    axis.set_ylabel("Mean F1/accuracy drop (percentage points)")
    axis.set_title("Paired robustness by severity (95% example-bootstrap CI)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_severity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(output_dir / "cross_model_severity_estimates.csv", index=False)
    return table


def save_typo_comparison(
    drops: pd.DataFrame, output_dir: Path, n_boot: int, seed: int
) -> pd.DataFrame:
    maximum_severity = int(drops["severity"].max())
    subset = drops[drops["severity"] == maximum_severity]
    effects = aggregate_example_effects(subset, ["typo_type"])
    table = bootstrap_group_table(
        effects, ["model_label", "typo_type"], n_boot=n_boot, seed=seed + 10_000
    )
    models = list(table["model_label"].unique())
    typo_types = [t for t in TYPO_ORDER if t in set(table["typo_type"])]
    x = np.arange(len(typo_types))
    width = 0.8 / max(1, len(models))
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, model in enumerate(models):
        model_table = (
            table[table["model_label"] == model]
            .set_index("typo_type")
            .reindex(typo_types)
        )
        values = 100 * model_table["mean"]
        errors = np.vstack((
            values - 100 * model_table["ci_2.5"],
            100 * model_table["ci_97.5"] - values,
        ))
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(
            x + offset, values, width=width, yerr=errors, capsize=3,
            color=COLORS[index % len(COLORS)], label=model,
        )
    axis.set_xticks(x, typo_types)
    axis.set_xlabel("Typo type")
    axis.set_ylabel("Mean F1/accuracy drop (percentage points)")
    axis.set_title(
        f"Paired degradation at severity {maximum_severity} "
        "(95% example-bootstrap CI)"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_model_typo_types.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(output_dir / "cross_model_typo_type_estimates.csv", index=False)
    return table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--mitigation", choices=["none", "self_correct"], default="none")
    parser.add_argument("--typo_difficulty", choices=["standard", "hard"], default="hard")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="cross_model_figures")
    return parser.parse_args()


def main(args) -> None:
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap_samples must be at least 100")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, metric = load_runs(args.inputs, args.mitigation, args.typo_difficulty)
    drops = example_level_drops(combined, metric)
    summary = condition_summary(drops, args.bootstrap_samples, args.seed)
    summary.to_csv(output_dir / "cross_model_summary.csv", index=False)
    save_overall_curves(drops, output_dir, args.bootstrap_samples, args.seed)
    save_typo_comparison(drops, output_dir, args.bootstrap_samples, args.seed)
    gap_condition, gap_severity = model_gap_tables(
        drops, args.bootstrap_samples, args.seed
    )
    gap_condition.to_csv(output_dir / "cross_model_gap_by_condition.csv", index=False)
    gap_severity.to_csv(output_dir / "cross_model_gap_by_severity.csv", index=False)
    print(
        f"Saved paired cross-model comparison to {output_dir}. "
        "Positive model gaps mean model A degraded more than model B."
    )


if __name__ == "__main__":
    main(parse_args())
