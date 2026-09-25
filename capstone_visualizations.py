"""Visual and representation analysis for typo-robustness experiment outputs.

This script reads the raw CSV written by ``capstone_robustness.py`` and creates:

1. F1/accuracy degradation heatmaps by typo type, location, and severity.
2. Severity-response curves.
3. Tokenization-fragmentation plots.
4. Layer-wise clean/corrupted contextual-representation similarity for local
   Hugging Face models.
5. External embedding-space drift for OpenAI API experiment outputs.
6. A 2D PCA/UMAP projection with clean -> severity trajectories.

Example
-------
python capstone_visualizations.py \
    --input qwen_squad_nested.csv \
    --model Qwen/Qwen3-4B \
    --embedding_typo keyboard \
    --embedding_location beginning \
    --embedding_samples 40 \
    --output_dir qwen_squad_figures
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from capstone.tasks import make_messages


TYPO_ORDER = ["keyboard", "deletion", "transposition"]
LOCATION_ORDER = ["beginning", "middle", "end", "keyword", "any"]
COLORS = {
    "keyboard": "#D55E00",
    "deletion": "#0072B2",
    "transposition": "#009E73",
}


def ordered_present(preferred: Sequence[str], values: Sequence[str]) -> List[str]:
    present = list(dict.fromkeys(str(value) for value in values))
    return [value for value in preferred if value in present] + [
        value for value in present if value not in preferred
    ]


def metric_for(df: pd.DataFrame) -> str:
    if "f1" in df and df["f1"].notna().any():
        return "f1"
    if "accuracy" in df and df["accuracy"].notna().any():
        return "accuracy"
    raise ValueError("The input CSV has neither usable 'f1' nor 'accuracy' values")


def condition_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    clean = df[df["condition"] == "clean"].groupby("example_id")[metric].mean()
    corrupt = df[df["condition"] == "corrupted"].copy()
    corrupt["clean_example_score"] = corrupt["example_id"].map(clean)

    columns = ["typo_type", "location", "severity"]
    if "typo_difficulty" in corrupt.columns:
        columns.append("typo_difficulty")
    # Average corruption replicates within each example. Error bars must reflect
    # independent examples, not repeated perturbations of an example.
    corrupt = (
        corrupt.groupby(["example_id", *columns], dropna=False, as_index=False)
        .agg(
            score=(metric, "mean"),
            usable_replicates=(metric, "count"),
            requested_replicates=(metric, "size"),
            clean_example_score=("clean_example_score", "first"),
            actual_severity=("actual_severity", "mean"),
            severity_achieved=("severity_achieved", "mean"),
        )
    )
    corrupt.loc[
        corrupt["usable_replicates"] != corrupt["requested_replicates"], "score"
    ] = np.nan
    corrupt["example_drop"] = corrupt["clean_example_score"] - corrupt["score"]
    return (
        corrupt.groupby(columns, dropna=False)
        .agg(
            score=("score", "mean"),
            score_sem=("score", "sem"),
            absolute_drop=("example_drop", "mean"),
            drop_sem=("example_drop", "sem"),
            score_std=("score", "std"),
            n=("score", "count"),
            requested_n=("example_id", "nunique"),
            mean_actual_severity=("actual_severity", "mean"),
            severity_completion_rate=("severity_achieved", "mean"),
        )
        .reset_index()
    )


def save_performance_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    severities = sorted(summary["severity"].astype(int).unique())
    typo_types = ordered_present(TYPO_ORDER, summary["typo_type"].unique())
    locations = ordered_present(LOCATION_ORDER, summary["location"].unique())
    vmax = max(float(summary["absolute_drop"].max()), 1e-9)
    fig, axes = plt.subplots(
        1, len(severities), figsize=(5.0 * len(severities), 4.3), squeeze=False
    )

    image = None
    for axis, severity in zip(axes[0], severities):
        subset = summary[summary["severity"] == severity]
        pivot = subset.pivot_table(
            index="typo_type", columns="location", values="absolute_drop"
        ).reindex(index=typo_types, columns=locations)
        image = axis.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=vmax)
        axis.set_title(f"Severity {severity}")
        axis.set_xticks(range(len(locations)), locations, rotation=35, ha="right")
        axis.set_yticks(range(len(typo_types)), typo_types)
        axis.set_xlabel("Typo location")
        axis.set_ylabel("Typo type")
        for row in range(len(typo_types)):
            for col in range(len(locations)):
                value = pivot.iloc[row, col]
                label = "–" if pd.isna(value) else f"{value:.3f}"
                axis.text(col, row, label, ha="center", va="center", fontsize=9)

    fig.suptitle("Performance loss from clean baseline", y=1.02)
    if image is not None:
        cbar = fig.colorbar(image, ax=list(axes[0]), shrink=0.82, pad=0.02)
        cbar.set_label("Mean absolute score drop")
    fig.savefig(output_dir / "performance_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_severity_curves(summary: pd.DataFrame, output_dir: Path, metric: str) -> None:
    locations = ordered_present(LOCATION_ORDER, summary["location"].unique())
    typo_types = ordered_present(TYPO_ORDER, summary["typo_type"].unique())
    ncols = min(2, len(locations))
    nrows = math.ceil(len(locations) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.3 * ncols, 4 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    for axis, location in zip(axes.flat, locations):
        subset = summary[summary["location"] == location]
        for typo_type in typo_types:
            line = subset[subset["typo_type"] == typo_type].sort_values("severity")
            if line.empty:
                continue
            axis.errorbar(
                line["severity"], line["score"],
                yerr=1.96 * line["score_sem"].fillna(0),
                marker="o", linewidth=2, capsize=3,
                color=COLORS.get(typo_type), label=typo_type,
            )
        axis.set_title(location.title())
        axis.set_xlabel("Corrupted words (severity)")
        axis.set_ylabel(metric.upper())
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    for axis in axes.flat[len(locations):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Performance under increasingly severe corruption (95% normal CIs)", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "severity_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_tokenizer(model_name: str, revision: str = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class OpenAITokenizer:
    """Small adapter giving tiktoken the encode interface used below."""

    def __init__(self, model_name: str):
        import tiktoken

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.used_fallback = False
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.used_fallback = True
        self.encoding_name = self.encoding.name

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return self.encoding.encode(text)


def select_embedding_rows(
    df: pd.DataFrame,
    typo_type: str,
    location: str,
    n_samples: int,
    seed: int,
    mitigation: str,
) -> pd.DataFrame:
    corrupt = df[
        (df["condition"] == "corrupted")
        & (df["typo_type"] == typo_type)
        & (df["location"] == location)
        & (df["mitigation"] == mitigation)
        & (df["severity_achieved"] == True)
    ].copy()
    if corrupt.empty:
        raise ValueError(f"No rows found for {typo_type=} and {location=}")
    if "corruption_replicate" in corrupt.columns:
        corrupt = corrupt[corrupt["corruption_replicate"].fillna(0) == 0].copy()
    corrupt = corrupt.sort_values("severity").drop_duplicates(
        ["example_id", "severity"]
    )

    complete_ids = []
    expected = set(corrupt["severity"].astype(int).unique())
    for example_id, group in corrupt.groupby("example_id"):
        if set(group["severity"].astype(int)) == expected:
            complete_ids.append(example_id)
    if not complete_ids:
        raise ValueError("No examples contain every requested severity level")

    rng = random.Random(seed)
    rng.shuffle(complete_ids)
    selected_ids = set(complete_ids[: min(n_samples, len(complete_ids))])
    corrupt = corrupt[corrupt["example_id"].isin(selected_ids)]
    clean = df[
        (df["condition"] == "clean")
        & (df["mitigation"] == mitigation)
        & df["example_id"].isin(selected_ids)
    ].drop_duplicates("example_id")
    selected = pd.concat([clean, corrupt], ignore_index=True)
    selected["severity"] = selected["severity"].fillna(0).astype(int)
    return selected.sort_values(["example_id", "severity"]).reset_index(drop=True)


def add_tokenization_statistics(
    selected: pd.DataFrame, tokenizer, metric: str
) -> pd.DataFrame:
    result = selected.copy()
    result["token_count"] = result["input_question"].map(
        lambda text: len(tokenizer.encode(str(text), add_special_tokens=False))
    )
    clean_counts = (
        result[result["condition"] == "clean"]
        .set_index("example_id")["token_count"]
    )
    clean_scores = (
        result[result["condition"] == "clean"]
        .set_index("example_id")[metric]
    )
    result["clean_token_count"] = result["example_id"].map(clean_counts)
    result["token_increase"] = result["token_count"] - result["clean_token_count"]
    result["clean_example_score"] = result["example_id"].map(clean_scores)
    result["example_drop"] = result["clean_example_score"] - result[metric]
    return result


def save_tokenization_plots(selected: pd.DataFrame, output_dir: Path) -> None:
    corrupt = selected[selected["condition"] == "corrupted"].copy()
    grouped = corrupt.groupby("severity")["token_increase"].agg(["mean", "std", "count"])
    errors = grouped["std"].fillna(0) / np.sqrt(grouped["count"].clip(lower=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].errorbar(
        grouped.index, grouped["mean"], yerr=errors, marker="o", capsize=4,
        color="#0072B2", linewidth=2,
    )
    axes[0].axhline(0, color="0.45", linewidth=1)
    axes[0].set_xlabel("Corrupted words (severity)")
    axes[0].set_ylabel("Additional subword tokens")
    axes[0].set_title("Tokenization fragmentation")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        corrupt["token_increase"], corrupt["example_drop"],
        c=corrupt["severity"], cmap="viridis", alpha=0.65, edgecolors="none",
    )
    axes[1].set_xlabel("Additional subword tokens")
    axes[1].set_ylabel("Per-example score drop")
    axes[1].set_title("Fragmentation vs. performance (colored by severity)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tokenization_fragmentation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    columns = [
        "example_id", "condition", "typo_type", "location", "severity",
        "clean_question", "input_question", "token_count", "token_increase",
        "example_drop", "edits",
    ]
    corrupt[[c for c in columns if c in corrupt.columns]].to_csv(
        output_dir / "tokenization_analysis.csv", index=False
    )
    correlation_table(
        corrupt, "token_increase", "example_drop"
    ).to_csv(output_dir / "tokenization_correlations.csv", index=False)


def load_embedding_model(
    model_name: str, load_in_4bit: bool, revision: str = None
):
    import torch
    from transformers import AutoModel, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModel.from_pretrained(model_name, revision=revision, **kwargs)
    model.eval()
    return model, torch


def rendered_prompts_and_spans(
    selected: pd.DataFrame, tokenizer
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Render the inference prompt and locate the question within each prompt."""
    prompts = []
    spans = []
    for row in selected.itertuples(index=False):
        example = {"task": row.task}
        if row.task == "squad":
            context = getattr(row, "context", None)
            if context is None or pd.isna(context):
                raise ValueError(
                    "The CSV has no SQuAD context. Re-run the experiment with the "
                    "current runner before extracting internal representations."
                )
            example["context"] = str(context)
        question = str(row.input_question)
        messages = make_messages(example, question, row.mitigation)
        template_kwargs = {}
        if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).lower():
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        # Several chat templates (including Llama's) call ``strip()`` on each
        # message. SQuAD contains a few questions with leading/trailing spaces,
        # so the raw CSV value may not occur verbatim in the rendered prompt.
        # Try the exact value first, then only boundary-whitespace variants.
        rendered_question = None
        start = -1
        candidates = list(dict.fromkeys([
            question,
            question.strip(),
            question.rstrip(),
            question.lstrip(),
        ]))
        for candidate in candidates:
            if not candidate:
                continue
            candidate_start = prompt.rfind(candidate)
            if candidate_start >= 0:
                rendered_question = candidate
                start = candidate_start
                break
        if rendered_question is None:
            example_id = getattr(row, "example_id", "unknown")
            raise ValueError(
                "Could not locate the question in the rendered chat prompt "
                f"for example_id={example_id!r}. Question={question!r}"
            )
        prompts.append(prompt)
        spans.append((start, start + len(rendered_question)))
    return prompts, spans


def mean_pool_hidden_states(
    texts: Sequence[str],
    question_spans: Sequence[Tuple[int, int]],
    tokenizer,
    model,
    torch,
    batch_size: int,
) -> np.ndarray:
    """Pool question tokens from the exact chat prompts used for inference."""
    batches: List[np.ndarray] = []
    input_device = model.get_input_embeddings().weight.device
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        batch_spans = question_spans[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        question_mask = torch.zeros_like(encoded["attention_mask"])
        for row_index, (question_start, question_end) in enumerate(batch_spans):
            for token_index, (token_start, token_end) in enumerate(
                offsets[row_index].tolist()
            ):
                if token_end > question_start and token_start < question_end:
                    question_mask[row_index, token_index] = 1
        if (question_mask.sum(dim=1) == 0).any():
            raise ValueError("Question-span tokenization produced an empty token mask")
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        question_mask = question_mask.to(input_device)
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        mask = question_mask.unsqueeze(-1)
        pooled_layers = []
        for hidden in output.hidden_states:
            local_mask = mask.to(hidden.device, dtype=hidden.dtype)
            pooled = (hidden * local_mask).sum(dim=1) / local_mask.sum(dim=1).clamp(min=1)
            pooled_layers.append(pooled.float().cpu().numpy())
        batches.append(np.stack(pooled_layers, axis=1))
        del output, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def correlation_table(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> pd.DataFrame:
    """Report both Pearson and rank correlations overall and within severity."""
    rows = []
    strata = [("overall", frame)] + [
        (f"severity_{int(severity)}", group)
        for severity, group in frame.groupby("severity")
    ]
    for stratum, group in strata:
        valid = group[[x_column, y_column]].dropna()
        for method in ("pearson", "spearman"):
            value = (
                valid[x_column].corr(valid[y_column], method=method)
                if len(valid) > 2 else np.nan
            )
            rows.append({
                "stratum": stratum,
                "method": method,
                "x": x_column,
                "y": y_column,
                "correlation": value,
                "n": len(valid),
            })
    return pd.DataFrame(rows)


def openai_text_embeddings(
    texts: Sequence[str], model_name: str, batch_size: int
) -> np.ndarray:
    """Embed text through OpenAI's dedicated embeddings endpoint."""
    from openai import OpenAI

    client = OpenAI()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(
            model=model_name,
            input=batch,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} texts")
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return numerator / np.clip(denominator, 1e-12, None)


def representation_similarities(
    selected: pd.DataFrame, embeddings: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    clean_index: Dict[str, int] = {
        str(row.example_id): index
        for index, row in selected.iterrows()
        if row.condition == "clean"
    }
    records = []
    all_similarities = []
    for index, row in selected.iterrows():
        if row.condition != "corrupted":
            continue
        clean_embedding = embeddings[clean_index[str(row.example_id)]]
        similarities = cosine_similarity(clean_embedding, embeddings[index])
        all_similarities.append(similarities)
        for layer, value in enumerate(similarities):
            records.append(
                {
                    "example_id": row.example_id,
                    "severity": int(row.severity),
                    "layer": layer,
                    "cosine_similarity": float(value),
                    "cosine_distance": float(1.0 - value),
                    "example_score_drop": float(row.example_drop),
                }
            )
    return pd.DataFrame(records), np.asarray(all_similarities)


def save_layer_similarity(similarities: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for severity, group in similarities.groupby("severity"):
        line = group.groupby("layer")["cosine_similarity"].agg(["mean", "sem"])
        layers = line.index.to_numpy(dtype=float)
        means = line["mean"].to_numpy(dtype=float)
        errors = line["sem"].fillna(0).to_numpy(dtype=float)
        axis.plot(layers, means, linewidth=2, label=f"Severity {severity}")
        axis.fill_between(
            layers,
            means - errors,
            means + errors,
            alpha=0.15,
        )
    axis.set_xlabel("Model layer (0 = token embedding output)")
    axis.set_ylabel("Cosine similarity to clean question")
    axis.set_title("Layer-wise representation preservation")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    similarities.to_csv(output_dir / "layerwise_similarity.csv", index=False)

    final_layer = similarities[
        similarities["layer"] == similarities["layer"].max()
    ].copy()
    fig, axis = plt.subplots(figsize=(6.5, 5))
    axis.scatter(
        final_layer["cosine_distance"], final_layer["example_score_drop"],
        c=final_layer["severity"], cmap="viridis", alpha=0.7, edgecolors="none",
    )
    axis.set_xlabel("Final-layer cosine distance from clean question")
    axis.set_ylabel("Per-example score drop")
    axis.set_title("Representation drift vs. performance (colored by severity)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "internal_drift_vs_performance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    correlation_table(
        final_layer, "cosine_distance", "example_score_drop"
    ).to_csv(output_dir / "internal_drift_correlations.csv", index=False)


def external_embedding_drift(
    selected: pd.DataFrame, embeddings: np.ndarray
) -> pd.DataFrame:
    clean_index: Dict[str, int] = {
        str(row.example_id): index
        for index, row in selected.iterrows()
        if row.condition == "clean"
    }
    records = []
    for index, row in selected.iterrows():
        if row.condition != "corrupted":
            continue
        similarity = float(
            cosine_similarity(
                embeddings[clean_index[str(row.example_id)]], embeddings[index]
            )
        )
        records.append(
            {
                "example_id": row.example_id,
                "typo_type": row.typo_type,
                "location": row.location,
                "severity": int(row.severity),
                "cosine_similarity": similarity,
                "cosine_distance": 1.0 - similarity,
                "example_score_drop": float(row.example_drop),
            }
        )
    return pd.DataFrame(records)


def save_external_embedding_drift(drift: pd.DataFrame, output_dir: Path) -> None:
    grouped = drift.groupby("severity")["cosine_distance"].agg(["mean", "sem"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].errorbar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["sem"].fillna(0),
        marker="o",
        linewidth=2,
        capsize=4,
        color="#D55E00",
    )
    axes[0].set_xlabel("Corrupted words (severity)")
    axes[0].set_ylabel("Cosine distance from clean question")
    axes[0].set_title("External embedding-space drift")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        drift["cosine_distance"],
        drift["example_score_drop"],
        c=drift["severity"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )
    axes[1].set_xlabel("Cosine distance from clean question")
    axes[1].set_ylabel("Per-example score drop")
    axes[1].set_title("Embedding drift vs. performance (colored by severity)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "external_embedding_drift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    drift.to_csv(output_dir / "external_embedding_drift.csv", index=False)
    correlation_table(
        drift, "cosine_distance", "example_score_drop"
    ).to_csv(output_dir / "external_drift_correlations.csv", index=False)


def project_embeddings(final_embeddings: np.ndarray, method: str, seed: int):
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("Install umap-learn or use --projection pca") from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, len(final_embeddings) - 1)),
            min_dist=0.15,
            metric="cosine",
            random_state=seed,
        )
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(final_embeddings)


def save_embedding_trajectories(
    selected: pd.DataFrame,
    final_embeddings: np.ndarray,
    output_dir: Path,
    method: str,
    seed: int,
    representation_label: str = "representation",
) -> None:
    coordinates = project_embeddings(final_embeddings, method, seed)
    plotted = selected.copy()
    plotted["x"] = coordinates[:, 0]
    plotted["y"] = coordinates[:, 1]

    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    for _, group in plotted.groupby("example_id"):
        group = group.sort_values("severity")
        axis.plot(group["x"], group["y"], color="0.72", linewidth=0.8, alpha=0.55)
    severities = sorted(plotted["severity"].unique())
    cmap = plt.get_cmap("viridis")
    maximum = max(severities) or 1
    for severity in severities:
        group = plotted[plotted["severity"] == severity]
        label = "Clean" if severity == 0 else f"Severity {severity}"
        marker = "o" if severity == 0 else ">"
        axis.scatter(
            group["x"], group["y"], s=34, marker=marker,
            color=cmap(severity / maximum), label=label, alpha=0.85,
        )
    axis.set_xlabel(f"{method.upper()} dimension 1")
    axis.set_ylabel(f"{method.upper()} dimension 2")
    axis.set_title(f"Clean-to-corrupted {representation_label} trajectories")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"embedding_trajectories_{method}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    plotted.to_csv(output_dir / f"embedding_coordinates_{method}.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw experiment CSV")
    parser.add_argument("--model", help="HF model name; defaults to CSV model value")
    parser.add_argument("--model_revision", help="matching pinned HF revision, if used")
    parser.add_argument(
        "--embedding_backend", choices=["auto", "hf", "openai"], default="auto",
        help="auto uses the backend recorded in the experiment CSV",
    )
    parser.add_argument(
        "--openai_embedding_model", default="text-embedding-3-large",
        help="Dedicated embedding model used for external OpenAI analysis",
    )
    parser.add_argument("--output_dir", default="capstone_figures")
    parser.add_argument(
        "--mitigation", choices=["none", "self_correct"], default="none",
        help="analyze one matched mitigation condition at a time",
    )
    parser.add_argument(
        "--typo_difficulty", choices=["standard", "hard"], default="hard",
        help="analyze one mutation-intensity condition at a time",
    )
    parser.add_argument("--embedding_typo", choices=TYPO_ORDER, default="keyboard")
    parser.add_argument("--embedding_location", choices=LOCATION_ORDER, default="beginning")
    parser.add_argument("--embedding_samples", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--openai_batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--skip_embeddings", action="store_true",
        help="Create performance plots only; does not load the language model",
    )
    return parser.parse_args()


def main(args) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {
        "example_id", "condition", "typo_type", "location", "severity",
        "mitigation", "input_question", "model", "backend",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if "actual_severity" not in df.columns:
        df["actual_severity"] = df["severity"]
    if "severity_achieved" not in df.columns:
        df["severity_achieved"] = True
    df = df[df["mitigation"] == args.mitigation].copy()
    if "typo_difficulty" in df.columns:
        df = df[
            (df["condition"] == "clean")
            | (df["typo_difficulty"] == args.typo_difficulty)
        ].copy()
    if df.empty:
        raise ValueError(f"No rows found for mitigation={args.mitigation!r}")
    metric = metric_for(df)
    summary = condition_summary(df, metric)
    summary.to_csv(output_dir / "visualization_summary.csv", index=False)
    save_performance_heatmaps(summary, output_dir)
    save_severity_curves(summary, output_dir, metric)
    print(f"Saved performance plots to {output_dir}")

    if args.skip_embeddings:
        return

    recorded_backend = str(df["backend"].dropna().iloc[0]).lower()
    embedding_backend = recorded_backend if args.embedding_backend == "auto" else args.embedding_backend
    if embedding_backend not in {"hf", "openai"}:
        raise ValueError(
            f"Cannot infer embedding backend from CSV backend {recorded_backend!r}; "
            "pass --embedding_backend hf or openai"
        )
    model_name = args.model or str(df["model"].dropna().iloc[0])
    model_revision = args.model_revision
    if model_revision is None and "model_revision" in df.columns:
        recorded_revisions = df["model_revision"].dropna().astype(str).unique()
        if len(recorded_revisions) == 1:
            model_revision = recorded_revisions[0]
    selected = select_embedding_rows(
        df, args.embedding_typo, args.embedding_location,
        args.embedding_samples, args.seed, args.mitigation,
    )
    tokenizer = (
        OpenAITokenizer(model_name)
        if embedding_backend == "openai"
        else load_tokenizer(model_name, revision=model_revision)
    )
    selected = add_tokenization_statistics(selected, tokenizer, metric)
    save_tokenization_plots(selected, output_dir)

    question_texts = selected["input_question"].astype(str).tolist()
    if embedding_backend == "openai":
        print(
            f"Requesting external embeddings from {args.openai_embedding_model}. "
            "This does not expose the generation model's internal hidden states."
        )
        final_embeddings = openai_text_embeddings(
            question_texts, args.openai_embedding_model, args.openai_batch_size
        )
        drift = external_embedding_drift(selected, final_embeddings)
        save_external_embedding_drift(drift, output_dir)
        layer_count = None
        representation_label = "external embedding"
    else:
        print(f"Loading {model_name} to extract contextual representations...")
        model, torch = load_embedding_model(
            model_name, args.load_in_4bit, revision=model_revision
        )
        prompts, question_spans = rendered_prompts_and_spans(selected, tokenizer)
        hidden_embeddings = mean_pool_hidden_states(
            prompts, question_spans, tokenizer, model, torch, args.batch_size
        )
        similarities, _ = representation_similarities(selected, hidden_embeddings)
        save_layer_similarity(similarities, output_dir)
        final_embeddings = hidden_embeddings[:, -1, :]
        layer_count = int(hidden_embeddings.shape[1])
        representation_label = "internal representation"

    save_embedding_trajectories(
        selected, final_embeddings, output_dir, args.projection, args.seed,
        representation_label=representation_label,
    )

    metadata = {
        "input": str(input_path),
        "model": model_name,
        "model_revision": model_revision,
        "embedding_backend": embedding_backend,
        "embedding_model": (
            args.openai_embedding_model if embedding_backend == "openai" else model_name
        ),
        "metric": metric,
        "mitigation": args.mitigation,
        "typo_difficulty": args.typo_difficulty,
        "embedding_typo": args.embedding_typo,
        "embedding_location": args.embedding_location,
        "embedding_samples": int(selected["example_id"].nunique()),
        "projection": args.projection,
        "num_layers_including_embedding_output": layer_count,
        "representation_input": (
            "external_question_only"
            if embedding_backend == "openai"
            else "question_token_span_within_exact_inference_chat_prompt"
        ),
        "tokenizer_or_encoding": (
            tokenizer.encoding_name
            if embedding_backend == "openai"
            else type(tokenizer).__name__
        ),
        "tokenizer_fallback_used": (
            tokenizer.used_fallback if embedding_backend == "openai" else False
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved embedding and tokenization analysis to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
