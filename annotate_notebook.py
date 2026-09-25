"""Add explanatory Markdown to Capstone_Experiments.ipynb without changing code."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Capstone_Experiments.ipynb"
BACKUP = ROOT / "Capstone_Experiments_before_annotations.ipynb"


def markdown(text: str) -> dict:
    cleaned = text.strip() + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": cleaned.splitlines(keepends=True),
    }


INTRODUCTION = markdown(
    r"""
# Controlled Typographical Corruption in Language Models

This notebook runs the complete experimental pipeline for evaluating how two instruction-tuned language models respond to controlled misspellings. The central outcome is **paired degradation**: for each example and corruption condition, the score on corrupted input is subtracted from that same example's clean-input score. Pairing prevents naturally easy or difficult examples from dominating the robustness comparison.

## Research questions

1. Does performance deteriorate as more words are corrupted?
2. Do Qwen3-4B and Llama-3.2-3B-Instruct differ in robustness when they receive the same examples and matched typo targets?
3. Does the position of a typo matter?
4. Are tokenization fragmentation and hidden-state displacement associated with performance loss?
5. Can a generic self-correction instruction mitigate the damage?
6. Do conclusions from extractive question answering generalize to mathematical reasoning?
7. Are the results sensitive to the random realization of a corruption?

## Experimental conventions

- **Severity** is the number of distinct corrupted words, not the number of character edits.
- **Standard difficulty** applies one character operation per selected word. **Hard difficulty** targets words that support two non-conflicting operations.
- `--match_typo_targets` requires target words to be eligible for every requested typo type. This keeps keyboard, deletion, and transposition comparisons from being confounded by different word choices.
- `--locations any` samples eligible words anywhere in the question. The position experiment separately constrains targets to the beginning, middle, end, or an IDF-selected keyword.
- Hugging Face inference is deterministic greedy decoding. The seed controls dataset shuffling and corruption generation.
- Confidence intervals are paired, example-level bootstrap intervals. Corruption conditions and replicates are averaged within an example before inference so they are not treated as independent observations.

The notebook is organized in the order the experiments should be interpreted, rather than as a single undifferentiated sequence of commands.
"""
)


SETUP = markdown(
    r"""
## 0. Environment setup and validation

The next cells mount Google Drive, move into the project directory, install the pinned project requirements, run the repository's validation checks, and report whether PyTorch can see the Colab GPU.

**Why this matters.** Model inference is the expensive part of the study. Running validation first catches missing files, incompatible arguments, and basic integration errors before a multi-hour GPU experiment begins. The CUDA check confirms that the Hugging Face models can run on the GPU rather than silently falling back to CPU execution.

Expected setup sequence:

1. Mount Drive and enter `Capstone_Thesis/`.
2. Install `requirements.txt`.
3. Run `validate_capstone.py`; do not continue if validation fails.
4. Confirm `CUDA available: True` and record the GPU model for reproducibility.
"""
)


PILOT = markdown(
    r"""
## 1. Pilot study: corruption feasibility and calibration

### Purpose

This is a **design-validation pilot**, not a confirmatory model comparison. It runs Qwen3-4B on 20 shuffled SQuAD validation examples beginning at offset 200. The offset deliberately separates these examples from the 200-example final sample beginning at offset 0.

### What is varied and controlled

- Typo family: keyboard substitution, deletion, and adjacent-character transposition.
- Severity: 1, 2, or 3 distinct corrupted words.
- Location: `any`, so target words may occur anywhere in the question.
- Difficulty: `hard`, requesting two non-conflicting character operations per selected word.
- Target selection: matched across typo families.
- Model, task, decoding, sample, and seed are held fixed.

### What this run is meant to check

The primary pilot outputs are operational diagnostics rather than model-performance claims:

- `error_rate`: whether generation completed successfully;
- `mean_actual_severity`: the number of words that were actually corrupted;
- `severity_completion_rate`: the fraction of examples reaching the requested word-level severity;
- `mean_actual_char_operations`: the realized character-edit count.

The command writes raw rows to `results/qwen_squad_pilot.csv` and also produces a summary CSV and metadata JSON. Pilot observations must remain excluded from final hypothesis tests because they informed the experimental design.
"""
)


PILOT_READING = markdown(
    r"""
### Pilot diagnostic reading guide

The following table checks whether the corruption generator achieved what was requested. A value below the requested severity is not a model error; it means the question did not contain enough eligible, distinct words for that condition. For hard corruption, a fully completed condition should usually have about two character operations per corrupted word.

The pilot supports proceeding when generation errors are absent and completion is high enough for the intended conditions. The final analysis must still report realized severity and completion so that nominal severity is not mistaken for guaranteed severity.
"""
)


SQUAD_CORE = markdown(
    r"""
## 2. Core SQuAD robustness experiment

### Research question

How much does extractive question-answering F1 decline as one, two, or three words in a question receive hard typographical corruption, and is that decline different for Qwen3-4B and Llama-3.2-3B-Instruct?

### Design

Both models receive the same 200 deterministically shuffled SQuAD validation examples, the same three typo families, and the same random seed. `--match_typo_targets` aligns eligible target words across typo families. The unmodified context is always supplied; only the question is corrupted. Each condition is scored against the example's clean baseline, making the analysis paired at the example level.

The first command runs Qwen. The later Llama command repeats the design without changing the sample or corruption configuration. Raw outputs are saved as `qwen_squad_core.csv` and `llama_squad_core.csv`; each run also produces condition summaries, a primary severity summary, and metadata.

### Primary estimand

For example $i$ and condition $c$,

$$d_{ic}=\text{F1}_{i,\mathrm{clean}}-\text{F1}_{i,c}.$$

Positive values mean that corruption reduced performance. The primary severity analysis balances typo families by averaging their drops within each example before estimating the mean and its bootstrap interval.
"""
)


HF_AUTH = markdown(
    r"""
### Hugging Face authentication for Llama

`meta-llama/Llama-3.2-3B-Instruct` is a gated Hugging Face repository. The login cell authenticates the Colab session and prints the active account so access problems can be distinguished from code failures. The account must already have accepted Meta's model license. Do not place an access token directly in the notebook; use the interactive login prompt or a Colab secret.
"""
)


SQUAD_QC = markdown(
    r"""
### Corruption and execution quality check

Before comparing models, inspect the Llama condition summary below. The same fields examined in the pilot verify that requested severities were realized and that no conditions failed during generation. A nominal severity should only be interpreted as fully controlled when `severity_completion_rate` is near 1.0; any shortfall is retained and reported rather than silently discarded.
"""
)


SQUAD_COMPARE = markdown(
    r"""
### Paired cross-model comparison

`compare_models.py` loads the two raw SQuAD runs, restricts them to the shared example IDs, reconstructs clean-minus-corrupted F1 drops, and computes 5,000 example-level bootstrap samples.

Two complementary tables are produced:

- `cross_model_gap_by_severity.csv` averages over the matched typo families and answers the main model-comparison question at each severity.
- `cross_model_gap_by_condition.csv` retains typo family and severity for exploratory diagnosis.

The reported gap is

$$\text{gap}=\text{drop}_{\mathrm{Qwen}}-\text{drop}_{\mathrm{Llama}}.$$

Therefore, a **negative** gap favors Qwen (Qwen degraded less), while a **positive** gap favors Llama. A confidence interval containing zero does not establish a reliable difference at that condition. The typo-specific table contains multiple exploratory contrasts and should not be treated as a set of independently pre-registered tests.
"""
)


POSITION = markdown(
    r"""
## 3. Position-specific SQuAD robustness

### Research question

Does the location of a single, standard typo change model robustness, and does either model show a consistent positional advantage?

### Why this is a separate experiment

The core study uses difficult two-operation corruptions at unrestricted locations. This experiment deliberately uses **severity 1** and **standard difficulty** to isolate location without simultaneously changing the number of corrupted words. It should not be pooled with the hard-corruption core study.

### Location definitions

- `beginning`, `middle`, and `end` restrict eligible words to the corresponding third of the question.
- `keyword` selects the eligible non-stopword with the highest smoothed corpus IDF, using word length as a tie-break.
- Location fallback is disabled. If a requested region lacks an eligible word, the condition is recorded as incomplete rather than corrupted elsewhere.

Qwen and Llama are run on the same 200 SQuAD examples. The diagnostic cell verifies one realized corrupted word, one character operation, zero generation errors, and full completion before the models are compared.
"""
)


POSITION_COMPARE = markdown(
    r"""
### Position experiment: model comparison

The comparison uses paired clean-minus-corrupted F1 drops and 5,000 example-bootstrap samples. The severity-level table gives the overall model gap after balancing typo types and locations. The condition table gives 12 exploratory typo-by-location gaps.

As in the core study, the sign is Qwen drop minus Llama drop: negative values favor Qwen. Position-specific claims should be made only when the corresponding interval excludes zero and the pattern is coherent across related conditions; isolated point estimates are not sufficient evidence of a general location effect.
"""
)


MECHANISM = markdown(
    r"""
## 4. Tokenization and internal-representation analysis

### Purpose

Performance scores show **whether** typos hurt, but not how the input changes inside each model. These commands analyze 40 matched keyboard-corruption trajectories from the hard SQuAD core runs.

The visualization script measures:

1. the increase in question-token count relative to the clean question;
2. clean-to-corrupted cosine similarity or distance across hidden layers;
3. correlations between tokenization change, final-layer displacement, and per-example F1 loss;
4. a PCA projection for qualitative visualization.

The analysis is run separately for Qwen and Llama because their tokenizers, hidden dimensions, and representation spaces differ. **PCA coordinates and raw vectors must not be compared directly across models.** Cross-model interpretation should focus on within-model clean-versus-corrupted trajectories, layerwise patterns, and separately computed summary statistics.

This is a descriptive mechanism analysis. A monotonic average increase in fragmentation or distance does not by itself demonstrate that the quantity causes individual prediction failures; the within-severity correlations are the more relevant diagnostic for that claim.
"""
)


MITIGATION = markdown(
    r"""
## 5. Prompt-based self-correction mitigation

### Research question

Can a generic instruction to infer the intended wording before answering recover performance on corrupted SQuAD questions?

### Design

The experiment uses Llama, the same 200-example SQuAD sampling procedure, hard corruption, matched typo targets, and severities 2 and 3, where degradation is large enough for mitigation to be meaningful. Every corrupted condition is evaluated twice:

- `none`: the original task prompt;
- `self_correct`: adds an instruction that the input may contain typographical errors and asks the model to infer the intended wording silently.

The mitigation summary directly pairs the two corrupted-input scores for each example and reports

$$\text{improvement}=\text{score}_{\mathrm{self\_correct}}-\text{score}_{\mathrm{none}}.$$

Positive values favor self-correction. This direct contrast is the correct estimand because the mitigation changes the prompt and can also change clean performance. A smaller clean-minus-corrupted drop under mitigation is not automatically an improvement if the clean score also fell. Results should be interpreted by typo family and severity, with the bootstrap interval and clean-score cost reported together.
"""
)


GSM8K = markdown(
    r"""
## 6. Task generalization: GSM8K mathematical reasoning

### Research question

Does the robustness ordering observed on SQuAD generalize to a task that requires multi-step mathematical reasoning and exact numeric answers?

### Design differences from SQuAD

- The sample contains 100 deterministically shuffled GSM8K test problems.
- The metric is exact final-answer accuracy rather than token-level F1.
- The prompt requires a final answer in `FINAL: <answer>` form.
- `--max_new_tokens 512` allows enough space for a reasoning trace.
- The same typo families, matched targets, hard difficulty, severities 1--3, seed, and unrestricted location are retained.

Qwen and Llama run on the same examples, enabling paired cross-model gaps. Because accuracy is binary and the sample is smaller than SQuAD, wider confidence intervals are expected. The purpose is to test transfer of the robustness conclusion, not to merge SQuAD F1 and GSM8K accuracy into one score.
"""
)


GSM_COMPARE = markdown(
    r"""
### GSM8K quality check and paired model comparison

The displayed Llama summaries first verify generation success and realized corruption severity. `compare_models.py` then applies the same paired analysis used for SQuAD, now using accuracy drops. The gap remains Qwen drop minus Llama drop, so negative values favor Qwen and positive values favor Llama.

The severity table is the main task-generalization result. The typo-by-severity table is diagnostic. If the point-estimate ranking changes but the interval includes zero, the appropriate conclusion is that the experiment does not establish a reliable cross-model difference—not that the models are proven equally robust.
"""
)


REPLICATION = markdown(
    r"""
## 7. Sensitivity to the random corruption realization

### Motivation

A single seed produces only one valid misspelling of each selected word. A result could therefore reflect the particular replacement letters, deleted characters, or transposed pairs rather than a stable response to the corruption condition.

### Design

This sensitivity analysis reruns the most demanding SQuAD condition—hard severity 3—on 100 examples with `--corruption_replicates 3`. Each replicate is an independent corruption realization for the same example and condition. Qwen and Llama again receive matched samples and settings.

Replicates are **not** additional independent observations. The analysis averages them within each example before bootstrapping, so the inferential sample size remains 100 examples rather than 300 pseudo-replicates. The replicated model gap should be compared with the direction and uncertainty of the 200-example core severity-3 result. Agreement in direction supports stability; a wider interval is expected from the smaller sample.
"""
)


OUTPUT_GUIDE = markdown(
    r"""
## 8. Output map and interpretation checklist

Each robustness command creates three principal artifacts:

- `<name>.csv`: example-level clean and corrupted predictions, scores, corruption details, and any generation errors;
- `<name>_summary.csv` and `<name>_primary_summary.csv`: condition-level and severity-level estimates;
- `<name>_metadata.json`: arguments, package versions, model runtime details, decoding configuration, and provenance.

Comparison and visualization scripts create additional CSV tables and PNG figures in their specified output directories.

Before using a result in the paper, verify:

1. the intended examples and model were loaded;
2. `error_rate` is reported and failed generations are not silently scored;
3. realized severity and character-operation counts match the intended manipulation;
4. the clean baseline is paired to the same example;
5. the independent unit is `example_id`, including replicated runs;
6. the sign of each model or mitigation contrast is stated explicitly;
7. confidence intervals, sample sizes, and exploratory-versus-primary status accompany point estimates;
8. the 20-example pilot is not pooled into final results.

Together, these experiments separate the main robustness finding (core SQuAD), boundary conditions (position and GSM8K), a candidate mechanism (tokenization and representation drift), an intervention test (self-correction), and a random-corruption sensitivity analysis.
"""
)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original_code = [
        copy.deepcopy(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
    ]

    if not BACKUP.exists():
        shutil.copy2(NOTEBOOK, BACKUP)

    insert_before = {
        0: [INTRODUCTION, SETUP],
        5: [PILOT],
        9: [HF_AUTH],
        11: [SQUAD_QC],
        12: [SQUAD_COMPARE],
        19: [POSITION_COMPARE],
        22: [MECHANISM],
        32: [GSM_COMPARE],
    }
    replacements = {
        7: SQUAD_CORE,
        15: POSITION,
        24: MITIGATION,
        27: GSM8K,
        35: REPLICATION,
        39: OUTPUT_GUIDE,
    }
    insert_after = {6: [PILOT_READING]}
    discard_empty = {28}

    cells = []
    for index, cell in enumerate(notebook["cells"]):
        cells.extend(copy.deepcopy(insert_before.get(index, [])))
        if index in replacements:
            cells.append(copy.deepcopy(replacements[index]))
        elif index in discard_empty:
            pass
        elif not (
            cell.get("cell_type") == "code"
            and not "".join(cell.get("source", [])).strip()
        ):
            cells.append(cell)
        cells.extend(copy.deepcopy(insert_after.get(index, [])))

    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["capstone_annotation_version"] = 1

    revised_code = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
    ]
    if revised_code != original_code:
        raise RuntimeError("Executable cells changed while annotating the notebook")

    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Annotated {NOTEBOOK.name}: {len(notebook['cells'])} cells, "
        f"{sum(c['cell_type'] == 'markdown' for c in notebook['cells'])} Markdown cells."
    )
    print(f"Original preserved as {BACKUP.name}.")


if __name__ == "__main__":
    main()
