# LLM Typo Robustness Capstone — Final Run Guide

## 1. Keep this structure in a fresh project folder

```text
Capstone/
├── capstone/
│   ├── __init__.py
│   ├── config.py       # CLI arguments and validation
│   ├── experiment.py   # experiment loop and output files
│   ├── models.py       # Hugging Face and OpenAI adapters
│   ├── statistics.py   # paired bootstrap summaries
│   ├── tasks.py        # datasets, prompts, parsing, and metrics
│   └── typos.py        # targeting and deterministic corruptions
├── capstone_robustness.py
├── capstone_visualizations.py
├── compare_models.py
├── validate_capstone.py
├── ANALYSIS_PLAN.md
├── requirements.txt
└── CAPSTONE_RUN_GUIDE.md
```

- `capstone_robustness.py` — backward-compatible CLI entry point
- `capstone_visualizations.py` — plots and representation analysis
- `compare_models.py` — paired cross-model comparison
- `validate_capstone.py` — fast model-free correctness checks
- `ANALYSIS_PLAN.md` — frozen primary/secondary analysis definitions
- `requirements.txt` — dependencies
- `CAPSTONE_RUN_GUIDE.md` — this workflow

Do not copy old result CSVs into the fresh folder. The runner refuses to replace
an existing CSV unless `--overwrite` is explicitly supplied.

For Colab, zip and upload the entire project rather than uploading the Python
files individually. After extraction, run commands from the directory that
contains `capstone_robustness.py`; the `capstone/` package must remain beside it.

In Colab, make the project directory explicit in the first cell so results are
written to persistent Google Drive storage:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/Capstone_Thesis

from pathlib import Path
required = [
    "capstone_robustness.py",
    "capstone_visualizations.py",
    "compare_models.py",
    "validate_capstone.py",
    "requirements.txt",
    "capstone",
]
missing = [name for name in required if not Path(name).exists()]
assert not missing, f"Wrong Colab directory or missing uploads: {missing}"
```

Do not leave the `%cd` line commented. At the end, download or retain the raw
CSV, primary/condition summaries, metadata JSON, and figure directories—not
only the PNG files.

## 2. Install and validate

```bash
pip install -r requirements.txt
python validate_capstone.py
```

The validation command checks nested severity, strict location fidelity,
standard/hard target equivalence, matched targets and character-operation counts
across typo types, and metric behavior without loading an LLM.

For a gated Llama model, accept its Hugging Face license and authenticate. For
OpenAI runs, set `OPENAI_API_KEY` in the environment; never put the key in a
notebook that will be shared.

For final confirmatory runs, replace `MODEL_COMMIT_HASH` and
`DATASET_COMMIT_HASH` below with real Hugging Face commit hashes. You may omit
those two flags while testing the workflow, but an unpinned test run should not
be presented as the frozen final result.

## 3. Research design

The runner supports a paired factorial design:

- task: SQuAD extractive QA or GSM8K reasoning
- typo type: keyboard substitution, deletion, or transposition
- target: beginning, middle, end, keyword, or any
- severity: number of distinct corrupted words
- difficulty: standard (one character operation per word) or hard (two
  non-conflicting operations on each successfully selected word)
- mitigation: none or prompt-based self-correction

By default, `--match_typo_targets` is enabled. Keyboard substitution, deletion,
and transposition therefore attack the same words, and hard attacks perform two
character operations per selected word. This removes target selection and edit
count as explanations for differences among typo types. Pass
`--no-match_typo_targets` only for a clearly labeled natural-eligibility
ablation.

Severity is nested: severity 2 preserves severity 1's edit and adds another;
severity 3 preserves both earlier edits. Positional attacks never spill outside
their requested third. If an input cannot support the requested number of edits,
the CSV records `actual_severity` and `severity_achieved`, and the summary reports
the completion rate.

The default keyword strategy selects the non-stopword with highest smoothed IDF
in the sampled question corpus, with word length as a tie-break. Use
`--keyword_strategy longest` only as a documented ablation.

## 4. Pilot before every new model/task

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 20 \
  --sample_offset 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --output qwen_squad_pilot.csv
```

Check the summary's `error_rate` and `severity_completion_rate` before scaling.
Do not proceed if generation errors are nonzero or completion is unexpectedly
low. `--sample_offset 200` keeps these pilot cases disjoint from the first 200
examples used in the confirmatory run; do not combine pilot results with the
final analysis.

## 5. Recommended core experiment

Run the same 200 sampled SQuAD examples for each model by keeping `--seed 42`.
The primary severity experiment uses `location=any`: fixed thirds often do not
contain three words eligible for two non-conflicting operations. Positional
effects are tested separately at severity 1.

### Qwen

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --model_revision MODEL_COMMIT_HASH \
  --dataset_revision DATASET_COMMIT_HASH \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --keyword_strategy idf \
  --output qwen_squad_hard.csv
```

### Llama

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --model_revision MODEL_COMMIT_HASH \
  --dataset_revision DATASET_COMMIT_HASH \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --keyword_strategy idf \
  --output llama_squad_hard.csv
```

### OpenAI

Use a pinned API model snapshot available to your account rather than a moving
alias, and record its exact identifier in the report.

```bash
python capstone_robustness.py \
  --task squad \
  --backend openai \
  --model PINNED_MODEL_ID \
  --dataset_revision DATASET_COMMIT_HASH \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --keyword_strategy idf \
  --output openai_squad_hard.csv
```

## 6. Position experiment

Test location separately with one standard operation on one word. This avoids
turning the location comparison into a comparison of attack-completion rates.
Only interpret locations whose completion rates are acceptably high.

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations beginning middle end keyword \
  --severities 1 \
  --typo_difficulty standard \
  --output qwen_squad_positions.csv
```

## 7. Focused difficulty ablation

Comparing standard and hard within one run attacks the same word sequence at
both intensities. Use `any` so severity is not confounded with regional word
availability:

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulties standard hard \
  --output qwen_squad_difficulty_ablation.csv
```

## 8. Mitigation experiment

This run creates a separate matched clean baseline for each mitigation and a
paired `<name>_mitigation_summary.csv`.

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any keyword \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --mitigations none self_correct \
  --output qwen_squad_mitigation.csv
```

## 9. Corruption-replication check

The core factorial run uses one deterministic corruption per example. Run this
smaller confirmatory check with three independent corruptions to verify that the
main conclusion is not tied to one random keyboard neighbor or character
choice. Replicates are averaged within each example before bootstrapping, so
they do not inflate the sample size.

```bash
python capstone_robustness.py \
  --task squad \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 200 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --corruption_replicates 3 \
  --output qwen_squad_replicated.csv
```

## 10. Task-generalization experiment

GSM8K generations are longer and more expensive. A focused `any`-location run
tests whether findings generalize beyond extractive QA:

```bash
python capstone_robustness.py \
  --task gsm8k \
  --backend hf \
  --model Qwen/Qwen3-4B \
  --n_samples 100 \
  --seed 42 \
  --typo_types keyboard deletion transposition \
  --match_typo_targets \
  --locations any \
  --severities 1 2 3 \
  --typo_difficulty hard \
  --max_new_tokens 512 \
  --output qwen_gsm8k_generalization.csv
```

## 11. Visualization

Performance-only plots do not load a model:

```bash
python capstone_visualizations.py \
  --input qwen_squad_hard.csv \
  --mitigation none \
  --typo_difficulty hard \
  --output_dir qwen_squad_figures \
  --skip_embeddings
```

For Qwen/Llama internal representation analysis, analyze one trajectory at a
time to keep GPU use manageable:

```bash
python capstone_visualizations.py \
  --input qwen_squad_hard.csv \
  --model Qwen/Qwen3-4B \
  --embedding_backend hf \
  --embedding_typo keyboard \
  --embedding_location beginning \
  --embedding_samples 40 \
  --batch_size 4 \
  --projection pca \
  --output_dir qwen_keyboard_beginning_figures
```

Use the same numerical precision for Qwen and Llama in the core comparison. If
your GPU requires `--load_in_4bit`, add it to **every** Hugging Face experiment
and embedding-analysis command, then report quantization as part of the setup.
Do not compare a 4-bit model against a full-precision model without labeling
that as a separate quantization condition.

For OpenAI results, use a dedicated external embedding model:

```bash
python capstone_visualizations.py \
  --input openai_squad_hard.csv \
  --embedding_backend openai \
  --openai_embedding_model text-embedding-3-large \
  --embedding_typo keyboard \
  --embedding_location beginning \
  --embedding_samples 40 \
  --projection pca \
  --output_dir openai_keyboard_beginning_figures
```

OpenAI embedding drift is not the evaluated generation model's hidden state and
must be described as external embedding-space analysis. PCA is the primary,
reproducible projection; UMAP is exploratory.

## 12. Outputs and reporting

Every experiment writes:

- `<name>.csv`: per-example predictions, edits, achieved severity, latency/errors
- `<name>_summary.csv`: paired degradation estimates and bootstrap intervals
- `<name>_primary_summary.csv`: pre-specified severity effects, balanced across
  typo types and locations; use this table for the main statistical claims
- `<name>_metadata.json`: configuration, design definitions, and package versions
- `<name>_mitigation_summary.csv`: only when both mitigations are requested

Use F1 as the primary SQuAD metric and exact match as secondary. Use final-answer
accuracy for GSM8K. Report effect sizes and paired confidence intervals, not only
condition means or standard deviations. The
`bootstrap_probability_*_le_zero` columns are directional bootstrap summaries,
not classical multiple-comparison-adjusted p-values.

Internal Hugging Face representations are extracted from the exact system +
context + question chat prompt used at inference and pooled over the question
token span only. Correlation CSVs report Pearson and Spearman associations both
overall and within severity to avoid interpreting a severity-driven pooled
association as a within-severity mechanism.

The embedding analyses are explanatory and correlational. They can show that
token fragmentation or representation drift accompanies performance loss, but
they do not establish that either mechanism causes the failure.

After the Qwen, Llama, and OpenAI core runs are complete, create the direct
cross-model comparison:

```bash
python compare_models.py \
  --inputs qwen_squad_hard.csv llama_squad_hard.csv openai_squad_hard.csv \
  --mitigation none \
  --typo_difficulty hard \
  --output_dir cross_model_figures
```

The script verifies shared example IDs and restricts comparisons to their
intersection before plotting model-level severity and typo-type effects. Its
error bars are paired example-bootstrap 95% intervals, and
`cross_model_gap_by_severity.csv` directly estimates the difference in
degradation between each model pair. Positive gaps mean model A degraded more.
