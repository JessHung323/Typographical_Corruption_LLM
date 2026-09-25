# Controlled Typographical Corruption in Language Models

This project evaluates how instruction-tuned language models respond when words
in otherwise valid user questions contain controlled typographical errors. It
compares **Qwen3-4B** and **Llama-3.2-3B-Instruct** on extractive question
answering (SQuAD) and mathematical reasoning (GSM8K), then investigates typo
position, prompt-based self-correction, tokenization fragmentation, internal
representation drift, and sensitivity to random corruption realizations.

The central measurement is a paired score drop:

```text
degradation = clean score - corrupted-input score
```

Every corrupted result is paired with the clean score for the same example.
Positive degradation means that the typo reduced performance.

## Research questions

1. Does performance degrade as more distinct words are corrupted?
2. Do the two model families differ in robustness under matched conditions?
3. Does the location of a typo affect performance?
4. Do tokenization fragmentation and hidden-state displacement track failure?
5. Does a generic self-correction instruction mitigate corruption?
6. Does the model ranking transfer from SQuAD to GSM8K?
7. Are conclusions stable across different valid realizations of a typo?

## Repository structure

```text
Typographical_Corruption_LLM/
├── Capstone_Experiments.ipynb       # Annotated, end-to-end Colab notebook
├── capstone_robustness.py           # Main experiment CLI
├── capstone_visualizations.py       # Tokenization and representation analysis
├── compare_models.py                # Paired cross-model inference and plots
├── validate_capstone.py             # Fast model-free correctness checks
├── capstone/
│   ├── config.py                    # CLI arguments and validation
│   ├── experiment.py                # Experiment loop and output management
│   ├── models.py                    # Hugging Face and OpenAI adapters
│   ├── statistics.py                # Paired bootstrap summaries
│   ├── tasks.py                     # Datasets, prompts, parsing, and metrics
│   └── typos.py                     # Target selection and deterministic edits
├── results/                         # Raw runs, summaries, metadata, and figures
├── requirements.txt                 # Python dependencies
├── ANALYSIS_PLAN.md                 # Primary and secondary estimands
└── CAPSTONE_RUN_GUIDE.md            # Complete command reference
```

`Capstone_Experiments.ipynb` is the recommended entry point. Its Markdown cells
explain the purpose, controls, estimands, outputs, and interpretation of every
experiment. `CAPSTONE_RUN_GUIDE.md` contains additional command-line variants
and fresh-project instructions.

## Experimental definitions

### Typo families

- **Keyboard substitution:** replaces a character with a neighboring QWERTY
  key.
- **Deletion:** removes a character.
- **Transposition:** swaps adjacent, non-identical alphabetic characters.

### Severity and difficulty

- **Severity** is the number of distinct corrupted words, not the number of
  character edits.
- **Standard difficulty** applies one character operation per selected word.
- **Hard difficulty** selects words that support two non-conflicting operations
  and applies two operations per successfully selected word.

Severity is nested: when possible, severity 2 preserves the severity-1 target
and adds another word; severity 3 preserves the earlier targets and adds a
third. The output records `actual_severity`, `severity_achieved`, and realized
character-operation counts when a question cannot support the full request.

### Matched targets

The finalized experiments use `--match_typo_targets`. A selected word must be
eligible for keyboard substitution, deletion, and transposition, so comparisons
among typo families use the same target words and requested operation counts.
This removes target selection as an alternative explanation for typo-type
differences.

### Locations

- `any`: any eligible word in the question;
- `beginning`, `middle`, `end`: an eligible word in the corresponding third;
- `keyword`: the eligible non-stopword with the highest smoothed corpus IDF,
  with word length used as a tie-break.

The generator does not silently fall back to another region when a requested
location has no eligible target.

## Experiment map

| Study | Models | Sample | Corruption design | Purpose |
|---|---|---:|---|---|
| Pilot | Qwen | 20 SQuAD examples at offset 200 | Any location, hard, severities 1–3 | Validate generation and corruption completion without reusing the final sample |
| Core SQuAD | Qwen and Llama | 200 | Any location, hard, severities 1–3 | Primary robustness and cross-model comparison |
| Position | Qwen and Llama | 200 | Beginning/middle/end/keyword, standard, severity 1 | Isolate positional effects |
| Mechanism | Qwen and Llama | 40 matched trajectories per model | Keyboard, any location, hard | Measure token fragmentation and layerwise clean–corrupted displacement |
| Mitigation | Llama | 200 | Any location, hard, severities 2–3 | Compare no mitigation with generic self-correction |
| GSM8K | Qwen and Llama | 100 | Any location, hard, severities 1–3 | Test task generalization with exact-answer accuracy |
| Replication | Qwen and Llama | 100 SQuAD examples | Any location, hard, severity 3, three realizations | Test sensitivity to the random corruption realization |

The pilot is for design validation only and is not pooled with confirmatory
results.

## Quick start in Google Colab

### 1. Select a GPU runtime

In Colab, choose **Runtime → Change runtime type → GPU**. A T4 can run the
finalized 3B–4B models, although faster GPUs reduce total runtime.

### 2. Mount Drive and enter the project directory

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/Capstone_Thesis
```

The working directory must contain `capstone_robustness.py` and the adjacent
`capstone/` package.

### 3. Install and validate

```python
!pip install -r requirements.txt
!python validate_capstone.py

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

Do not start long model runs if validation fails or CUDA is unavailable.

### 4. Authenticate for the gated Llama repository

The Meta Llama model requires an approved Hugging Face account:

```python
from huggingface_hub import login, whoami
login()
print(whoami())
```

Accept the model license on Hugging Face before running this cell. Never store a
token directly in a shared notebook.

### 5. Run the annotated notebook

Open `Capstone_Experiments.ipynb` and execute the sections in order. The
notebook begins with a disjoint pilot, followed by the final experiments and
their comparisons. Existing saved outputs can be read without rerunning the
models.

## Minimal command-line example

The following command runs the finalized Qwen core SQuAD design:

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
  --keyword_strategy idf \
  --output results/qwen_squad_core.csv
```

The runner refuses to overwrite an existing output by default. Choose a new
filename or pass `--overwrite` deliberately.

## Output files

A run whose output is `results/example.csv` creates:

- `example.csv`: example-level clean and corrupted predictions, task scores,
  target words, realized edits, and errors;
- `example_summary.csv`: results by typo type, location, severity, difficulty,
  and mitigation;
- `example_primary_summary.csv`: severity-level results balanced within each
  example over the requested conditions;
- `example_metadata.json`: complete arguments, environment versions, decoding
  settings, resolved model information, and execution counts;
- `example_mitigation_summary.csv`: direct paired mitigation effects when both
  `none` and `self_correct` are requested.

`compare_models.py` writes cross-model figures and tables, including:

- `cross_model_gap_by_severity.csv`;
- `cross_model_gap_by_condition.csv`;
- `cross_model_summary.csv`;
- severity and typo-type comparison figures.

`capstone_visualizations.py` writes tokenization tables, layerwise similarity
tables, correlation summaries, PCA figures, and run metadata.

Keep raw CSVs and metadata JSON files, not only the figures. They are required
to audit sample pairing, corruption completion, package versions, and model
provenance.

## Statistical interpretation

The independent unit is always `example_id`.

- SQuAD uses token-level F1; GSM8K uses exact final-answer accuracy.
- Score drops are calculated within each example.
- Confidence intervals use 5,000 paired bootstrap samples over examples.
- Multiple typo conditions are balanced within an example before the primary
  severity estimate is calculated.
- Corruption replicates are averaged within an example and are never counted as
  additional independent observations.

Cross-model files define the gap as:

```text
model gap = degradation(model A) - degradation(model B)
```

When Qwen is model A and Llama is model B, a negative gap favors Qwen because
Qwen degraded less. A confidence interval containing zero does not establish a
reliable model difference.

The mitigation summary defines:

```text
improvement = corrupted score with self-correction
            - corrupted score without mitigation
```

Positive values favor self-correction. Report the direct improvement and any
change in clean performance; a smaller clean-minus-corrupted gap can be
misleading if the mitigation also lowers the clean score.

## Summary of finalized findings

- Both models degraded monotonically as more SQuAD question words were
  corrupted.
- At SQuAD severity 3, Qwen lost 12.76 F1 points and Llama lost 20.25. The
  paired Qwen-minus-Llama degradation gap was -7.48 points with a 95% interval
  of [-12.30, -2.70].
- The Qwen robustness advantage did not transfer reliably to GSM8K; every
  severity-level cross-model interval included zero.
- A generic self-correction instruction helped one keyboard condition, harmed
  one transposition condition, and was not a consistent defense.
- Typo severity increased token fragmentation and final-layer displacement,
  but within-severity correlations with individual F1 loss were weak.
- Averaging three severity-3 corruption realizations preserved the direction of
  the SQuAD comparison, but the smaller sensitivity sample produced a
  borderline cross-model interval.

These findings support a task- and corruption-dependent interpretation of typo
robustness rather than a single universal model ranking.

## Reproducibility checklist

Before treating a run as final:

1. Retain the same seed, sample offset, sample size, and precision setting for
   every compared model.
2. Record or pin exact model and dataset revisions. The metadata JSON records
   resolved model information from completed runs.
3. Confirm that `error_rate` is reported and generation failures are not
   silently converted into ordinary zero scores.
4. Inspect `mean_actual_severity`, `severity_completion_rate`, and
   `mean_actual_char_operations`.
5. Keep pilot examples disjoint from final examples.
6. Use `example_id` as the inferential unit, including replicated corruptions.
7. State the sign of every model or mitigation contrast.
8. Report point estimates, 95% intervals, and sample sizes together.
9. Label typo-specific, location-specific, correlation, and PCA analyses as
   exploratory unless they were pre-specified as primary.

## Important limitations

- The generator evaluates controlled synthetic typos; it does not estimate the
  frequency or average effect of naturally occurring user misspellings.
- The main model comparison covers two compact open models and should not be
  generalized to all contemporary language models.
- SQuAD and GSM8K represent only two task families.
- Position, typo-specific, mitigation, and mechanism analyses contain multiple
  exploratory comparisons.
- PCA is a visualization, not evidence of cluster separation or causality.
- Internal representations from different model architectures occupy different
  spaces; coordinates and hidden vectors are not directly comparable across
  Qwen and Llama.

## Troubleshooting

### `OSError: You are trying to access a gated repo`

Request access to `meta-llama/Llama-3.2-3B-Instruct`, accept the license, restart
the Colab runtime if necessary, and authenticate with `huggingface_hub.login()`.

### CUDA is unavailable

Enable a Colab GPU runtime and restart the notebook from the setup section.
CPU execution is technically possible but impractical for the full study.

### CUDA out of memory

Prefer a higher-memory runtime or reduce only a pilot batch/sample configuration.
If `--load_in_4bit` is used for final comparisons, apply the same quantization
policy to every compared model and document the change.

### Output already exists

The safeguard prevents accidental replacement of expensive results. Use a new
descriptive output name, or pass `--overwrite` only when replacement is
intentional.

### Results are written to an unexpected location

Check the Colab `%cd` output. Relative `results/...` paths are resolved from the
current working directory.

## Additional documentation

- See `ANALYSIS_PLAN.md` for the frozen estimands and inferential hierarchy.
- See `CAPSTONE_RUN_GUIDE.md` for the complete run catalog, including OpenAI and
  optional ablation commands.
- See the explanatory Markdown in `Capstone_Experiments.ipynb` for the rationale
  immediately beside each executed command and saved result.
