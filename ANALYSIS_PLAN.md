# Analysis plan

Use this plan for the new confirmatory runs. Results from earlier development
runs should be described as pilot evidence, not combined with the final sample.

## Research question

How much do controlled character-level typographical corruptions reduce language
model task performance, how does the loss change with the number of corrupted
words, and does robustness differ across model families?

The project measures robustness to a synthetic adversarial typo generator. It
does not estimate the prevalence or average effect of naturally occurring user
typos.

## Primary analysis

- Primary task: SQuAD validation.
- Primary outcome: paired clean-minus-corrupted token F1.
- Independent unit: `example_id`.
- Primary factors: severity (1, 2, 3) and model.
- Estimand: mean score drop at each severity under `location=any`, balanced
  within each example over requested typo types.
- Uncertainty: 95% paired bootstrap interval over examples.
- Main table: `<run>_primary_summary.csv`.
- Cross-model estimand: difference in paired score drop, model A minus model B.
  Positive values mean model A is less robust. Use
  `cross_model_gap_by_severity.csv`.

The primary conclusion is supported when degradation grows with severity and
the relevant confidence intervals exclude zero. Do not count corruption
replicates as independent observations.

## Controlled comparisons

The default matched-target design selects words that are eligible for every
requested typo type. Hard corruptions use two non-conflicting character
operations per selected word. Consequently, typo-type comparisons are not
confounded by different target words or operation counts.

Use one seed and the same sample offset for all compared models. Use the same
precision/quantization setting. Pin Hugging Face model and dataset revisions in
the final runs and retain every metadata JSON.

## Secondary analyses

- Typo type and location effects from `<run>_summary.csv`.
- A separate severity-1 standard-typo position experiment. Higher positional
  severities are not confirmatory because a fixed question third often lacks
  enough eligible words.
- Standard-versus-hard intensity ablation.
- Prompt-based self-correction mitigation, using its paired improvement table.
- GSM8K accuracy as task generalization.
- Three-replicate `any`-location run as a corruption-realization sensitivity
  check.

These analyses should emphasize effect sizes and confidence intervals. If many
individual cells are tested, label them exploratory rather than selecting only
the significant cells.

## Mechanistic/exploratory analyses

Token fragmentation and representation drift are correlational. Report Pearson
and Spearman correlations within each severity as well as pooled. Internal
Hugging Face representations are question-token means extracted from the exact
inference chat prompt. OpenAI embedding drift comes from a separate embedding
model and must not be described as the generator's hidden state.

PCA is the primary two-dimensional display. UMAP is exploratory and must not be
used as evidence of cluster separation without a quantitative test.

## Quality checks and exclusions

- Report generation error rate and severity completion rate for every run.
- Do not silently drop failed generations. Summaries report the number of usable
  examples and requested rows.
- Keep the pilot disjoint using `--sample_offset 200` for a 200-example core
  sample.
- Do not alter hypotheses or exclusion rules after viewing final results without
  labeling the change as exploratory.
