"""CLI entry point for the LLM typo-robustness experiment.

The implementation lives in the ``capstone`` package. Existing commands remain
unchanged, for example:

python capstone_robustness.py \
    --task squad \
    --backend hf \
    --model Qwen/Qwen3-4B \
    --n_samples 200 \
    --typo_types keyboard deletion transposition \
    --locations beginning middle end keyword \
    --severities 1 2 3 \
    --typo_difficulty hard \
    --output qwen_squad_hard.csv
"""

from capstone.config import parse_args
from capstone.experiment import run_experiment


if __name__ == "__main__":
    run_experiment(parse_args())
