"""Dataset loading, prompts, parsing, and task metrics."""

import collections
import re
import string
from typing import Dict, List, Optional


SQUAD_SYSTEM = (
    "Answer using only the supplied context. Return only the minimal answer span "
    "copied from the context. Do not write a sentence. Do not explain. Do not add "
    "punctuation unless it is part of the answer."
)

GSM8K_SYSTEM = (
    "Solve the math problem carefully. At the end, write the final numeric answer "
    "in the form: FINAL: <answer>."
)


def load_examples(
    task: str,
    n_samples: int,
    seed: int,
    revision: Optional[str] = None,
    sample_offset: int = 0,
) -> List[Dict]:
    from datasets import load_dataset

    if task == "squad":
        dataset = load_dataset(
            "rajpurkar/squad", split="validation", revision=revision
        )
        dataset = dataset.shuffle(seed=seed)
        stop = min(sample_offset + n_samples, len(dataset))
        dataset = dataset.select(range(sample_offset, stop))
        if not len(dataset):
            raise ValueError("--sample_offset is beyond the end of SQuAD validation")
        return [
            {
                "id": row["id"],
                "task": "squad",
                "question": row["question"],
                "context": row["context"],
                "answers": row["answers"]["text"],
            }
            for row in dataset
        ]

    if task == "gsm8k":
        dataset = load_dataset(
            "openai/gsm8k", "main", split="test", revision=revision
        )
        dataset = dataset.shuffle(seed=seed)
        stop = min(sample_offset + n_samples, len(dataset))
        dataset = dataset.select(range(sample_offset, stop))
        if not len(dataset):
            raise ValueError("--sample_offset is beyond the end of GSM8K test")
        return [
            {
                "id": str(index),
                "task": "gsm8k",
                "question": row["question"],
                "reference": row["answer"],
            }
            for index, row in enumerate(dataset)
        ]

    raise ValueError(f"Unknown task: {task}")


def make_messages(
    example: Dict, question: str, mitigation: str = "none"
) -> List[Dict[str, str]]:
    if example["task"] == "squad":
        user = f"Context:\n{example['context']}\n\nQuestion:\n{question}"
        system = SQUAD_SYSTEM
    else:
        user = f"Problem:\n{question}"
        system = GSM8K_SYSTEM

    if mitigation == "self_correct":
        user = (
            "The user input may contain typographical errors. First infer the "
            "intended wording silently, then answer the intended question.\n\n" + user
        )
    elif mitigation != "none":
        raise ValueError(f"Unknown mitigation: {mitigation}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def squad_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def squad_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not truth_tokens:
        return float(prediction_tokens == truth_tokens)
    common = collections.Counter(prediction_tokens) & collections.Counter(truth_tokens)
    shared = sum(common.values())
    if shared == 0:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, references: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalize_answer(reference) in normalized_prediction for reference in references))


def best_squad_scores(prediction: str, references: List[str]):
    return (
        max(squad_exact_match(prediction, reference) for reference in references),
        max(squad_f1(prediction, reference) for reference in references),
        contains_gold(prediction, references),
    )


def extract_gsm8k_reference(answer: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer)
    return numbers[-1].replace(",", "") if numbers else None


def extract_gsm8k_prediction(text: str) -> Optional[str]:
    final = re.findall(r"FINAL:\s*([-+]?\d[\d,]*(?:\.\d+)?)", text, flags=re.I)
    if final:
        return final[-1].replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def numeric_equal(a: Optional[str], b: Optional[str], tolerance: float = 1e-9) -> float:
    if a is None or b is None:
        return 0.0
    try:
        return float(abs(float(a) - float(b)) <= tolerance)
    except ValueError:
        return float(a == b)


def score_prediction(example: Dict, prediction: str) -> Dict:
    if example["task"] == "squad":
        exact_match, f1, containment = best_squad_scores(
            prediction, example["answers"]
        )
        return {
            "em": exact_match,
            "f1": f1,
            "semantic_correct": containment,
            "accuracy": float("nan"),
        }

    reference = extract_gsm8k_reference(example["reference"])
    parsed_prediction = extract_gsm8k_prediction(prediction)
    return {
        "reference_final": reference,
        "prediction_final": parsed_prediction,
        "accuracy": numeric_equal(parsed_prediction, reference),
        "em": float("nan"),
        "f1": float("nan"),
        "semantic_correct": float("nan"),
    }


def serialized_reference(example: Dict) -> str:
    import json

    if example["task"] == "squad":
        return json.dumps(example["answers"], ensure_ascii=False)
    return example["reference"]
