"""Fast, model-free validation checks for the capstone experiment code."""

import math

from capstone.tasks import numeric_equal, squad_exact_match, squad_f1
from capstone.typos import compute_idf, introduce_typos, word_spans


TEXTS = [
    "Which internationally recognized scientist discovered penicillin during laboratory research?",
    "In what year did the university establish its influential computer science department?",
    "How many students completed the difficult examination before the semester ended?",
]


def check_nested_severity() -> None:
    for text in TEXTS:
        for typo_type in ("keyboard", "deletion", "transposition"):
            for location in ("beginning", "middle", "end", "keyword", "any"):
                levels = [
                    introduce_typos(text, typo_type, location, severity, 12345, "hard")[1]
                    for severity in (1, 2, 3)
                ]
                assert levels[1][: len(levels[0])] == levels[0]
                assert levels[2][: len(levels[1])] == levels[1]


def check_location_fidelity() -> None:
    text = TEXTS[0]
    n_words = len(word_spans(text))
    allowed = {
        "beginning": set(range(0, max(1, math.ceil(n_words / 3)))),
        "middle": set(range(n_words // 3, max(n_words // 3 + 1, math.ceil(2 * n_words / 3)))),
        "end": set(range(2 * n_words // 3, n_words)),
    }
    for location, valid_indices in allowed.items():
        _, edits = introduce_typos(text, "keyboard", location, 20, 99, "hard")
        assert edits
        assert {edit["word_index"] for edit in edits}.issubset(valid_indices)


def check_difficulty_isolation() -> None:
    for typo_type in ("keyboard", "deletion", "transposition"):
        kwargs = {
            "target_typo_types": ["keyboard", "deletion", "transposition"],
            "required_operations": 2,
        }
        _, standard = introduce_typos(
            TEXTS[0], typo_type, "any", 3, 77, "standard", **kwargs
        )
        _, hard = introduce_typos(
            TEXTS[0], typo_type, "any", 3, 77, "hard", **kwargs
        )
        assert [edit["word_index"] for edit in standard] == [
            edit["word_index"] for edit in hard
        ]
        assert sum(edit["operation_count"] for edit in hard) >= sum(
            edit["operation_count"] for edit in standard
        )
        for standard_edit, hard_edit in zip(standard, hard):
            assert set(standard_edit["char_indices"]).issubset(
                set(hard_edit["char_indices"])
            )


def check_matched_typo_targets_and_operations() -> None:
    typo_types = ["keyboard", "deletion", "transposition"]
    edits_by_type = {}
    for typo_type in typo_types:
        _, edits = introduce_typos(
            TEXTS[0], typo_type, "any", 3, 991, "hard",
            target_typo_types=typo_types,
            required_operations=2,
        )
        edits_by_type[typo_type] = edits
        assert all(edit["operation_count"] == 2 for edit in edits)
    target_sequences = {
        tuple(edit["word_index"] for edit in edits)
        for edits in edits_by_type.values()
    }
    assert len(target_sequences) == 1


def check_metrics() -> None:
    assert squad_exact_match("The Pacific Ocean", "Pacific Ocean") == 1.0
    assert squad_f1("Pacific", "Pacific Ocean") == 2 / 3
    assert numeric_equal("1250.0", "1250") == 1.0
    assert numeric_equal("12", "13") == 0.0


def check_idf_keyword_targeting() -> None:
    corpus = ["common alpha", "common beta", "common rareword"]
    scores = compute_idf(corpus)
    _, edits = introduce_typos(
        corpus[-1], "keyboard", "keyword", 1, 11, "hard",
        keyword_scores=scores,
    )
    assert edits[0]["old"] == "rareword"


def main() -> None:
    check_nested_severity()
    check_location_fidelity()
    check_difficulty_isolation()
    check_matched_typo_targets_and_operations()
    check_metrics()
    check_idf_keyword_targeting()
    print("All model-free capstone validation checks passed.")


if __name__ == "__main__":
    main()
