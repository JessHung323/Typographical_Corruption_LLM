"""Deterministic typo generation and target-word selection."""

import collections
import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple


QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qase", "e": "wsdr", "r": "edft", "t": "rfgy",
    "y": "tghu", "u": "yhji", "i": "ujko", "o": "iklp", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm",
    "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv",
    "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of",
    "to", "in", "on", "at", "for", "from", "by", "with", "as", "is",
    "are", "was", "were", "be", "been", "being", "what", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "would", "should", "will", "may", "might", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "we",
    "you", "your", "i", "he", "she",
}


def word_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return alphabetic word spans as ``(start, end, token)``."""
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-z]+", text)]


def compute_idf(texts: Sequence[str]) -> Dict[str, float]:
    """Compute smoothed document-frequency weights over sampled questions."""
    document_frequency = collections.Counter()
    for text in texts:
        terms = {
            word.lower() for _, _, word in word_spans(text)
            if word.lower() not in STOPWORDS
        }
        document_frequency.update(terms)
    n_documents = len(texts)
    return {
        term: math.log((1 + n_documents) / (1 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }


def eligible_char_positions(word: str, typo_type: str) -> List[int]:
    if typo_type == "keyboard":
        return [i for i, char in enumerate(word) if char.lower() in QWERTY_NEIGHBORS]
    if typo_type == "deletion":
        return list(range(len(word))) if len(word) >= 2 else []
    if typo_type == "transposition":
        return [
            i for i in range(len(word) - 1)
            if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]
        ]
    raise ValueError(f"Unknown typo_type: {typo_type}")


def supports_operation_count(word: str, typo_type: str, count: int) -> bool:
    """Whether a word supports exactly ``count`` non-conflicting operations."""
    positions = eligible_char_positions(word, typo_type)
    if count <= 1:
        return bool(positions)
    if len(word) < 4:
        return False
    if typo_type in {"keyboard", "deletion"}:
        return len(positions) >= count
    if typo_type == "transposition":
        return any(
            abs(first - second) > 1
            for first in positions
            for second in positions
        )
    return False


def choose_word_index(
    text: str,
    location: str,
    typo_type: str,
    rng: random.Random,
    excluded_words: Optional[set] = None,
    keyword_scores: Optional[Dict[str, float]] = None,
    target_typo_types: Optional[Sequence[str]] = None,
    required_operations: int = 1,
) -> Optional[int]:
    """Choose an eligible target without spilling outside the requested region."""
    spans = word_spans(text)
    excluded = excluded_words or set()
    eligibility_types = list(target_typo_types) if target_typo_types else [typo_type]
    eligible = [
        i for i, (_, _, word) in enumerate(spans)
        if all(
            supports_operation_count(word, candidate_type, required_operations)
            for candidate_type in eligibility_types
        )
        and i not in excluded
    ]
    if not eligible:
        return None

    if location == "beginning":
        cutoff = max(1, math.ceil(len(spans) / 3))
        pool = [i for i in eligible if i < cutoff]
    elif location == "middle":
        lower = len(spans) // 3
        upper = max(lower + 1, math.ceil(2 * len(spans) / 3))
        pool = [i for i in eligible if lower <= i < upper]
    elif location == "end":
        cutoff = 2 * len(spans) // 3
        pool = [i for i in eligible if i >= cutoff]
    elif location == "keyword":
        content = [i for i in eligible if spans[i][2].lower() not in STOPWORDS]
        if not content:
            return None
        if keyword_scores:
            maximum = max(
                keyword_scores.get(spans[i][2].lower(), 0.0) for i in content
            )
            pool = [
                i for i in content
                if keyword_scores.get(spans[i][2].lower(), 0.0) == maximum
            ]
        else:
            pool = content
        longest = max(len(spans[i][2]) for i in pool)
        pool = [i for i in pool if len(spans[i][2]) == longest]
    elif location == "any":
        pool = eligible
    else:
        raise ValueError(f"Unknown location: {location}")

    return rng.choice(pool) if pool else None


def mutate_word(
    word: str,
    typo_type: str,
    rng: random.Random,
    difficulty: str = "standard",
    required_operations: Optional[int] = None,
) -> Tuple[Optional[str], Dict]:
    """Apply one standard or up to two hard character operations."""
    positions = eligible_char_positions(word, typo_type)
    if not positions:
        return None, {}
    ordered_positions = list(positions)
    rng.shuffle(ordered_positions)
    operation_count = 2 if difficulty == "hard" and len(word) >= 4 else 1

    if typo_type == "keyboard":
        chars = list(word)
        chosen = ordered_positions[: min(operation_count, len(ordered_positions))]
        for position in chosen:
            old = chars[position]
            replacement = rng.choice(QWERTY_NEIGHBORS[old.lower()])
            chars[position] = replacement.upper() if old.isupper() else replacement
        return "".join(chars), {
            "char_indices": sorted(chosen), "operation_count": len(chosen)
        }

    if typo_type == "deletion":
        chosen = ordered_positions[: min(operation_count, len(ordered_positions))]
        chosen_set = set(chosen)
        return "".join(
            char for position, char in enumerate(word) if position not in chosen_set
        ), {
            "char_indices": sorted(chosen),
            "deleted_chars": [word[position] for position in sorted(chosen)],
            "operation_count": len(chosen),
        }

    if typo_type == "transposition":
        chars = list(word)
        valid_pairs = [
            (first, second)
            for pair_index, first in enumerate(ordered_positions)
            for second in ordered_positions[pair_index + 1:]
            if abs(first - second) > 1
        ]
        needs_pair = operation_count == 2 or required_operations == 2
        if needs_pair and valid_pairs:
            first, second = rng.choice(valid_pairs)
            # When standard and hard are run together, the standard swap is a
            # strict subset of the hard corruption on the same word.
            chosen = [first, second] if operation_count == 2 else [first]
        else:
            chosen = [ordered_positions[0]]
        for position in chosen:
            chars[position], chars[position + 1] = chars[position + 1], chars[position]
        return "".join(chars), {
            "char_indices": sorted(chosen), "operation_count": len(chosen)
        }

    return None, {}


def introduce_typos(
    text: str,
    typo_type: str,
    location: str,
    severity: int,
    seed: int,
    difficulty: str = "hard",
    keyword_scores: Optional[Dict[str, float]] = None,
    target_typo_types: Optional[Sequence[str]] = None,
    required_operations: Optional[int] = None,
) -> Tuple[str, List[Dict]]:
    """Corrupt ``severity`` distinct words using a deterministic edit sequence."""
    if severity < 1:
        raise ValueError("severity must be at least 1")

    selection_rng = random.Random(seed)
    if required_operations is None:
        required_operations = 2 if difficulty == "hard" else 1
    corrupted = text
    edits = []
    targeted_words = set()

    for edit_number in range(severity):
        spans = word_spans(corrupted)
        index = choose_word_index(
            corrupted,
            location,
            typo_type,
            selection_rng,
            excluded_words=targeted_words,
            keyword_scores=keyword_scores,
            target_typo_types=target_typo_types,
            required_operations=required_operations,
        )
        if index is None:
            break

        start, end, old_word = spans[index]
        mutation_rng = random.Random(seed * 1_000_003 + edit_number * 10_007)
        new_word, operation = mutate_word(
            old_word,
            typo_type,
            mutation_rng,
            difficulty=difficulty,
            required_operations=required_operations,
        )
        if new_word is None or new_word == old_word:
            continue

        corrupted = corrupted[:start] + new_word + corrupted[end:]
        targeted_words.add(index)
        edits.append({
            "edit_num": edit_number + 1,
            "old": old_word,
            "new": new_word,
            "location": location,
            "typo_type": typo_type,
            "difficulty": difficulty,
            "word_index": index,
            **operation,
        })

    return corrupted, edits
