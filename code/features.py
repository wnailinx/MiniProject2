"""Input-only candidate features for the MP2 selection experiments.

No function here opens references, uses IDs, or depends on a bank's size.
"""

from fractions import Fraction
import re

import numpy as np

from common import parse_answer


WORD = re.compile(r"[A-Za-z]+")
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
LOCAL_NAMES = (
    "log_chars", "log_words", "log_numbers", "log_equals", "log_dollars",
    "has_parsed_answer", "has_boxed", "has_numeric_tag", "negative_answer",
    "log_answer_magnitude", "fractional_answer", "question_word_overlap",
)
RELATIVE_NAMES = (
    "answer_group_size", "answer_group_share", "unique_answer", "answer_is_min",
    "answer_is_max", "answer_log_distance_to_median", "relative_log_chars",
    "relative_log_numbers", "mean_embedding_cosine", "max_embedding_cosine",
    "min_embedding_cosine", "embedding_centroid_distance",
)


def _answer_value(parsed):
    if parsed is None:
        return None
    try:
        value = float(Fraction(parsed))
    except (ValueError, OverflowError):
        return None
    return float(np.clip(value, -1e12, 1e12)) if np.isfinite(value) else None


def feature_matrix(bank, arrays, include_relative=False):
    """Return [4*N, 1024 + local (+ relative)] float32 features."""
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    n = len(bank)
    if embeddings.shape != (n, 4, 1024):
        raise ValueError("Expected one 1024-vector per candidate")
    if n == 0:
        width = 1024 + len(LOCAL_NAMES) + (len(RELATIVE_NAMES) if include_relative else 0)
        return np.empty((0, width), dtype=np.float32)

    local = np.zeros((n, 4, len(LOCAL_NAMES)), dtype=np.float32)
    answers = []
    values = np.empty((n, 4), dtype=np.float64)
    values.fill(np.nan)
    for k, row in enumerate(bank):
        question_words = set(w.lower() for w in WORD.findall(row["question"]) if len(w) > 2)
        row_answers = []
        for j, candidate in enumerate(row["candidates"]):
            content = candidate["text"]
            parsed = parse_answer(content)
            row_answers.append(parsed)
            numeric_value = _answer_value(parsed)
            if numeric_value is not None:
                values[k, j] = numeric_value
            words = WORD.findall(content)
            overlap = len(question_words.intersection(w.lower() for w in words))
            local[k, j] = (
                np.log1p(len(content)), np.log1p(len(words)),
                np.log1p(len(NUMBER.findall(content))), np.log1p(content.count("=")),
                np.log1p(content.count("$")), float(parsed is not None),
                float("\\boxed{" in content), float("<numeric answer>" in content),
                float(numeric_value is not None and numeric_value < 0),
                np.log1p(abs(numeric_value)) if numeric_value is not None else 0.0,
                float(numeric_value is not None and not numeric_value.is_integer()),
                np.log1p(overlap),
            )
        answers.append(row_answers)

    parts = [embeddings, local]
    if include_relative:
        relative = np.zeros((n, 4, len(RELATIVE_NAMES)), dtype=np.float32)
        normalized = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=2, keepdims=True), 1e-8
        )
        cosine = np.matmul(normalized, normalized.transpose(0, 2, 1))
        for k in range(n):
            valid = values[k, np.isfinite(values[k])]
            median = float(np.median(valid)) if len(valid) else 0.0
            for j in range(4):
                answer = answers[k][j]
                group_size = sum(other == answer for other in answers[k]) if answer is not None else 0
                other_cos = np.delete(cosine[k, j], j)
                val = values[k, j]
                relative[k, j] = (
                    group_size, group_size / 4.0, float(group_size == 1),
                    float(np.isfinite(val) and len(valid) and val == valid.min()),
                    float(np.isfinite(val) and len(valid) and val == valid.max()),
                    abs(np.sign(val) * np.log1p(abs(val)) - np.sign(median) * np.log1p(abs(median)))
                    if np.isfinite(val) else 0.0,
                    local[k, j, 0] - local[k, :, 0].mean(),
                    local[k, j, 2] - local[k, :, 2].mean(),
                    other_cos.mean(), other_cos.max(), other_cos.min(),
                    np.linalg.norm(normalized[k, j] - normalized[k].mean(axis=0)),
                )
        parts.append(relative)
    result = np.concatenate(parts, axis=2).reshape(n * 4, -1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite input feature")
    return result


def agreement_choices(bank):
    """Select one candidate from the largest parsed-answer group; ties use first index."""
    selections = []
    for row in bank:
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = [answers.count(answer) if answer is not None else 0 for answer in answers]
        selections.append(int(np.argmax(counts)))
    return selections
