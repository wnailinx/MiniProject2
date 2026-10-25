"""MP2 candidate selector. Checkpoints are trained on the supplied train split only."""

from pathlib import Path

import numpy as np

from common import parse_answer
from features import agreement_choices, feature_matrix


def _grouped_choices(bank, scores, strategy, bonus):
    """Choose an answer group, then its highest-scoring candidate."""
    selections = []
    for row, row_scores in zip(bank, scores):
        groups = {}
        for j, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            key = answer if answer is not None else f"unparsed-{j}"
            groups.setdefault(key, []).append(j)
        best_group_score = -np.inf
        best_choice = 0
        for key, members in groups.items():
            if key.startswith("unparsed-") and len(groups) > 1:
                continue
            values = row_scores[members]
            aggregate = (float(values.mean()) if strategy == "mean" else float(values.max()))
            aggregate += bonus * (len(members) - 1)
            if aggregate > best_group_score:
                best_group_score = aggregate
                best_choice = members[int(np.argmax(values))]
        selections.append(int(best_choice))
    return selections


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    if checkpoint is None:
        raise ValueError("Provide a checkpoint from train.py, including for baselines")
    if not bank:
        return []
    with np.load(Path(checkpoint), allow_pickle=False) as bundle:
        mode = str(bundle["mode"].item())
        if mode == "fixed":
            return [0 for _ in bank]
        if mode == "agreement":
            return agreement_choices(bank)
        if mode not in ("pointwise", "relative", "grouped", "pairwise"):
            raise ValueError(f"Unknown checkpoint mode: {mode}")
        matrix = feature_matrix(bank, arrays, include_relative=(mode == "relative"))
        mean = bundle["mean"]
        scale = bundle["scale"]
        weight = bundle["weight"]
        intercept = float(bundle["intercept"])
        if matrix.shape[1] != len(weight):
            raise ValueError("Checkpoint feature shape mismatch")
        scores = ((matrix - mean) / scale) @ weight + intercept
        scores = scores.reshape(len(bank), 4)
        if mode == "grouped":
            strategy = str(bundle["strategy"].item())
            if strategy not in ("mean", "max"):
                raise ValueError("Unknown answer-group strategy")
            return _grouped_choices(bank, scores, strategy, float(bundle["bonus"]))
        return [int(index) for index in np.argmax(scores, axis=1)]
