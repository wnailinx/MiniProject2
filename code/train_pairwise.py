"""Fit a direct within-question pairwise ridge ranker on train references only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, read_jsonl
from features import feature_matrix
from train import candidate_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints-hard"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 1000.0, 10000.0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    labels = candidate_labels(bank, references).reshape(-1, 4)
    matrix = feature_matrix(bank, arrays).reshape(len(bank), 4, -1)
    center = matrix.reshape(-1, matrix.shape[-1]).mean(axis=0).astype(np.float32)
    scale = matrix.reshape(-1, matrix.shape[-1]).std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - center) / scale
    differences = []
    for row, marks in zip(normalized, labels):
        positives = np.flatnonzero(marks)
        negatives = np.flatnonzero(1 - marks)
        for positive in positives:
            for negative in negatives:
                differences.append(row[positive] - row[negative])
    pairs = np.asarray(differences, dtype=np.float32)
    gram = (pairs.T @ pairs).astype(np.float64)
    cross = pairs.sum(axis=0, dtype=np.float64)
    eye = np.eye(gram.shape[0], dtype=np.float64)
    checkpoints = []
    for alpha in args.alphas:
        weight = np.linalg.solve(gram + alpha * eye, cross).astype(np.float32)
        path = args.output_dir / f"pairwise-ridge-{alpha:g}.npz"
        np.savez(path, mode=np.asarray("pairwise"), mean=center, scale=scale,
                 weight=weight, intercept=np.asarray(0.0, dtype=np.float32),
                 ridge_alpha=np.asarray(alpha, dtype=np.float64))
        checkpoints.append({"path": str(path), "bytes": path.stat().st_size})
    record = {"training_split": "train", "number_positive_negative_pairs": len(pairs),
              "alphas": args.alphas, "checkpoints": checkpoints,
              "training_seconds": time.perf_counter() - tick}
    target = args.output_dir / "pairwise-training.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
