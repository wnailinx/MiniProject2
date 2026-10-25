"""Fit train-only ridge candidate scorers and write small inference checkpoints.

Run from code/: python train.py --output-dir checkpoints
This program never loads validation or test references.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import feature_matrix


def candidate_labels(bank, references):
    if len(bank) != len(references):
        raise ValueError("Training reference length mismatch")
    labels = []
    for row, reference in zip(bank, references):
        if row["id"] != reference["id"] or row["question"] != reference["question"]:
            raise ValueError("Training reference alignment mismatch")
        target = parse_answer(reference["answer"])
        labels.extend(
            int(target is not None and parse_answer(c["text"]) == target)
            for c in row["candidates"]
        )
    return np.asarray(labels, dtype=np.float64)


def save_baseline(output_dir, mode):
    target = output_dir / f"{mode}.npz"
    np.savez(target, mode=np.asarray(mode))
    return target


def fit_variant(bank, arrays, labels, mode, alphas, output_dir):
    started = time.perf_counter()
    matrix = feature_matrix(bank, arrays, include_relative=(mode == "relative"))
    center = matrix.mean(axis=0).astype(np.float32)
    scale = matrix.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - center) / scale
    gram = (normalized.T @ normalized).astype(np.float64)
    intercept = float(labels.mean())
    cross = (normalized.T @ (labels - intercept).astype(np.float32)).astype(np.float64)
    eye = np.eye(gram.shape[0], dtype=np.float64)
    results = []
    for alpha in alphas:
        weight = np.linalg.solve(gram + alpha * eye, cross).astype(np.float32)
        target = output_dir / f"{mode}-ridge-{alpha:g}.npz"
        np.savez(
            target, mode=np.asarray(mode), mean=center, scale=scale,
            weight=weight, intercept=np.asarray(intercept, dtype=np.float32),
            ridge_alpha=np.asarray(alpha, dtype=np.float64),
        )
        results.append({"checkpoint": str(target), "alpha": alpha, "bytes": target.stat().st_size})
    return {"mode": mode, "dimensions": matrix.shape[1],
            "elapsed_seconds": time.perf_counter() - started, "models": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--hard-only", action="store_true",
                        help="Fit on train groups with at least two distinct parsed answers")
    args = parser.parse_args()
    if any(value <= 0 for value in args.alphas):
        raise ValueError("Ridge penalties must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    bank, arrays = load_inputs("train")
    references = read_jsonl(DATA / "gsm8k_train.jsonl")
    if args.hard_only:
        selected = [i for i, row in enumerate(bank) if len({
            answer for answer in (parse_answer(c["text"]) for c in row["candidates"])
            if answer is not None
        }) >= 2]
        bank = [bank[i] for i in selected]
        references = [references[i] for i in selected]
        arrays = {key: value[selected] for key, value in arrays.items()}
    labels = candidate_labels(bank, references)
    records = {
        "training_split": "train", "hard_only": args.hard_only,
        "n_questions": len(bank),
        "n_positive_candidates": int(labels.sum()),
        "numpy_version": np.__version__, "alphas": args.alphas,
        "baselines": [str(save_baseline(args.output_dir, mode))
                      for mode in ("fixed", "agreement")],
        "variants": [],
    }
    for mode in ("pointwise", "relative"):
        record = fit_variant(bank, arrays, labels, mode, args.alphas, args.output_dir)
        records["variants"].append(record)
        print(json.dumps(record), flush=True)
    records["total_elapsed_seconds"] = time.perf_counter() - started
    record_path = args.output_dir / "training.json"
    record_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training_record": str(record_path),
                      "total_elapsed_seconds": records["total_elapsed_seconds"]}))


if __name__ == "__main__":
    main()
