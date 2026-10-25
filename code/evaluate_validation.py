"""Run every Stage 1 checkpoint through the independent validation tools."""

import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from common import DATA, load_bank, parse_answer, read_jsonl


ROOT = Path(__file__).resolve().parent


def _run(args, env):
    completed = subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _correctness(bank, references, choices):
    return [
        parse_answer(row["candidates"][choice]["text"]) == parse_answer(ref["answer"])
        for row, ref, choice in zip(bank, references, choices)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "validation")
    args = parser.parse_args()
    checkpoints_dir = args.checkpoint_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "4"
    checkpoints = sorted(checkpoints_dir.glob("*.npz"))
    if not checkpoints:
        raise ValueError("Run train.py first")
    records = []
    selections = {}
    for checkpoint in checkpoints:
        name = checkpoint.stem
        prediction = out / f"{name}-predictions.json"
        score = out / f"{name}-score.json"
        started = time.perf_counter()
        _run(["predict.py", "--method", "submission.py", "--checkpoint",
              str(checkpoint), "--split", "validation", "--output", str(prediction)], env)
        _run(["score_predictions.py", "--split", "validation", "--predictions",
              str(prediction), "--output", str(score)], env)
        wall = time.perf_counter() - started
        official = json.loads(score.read_text(encoding="utf-8"))
        choices = json.loads(prediction.read_text(encoding="utf-8"))["selected"]
        selections[name] = choices
        record = {
            "method": name, "correct": official["correct"], "n": official["n"],
            "accuracy": official["accuracy"], "fresh_process_predict_plus_score_seconds": wall,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "prediction_file": str(prediction.relative_to(ROOT)),
            "official_score_file": str(score.relative_to(ROOT)),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    bank = load_bank("validation")
    refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    if any(row["id"] != ref["id"] for row, ref in zip(bank, refs)):
        raise ValueError("Validation alignment mismatch")
    distinct = [len({answer for answer in (parse_answer(c["text"]) for c in row["candidates"])
                     if answer is not None}) for row in bank]
    hard = [value >= 2 for value in distinct]
    correctness = {name: _correctness(bank, refs, choice) for name, choice in selections.items()}
    for record in records:
        flags = correctness[record["method"]]
        record["disagreement_group"] = {
            "n": sum(hard), "correct": sum(ok for ok, group in zip(flags, hard) if group),
        }
        record["unanimous_or_unparsed_group"] = {
            "n": len(hard) - sum(hard),
            "correct": sum(ok for ok, group in zip(flags, hard) if not group),
        }

    point = max((r for r in records if r["method"].startswith("pointwise-")),
                key=lambda r: (r["correct"], -float(r["method"].split("-")[-1])))
    relative = max((r for r in records if r["method"].startswith("relative-")),
                   key=lambda r: (r["correct"], -float(r["method"].split("-")[-1])))
    pflags = correctness[point["method"]]
    rflags = correctness[relative["method"]]
    helped = [i for i, (p, r) in enumerate(zip(pflags, rflags)) if r and not p]
    hurt = [i for i, (p, r) in enumerate(zip(pflags, rflags)) if p and not r]
    examples = []
    for category, indices in (("relative_helped", helped[:5]), ("relative_hurt", hurt[:5])):
        for index in indices:
            row = bank[index]
            examples.append({
                "category": category, "validation_position": index,
                "question": row["question"],
                "candidate_final_answers": [parse_answer(c["text"]) for c in row["candidates"]],
                "reference_final_answer": parse_answer(refs[index]["answer"]),
                "pointwise_choice": selections[point["method"]][index],
                "relative_choice": selections[relative["method"]][index],
            })
    summary = {
        "split": "validation", "python": platform.python_version(),
        "numpy": np.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "thread_environment": {
            key: env[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "records": records,
        "comparison": {
            "best_pointwise": point["method"], "best_relative": relative["method"],
            "relative_helped_count": len(helped), "relative_hurt_count": len(hurt),
            "relative_helped_positions": helped, "relative_hurt_positions": hurt,
            "examples": examples,
        },
    }
    target = out / "summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(target.relative_to(ROOT)),
                      "best_pointwise": point["method"], "best_relative": relative["method"],
                      "relative_helped": len(helped), "relative_hurt": len(hurt)}))


if __name__ == "__main__":
    main()
