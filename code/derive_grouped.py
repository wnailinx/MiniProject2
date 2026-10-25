"""Derive predeclared answer-group comparison rules from a train-only pointwise model."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.source, allow_pickle=False) as source:
        if str(source["mode"].item()) != "pointwise":
            raise ValueError("Source must be a pointwise checkpoint")
        common = {key: source[key] for key in ("mean", "scale", "weight", "intercept", "ridge_alpha")}
    for strategy, bonuses in (("mean", (0.0, 0.05, 0.10)),
                              ("max", (0.05, 0.10, 0.20))):
        for bonus in bonuses:
            name = f"group-{strategy}-bonus-{bonus:.2f}.npz"
            target = args.output_dir / name
            np.savez(target, mode=np.asarray("grouped"),
                     strategy=np.asarray(strategy), bonus=np.asarray(bonus), **common)
            print(target)


if __name__ == "__main__":
    main()
