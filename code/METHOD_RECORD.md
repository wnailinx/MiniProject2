# MP2 Stage 1: method and experiment record

This is the historical pre-test experiment record. The final score and updated
interpretation are in `REPORT.md`. The curated submission retains the final
checkpoint and experiment JSON records; alternative `.npz` files can be
regenerated with the commands in the repository README.

Status: validation study completed on 20 October 2026. This is a working record, not the final 10-page report. No final test score, public repository, or leaderboard submission has been produced.

## Question and result

The preregistered question in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) is whether comparing the four candidates improves selection over scoring each candidate independently. The initial hypothesis predicted a gain from within-question relative features. It was **not supported** by the validation experiments below. The current recommended checkpoint is the hard-train pointwise ridge scorer, `checkpoints-hard/pointwise-ridge-100.npz`, subject to the user's decision before freezing and final testing.

The official metric checks final parsed numeric answers, so every training candidate whose final answer matches its reference is a positive label. Several candidates in a group may be positive. No step-correctness labels are available.

## Methods and fair comparison

1. `fixed`: always select candidate 0.
2. `agreement`: select a candidate from the largest group of equal parsed final answers, with first-index tie breaking.
3. `pointwise`: a ridge least-squares scorer on each candidate's supplied 1024-dimensional whole-trace embedding and 12 candidate-local text/answer features. The highest candidate score wins.
4. `relative`: the **same labels, ridge learner, training groups, local features and alpha grid** as pointwise, plus 12 within-question features: answer-group size and position, relative length and numeral count, and embedding similarities/distance within the four candidates. Removing these 12 features is the key ablation.

Training means and standard deviations are computed on train only and saved with the weights. The ridge penalties tried were 100, 1,000 and 10,000. `train.py` does not read validation or test references. All validation numbers below come from the unmodified `predict.py` and `score_predictions.py`.

The training and selection algorithms use no random sampling or initialization, so a random seed is not applicable. Candidate order, tie breaking and the alpha grid are fixed in code.

After the initial result, we performed exploratory follow-ups (not preregistered): fit the same pointwise and relative models on the 2,114 training questions with at least two distinct parsed candidate answers; combine candidate scores by final-answer group using six fixed aggregation settings; and fit a pairwise ridge ranker from positive-negative candidate differences in train. The pointwise/relative ablation remains controlled within each training population. The aggregation rules share the strongest train-only pointwise weights. Exploratory selection on validation can overfit and must be described as such later.

## Validation results

The bank has 798 questions. On 443 questions there is at most one distinct parsed candidate answer; the selected pointwise model and agreement baseline chose correctly on all 443, while the relative model chose correctly on 441. The other 355 questions carry the informative answer-disagreement comparison. The table reports official accuracy on all 798 and correct selections on those 355.

| Method | Training groups | Correct / 798 | Accuracy | Correct / 355 disagreement | Fresh prediction + scoring on this PC |
|---|---:|---:|---:|---:|---:|
| Fixed index 0 | none | 624 | 78.20% | 195 | 0.46 s |
| Answer agreement | none | 688 | 86.22% | 245 | 0.48 s |
| Pointwise, alpha 1000 | all 4755 | 734 | 91.98% | 291 | 0.58 s |
| Relative, alpha 1000 | all 4755 | 714 | 89.47% | 271 | 0.70 s |
| Pointwise, alpha 100 | 2114 disagreement groups | **742** | **92.98%** | **299** | 0.67 s |
| Relative, alpha 100 | 2114 disagreement groups | 724 | 90.73% | 283 | 0.83 s |
| Best answer-group aggregation, max + 0.10 per extra vote | same pointwise weights | 739 | 92.61% | 296 | 0.66 s |
| Best pairwise ridge, alpha 1000 | train positive-negative pairs | 739 | 92.61% | 296 | 0.67 s |

The all-train alpha-100 controlled ablation is 733/798 pointwise versus 720/798 relative; alpha 1000, shown above, is 734 versus 714. The disagreement-train alpha-100 controlled ablation is 742 versus 724. Full grids, timing, predictions and independent score JSON files are saved in `results/validation/` and `results/validation-hard/`; training settings are saved in `checkpoints/training.json`, `checkpoints-hard/training.json` and `checkpoints-hard/pairwise-training.json`.

For the selected hard-train alpha-100 pair, the relative method fixes 13 pointwise errors but creates 31 new ones. Thus the 18-question net loss is an observed paired difference, not just an aggregate score comparison. For example, validation position 90 has candidate answers `[48, 48, 32, 44]` with reference `48`: relative chooses a correct `48` while pointwise chooses `32`. At position 77, candidate answers are `[1120, 1120, 2240, 1120]` with reference `2240`: pointwise finds the correct singleton while relative chooses the incorrect majority. These positions identify validation examples for analysis only; the predictors never use positions or IDs.

This evidence does not establish that all candidate-comparison methods are worse. It shows that the tested relative features, aggregation rules and pairwise ridge objective did not beat the chosen pointwise scorer on this validation bank. The disagreement subset result is 299/355 = 84.23%, appreciably below the all-validation 92.98%; the public test bank is specified to contain answer disagreement, so the all-validation figure should not be treated as an expected test score.

## Reproduction and resources

Use Python 3.12 and `requirements.txt` (`numpy==2.5.3`). The local run used Python 3.12.14 on Windows 11, AMD Ryzen 9 7940HX with Radeon Graphics (32 logical processors). `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were each set to 4. In this Codex environment the bundled Python 3.12 venv needed pip bootstrap via `python -m pip --python .venv install -r requirements.txt`; a standard Python 3.12 installation can use the README's venv and pip commands. No GPU or additional Python package was used.

From `code/` in PowerShell, after installing the pinned dependency:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.venv\Scripts\python.exe check_data.py
.venv\Scripts\python.exe train.py --output-dir checkpoints
.venv\Scripts\python.exe train.py --hard-only --output-dir checkpoints-hard
.venv\Scripts\python.exe derive_grouped.py --source checkpoints-hard\pointwise-ridge-100.npz --output-dir checkpoints-hard
.venv\Scripts\python.exe train_pairwise.py --output-dir checkpoints-hard
.venv\Scripts\python.exe evaluate_validation.py
.venv\Scripts\python.exe evaluate_validation.py --checkpoint-dir checkpoints-hard --output-dir results\validation-hard
```

The evaluator launches a **new prediction process and then a separate scoring process**, timing both wall clocks together. The official runner's internal `prediction_seconds` omits some startup overhead. The recommended model's local validation measurement was about 0.67 s, including 798-input loading and the independent scorer; the 30 s eligibility limit is defined on the different reference Xeon Platinum 8457C for the complete test bank and cannot be certified from this PC's result. Its sole required inference checkpoint is 13,972 bytes on disk and 13,248 bytes of uncompressed NPZ members, below the 16 MiB limit. SHA-256: `609ca9452a9f37d5130e97de194075c755cb30d64a33c66e9803581b8dcef535`. Other experiment checkpoints are not needed to run this recommended method.

The method was also called on a seven-question slice and an empty bank without references; slice predictions matched the corresponding full-bank predictions and all outputs were Python integers. This checks that `predict()` uses its passed inputs without hard-coded sample counts. It is not a substitute for a separate instructor audit bank.

## Stage boundary and disclosure

Only training and validation references were used for model development. `check_data.py` verified the supplied package; there was no final test prediction or score. Do not use test results to adjust this checkpoint after the user decides to freeze it. The full formal report, public repository and submission are deferred to Stage 2.

Substantive AI assistance: Codex designed and implemented the features, training/evaluation scripts, experiments and this record. The student should inspect and understand the implementation before final submission and disclose this assistance in the eventual report. The supplied GSM8K/Qwen attribution and license files remain unchanged.
