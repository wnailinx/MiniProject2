# DASE7506 MP2 - Submission

### **Should Candidate Solutions Be Judged Independently? A Controlled Ablation of Pointwise and Within-Question Scoring**

This repository packages the original pointwise ridge selector, its training and ablation code, complete supplied data, experiment records, final report, and official scorer output. The implementation was frozen from source commit `43357483f3d948c0d9a1775e2493a1200e528bf8` before its first public-test evaluation. The submitted checkpoint is `code/checkpoints-hard/pointwise-ridge-100.npz` (SHA-256 `609ca9452a9f37d5130e97de194075c755cb30d64a33c66e9803581b8dcef535`). No other checkpoint is required for inference.

**Official local test score:** `389/467 = 83.29764453961457%`, from `code/results/test-final/score.json`. The test prediction and independent scoring used separate fresh Python processes. Local wall time was 0.5745892 seconds on Windows 11 / AMD Ryzen 9 7940HX with four-thread limits. Reference Xeon runtime and an instructor audit-bank result have not been measured.

**Report:** [`REPORT.pdf`](REPORT.pdf) (editable source: [`code/REPORT.md`](code/REPORT.md)). The research question is whether within-question comparison helps select a correct candidate; the principal ablation adds 12 relative features to an otherwise matched pointwise ridge scorer. The report states both the negative ablation and the later public-test-score exposure.

## Repository layout

- `code/submission.py`, `code/features.py`, `code/common.py`: inference and feature extraction.
- `code/checkpoints-hard/pointwise-ridge-100.npz`: submitted inference asset (13,248 uncompressed bytes).
- `code/checkpoints/training.json`: record of fitting on all 4,755 training questions. The other all-train checkpoints are reproducible intermediates and are not included.
- `code/checkpoints-hard/training.json`, `pairwise-training.json`: records of fitting on the 2,114 answer-disagreement training questions. The other hard-train checkpoints are reproducible intermediates and are not included.
- `code/train.py`, `code/train_pairwise.py`, `code/derive_grouped.py`: train-only fitting and exploratory methods.
- `code/evaluate_validation.py`, `code/predict.py`, `code/score_predictions.py`, `code/check_data.py`: evaluation and checks. The supplied parser and scorer are unchanged.
- `code/results/validation*/`, `code/results/test-final/`: saved official scorer outputs, predictions and summary records.
- `code/data/`: complete course data, manifest, license and attribution.
- `code/EXPERIMENT_PLAN.md`, `code/METHOD_RECORD.md`, `code/REPORT.md`: pre-experiment plan, historical Stage 1 record, final report source.

Only the named `pointwise-ridge-100.npz` file is the checkpoint to link in the course submission form. The `training.json` files describe experiments; they are not inference checkpoints.

## Exact reproduction

Use Python 3.12. From `code/`, create a virtual environment and install the pinned NumPy dependency. In PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.venv\Scripts\python.exe check_data.py
```

Score the supplied frozen checkpoint without retraining:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
.venv\Scripts\python.exe predict.py --method submission.py --checkpoint checkpoints-hard/pointwise-ridge-100.npz --split test --output runs/test-predictions.json
.venv\Scripts\python.exe score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

The scorer should report `389/467` and `83.29764453961457` in its `accuracy` field. For validation, use `--split validation` and corresponding validation filenames; the frozen checkpoint scored `742/798` overall and `299/355` on answer-disagreement questions.

To reproduce training and the main ablation, run the following from `code/` after installation. These commands use only supplied training references; development scoring uses validation. Generated files go under `runs/` so the submitted checkpoint remains untouched.

```powershell
.venv\Scripts\python.exe train.py --output-dir runs/checkpoints-all
.venv\Scripts\python.exe train.py --hard-only --output-dir runs/checkpoints-hard
.venv\Scripts\python.exe derive_grouped.py --source runs/checkpoints-hard/pointwise-ridge-100.npz --output-dir runs/checkpoints-hard
.venv\Scripts\python.exe train_pairwise.py --output-dir runs/checkpoints-hard
.venv\Scripts\python.exe evaluate_validation.py --checkpoint-dir runs/checkpoints-hard --output-dir runs/validation-hard
```

The pointwise and relative models use the same split, labels, ridge solver and alpha grid. The main hard-training comparison is 742/798 versus 724/798. The scripts use deterministic fitting; no random seed applies to these models. `code/REPORT.md` and `code/results/test-final/EVALUATION_RECORD.md` give the design, detailed numbers, chronology, hardware and resource calculations.

## Submission and disclosure

The original checkpoint was chosen and frozen using validation before its first public-test score. Later experimental branches also produced aggregate public-test scores, which were visible before choosing this package for submission. The report discloses this chronology and does not claim an untouched test comparison among those branches. No public-test reference was used to fit this checkpoint. OpenAI Codex assistance in design, implementation and evaluation is disclosed in the report.
