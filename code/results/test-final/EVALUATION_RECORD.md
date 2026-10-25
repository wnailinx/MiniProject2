# Frozen-model final test evaluation

Historical record written immediately after the first test run. The final
submission report and instructions are now `code/REPORT.md`, `REPORT.pdf`, and
the repository `README.md`; the "Still required" section below describes what
remained at the time of the test.

Date: 20 October 2026 (Asia/Shanghai). This is the first final-test scoring of the recommended Stage 1 checkpoint. The test result was not used for fitting, tuning or selecting a different method. No public submission was made.

## Frozen method and environment

- Method: submission.py pointwise ridge scorer (1024 supplied embedding features plus 12 candidate-local features), trained on the 2,114 training groups with at least two distinct parsed candidate answers; ridge alpha 100.
- Checkpoint: code/checkpoints-hard/pointwise-ridge-100.npz.
- Checkpoint SHA-256: 609ca9452a9f37d5130e97de194075c755cb30d64a33c66e9803581b8dcef535.
- Checkpoint size: 13,972 bytes on disk; 13,248 uncompressed NPZ-member bytes. No other inference asset is required.
- Python 3.12.14; NumPy 2.5.3; Windows 11; AMD Ryzen 9 7940HX (32 logical processors). OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4. No GPU was used.
- Protocol: dase7506-mp2-open-hard-verifier-v5. Official predictor and independent scorer were unmodified.
- Source SHA-256: submission.py E91B0A92C7BE4F49B9121E5A7A95B4CC668A496C782395943BCE86F1C90F7E54; features.py 98AFA553C2076324F4F36BBC9D83532A6C3CF8DDCA03769D693D98D58DF905C2.

## Exact evaluation

From code/ in Windows PowerShell, with the pinned environment active:

    $env:OMP_NUM_THREADS = '4'
    $env:OPENBLAS_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    python predict.py --method submission.py --checkpoint checkpoints-hard/pointwise-ridge-100.npz --split test --output results/test-final/predictions.json
    python score_predictions.py --split test --predictions results/test-final/predictions.json --output results/test-final/score.json

The two Python commands were run sequentially as separate fresh processes, with one outer PowerShell Stopwatch covering both. The measured interval was 2026-10-20T06:37:20.8614567Z to 2026-10-20T06:37:21.4438256Z; Stopwatch elapsed time was 0.5745892 s. This includes startup, imports, input and asset loading, prediction and independent scoring, plus ordinary process overhead. The predictor's internal prediction_seconds was 0.1478680 s and the scorer's internal scoring_seconds was 0.0200945 s; these narrower fields are **not** the elapsed-time figure used for the project resource comparison. The 30 s eligibility limit is defined on the reference Xeon Platinum 8457C and cannot be certified from this AMD PC's timing alone.

## Official result and integrity checks

- Official test score: **389 correct / 467 = 83.29764453961457% accuracy**.
- Official score JSON: code/results/test-final/score.json, SHA-256 E1F5EEDAE01D7EBFEC166CC836404B517C27C0FB9484AA422B93719A10190F75.
- Official predictions JSON: code/results/test-final/predictions.json, SHA-256 B3EFB583866C2957D82A6C55FE6D28C8EC45DC49DC2C01EC4829E3A4071BD836.
- Predictions contain exactly 467 distinct question IDs and 467 Python-integer selections in 0..3; protocol and split are correct. The independent scorer accepted them.
- The more comparable pre-test answer-disagreement validation subgroup was 299/355 = 84.22535211267606%; the test result is about 0.93 percentage points lower. The full validation score (742/798 = 92.98245614035088%) includes easier groups and should not be presented as a test forecast.

In this evaluation, test references were used for the official independent scoring process, not for fitting or selecting the method. We did not inspect per-question test failures or adapt the model after observing this score.

## Still required for coursework submission

Update the report with this final score and the validation/test distribution caveat; identify the submitted code version and matching checkpoint; provide public code and checkpoint/configuration links, the official score JSON, training and experiment records, and AI/reuse disclosure; then create the GitHub issue prepared by the MP2 submission website before the deadline. The report currently in output/pdf/ is a validation-stage draft and has not yet been updated with this result.
