# Stage 1 experiment plan (written before model experiments)

Research question: Does comparing the four candidates for one question improve selection over scoring each candidate independently?

## Hypotheses

- H1: On validation, a learned scorer using relative within-question information will outperform the same learner and training labels using only candidate-local features. The main comparison is selection accuracy on all 798 validation questions.
- H2: The gain, if any, will be concentrated among questions with at least two distinct parsed final answers. Report this subgroup separately, along with the easier unanimous-answer subgroup. These are descriptions of the supplied validation bank, not selection criteria for training.
- H3: Answer agreement alone may be a useful baseline but can fail when several candidates repeat the same mistake. Compare it with a fixed-index baseline and the learned scorers.

## Protocol fixed in advance

1. Run the supplied data checker. Train only on `train` inputs and references. Use `validation` references only for comparing and selecting methods. Do not inspect `test` references or run final test scoring in Stage 1.
2. Label each training candidate by equality of its final parsed numeric answer to the reference, following the supplied `common.parse_answer`. Multiple candidates can be positive.
3. Implement and evaluate: fixed-index baseline, answer-agreement baseline, pointwise learned scorer, and relative learned scorer. The ablation removes all within-question relative features while keeping the same candidate-local features, training labels, learner family and training procedure.
4. Use the independent `predict.py` and `score_predictions.py` on validation for official accuracy. Save predictions, scores, model settings and learned inference assets. Record full fresh-process wall time (prediction plus scoring), not only JSON timing fields.
5. Analyze paired successes and failures, answer-agreement groups and representative errors. If H1 is not supported, retain the stronger simple method as the recommended freeze candidate and report the negative result.

Constraints: four total CPU threads, no external models or data, no ID-dependent behavior, maximum 16 MiB inference assets and 30 seconds for fresh prediction plus scoring on the reference Xeon Platinum 8457C. Local timing is evidence for the local machine only.

## Exploratory follow-up after the first validation comparison

The first learned relative-feature model underperformed the pointwise scorer. The supplied manifest says test/audit banks contain answer disagreement, while the validation bank also has easy unanimous groups. We therefore added a train-only fit restricted to training groups with answer disagreement, using the same two feature sets and ridge learner. We also compare a small fixed set of answer-group aggregation rules using the strongest train-only pointwise checkpoint. These analyses were decided after seeing the initial validation result; they are exploratory and must be identified as such in the final write-up. No test reference is consulted.

A further exploratory model trains on positive-negative candidate pairs within each training question. It fits a linear preference scorer from their feature differences. This tests whether direct within-question supervision helps even when the inference score has the same candidate-local representation.
