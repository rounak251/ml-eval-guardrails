# ml-eval-guardrails

Small, tested checks that catch the most common ways an ML evaluation lies to you.
They come from building and solving held-out ML benchmark challenges. In that work, a
validation score that disagrees with the hidden test set costs real money.

| Check | Question it answers | Typical failure it catches |
|---|---|---|
| `cv_gap` + `discover_groups` | Does my random split overstate the score because samples share a hidden group (session, patient, source)? | CV 0.98 → hidden test 0.52 |
| `adversarial_validation` | Can a model tell train rows from test rows, and which features give it away? | Covariate shift, leaked IDs or timestamps |
| `check_grader_baselines` | Does the grader give the oracle full marks and trivial submissions low marks? | A verifier that only checks the file exists or is valid JSON |
| `check_determinism` | Does the pipeline write byte-identical output on two runs? | Unseeded randomness, non-deterministic GPU kernels, unordered dict/set output |
| `expected_f1_set`, `expected_mrr_order` | Given calibrated probabilities, what output maximises the metric's expected value? | Thresholding at 0.5 under F1 and losing recall |

## Demo

```
$ python examples/demo.py
hidden groups   random KFold 1.000  group KFold 0.373
adversarial AUC 0.825  most shifted feature #3
lazy grader      ['all_zero scores 1.0, as high as the oracle']
expected-F1 set [0, 1, 2] (E[F1]=0.504); threshold 0.5 predicts nothing
```

In the first line, the label belongs to a capture session. Random K-fold reports a perfect
score, while holding out whole sessions shows the model is at chance.

## Usage

```python
from guardrails import discover_groups, cv_gap, check_grader_baselines, check_determinism

groups = discover_groups(nuisance_features, distance_threshold=2.0)  # e.g. image mean/std, EXIF, background embedding
print(cv_gap(model, X, y, groups, scorer=accuracy_score))

report = check_grader_baselines(grade, truth, {"empty": [], "constant": [0] * len(truth)})
assert not report["problems"], report["problems"]

check_determinism(["python", "solution.py", "--output", "{out}"], "submission.csv")
```

`expected_f1_set` is exact. Tests check it against brute-force enumeration over every
possible label set.

## Install and test

```
pip install -e .
pytest -q
```

## Checklist I run before trusting any evaluation

1. Does the oracle score the maximum, and does an empty or constant submission score near zero?
2. Does the starting (broken) state fail the verifier?
3. Do two runs produce the same output, byte for byte?
4. Is the validation split built the way the test split was built (groups, time, novelty)?
5. Does adversarial validation show a shift I haven't modelled?
6. Is the decoded output optimal for the metric, not just for the loss?

MIT licensed.
