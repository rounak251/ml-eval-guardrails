"""Five checks on synthetic data. Run: python examples/demo.py"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from guardrails import adversarial_validation, check_grader_baselines, cv_gap, discover_groups, expected_f1_set

rng = np.random.default_rng(0)

# 1. Hidden groups: the label belongs to the recording session, not the sample.
g = np.repeat(np.arange(12), 40)
X = rng.normal(scale=5, size=(12, 4))[g] + rng.normal(scale=0.3, size=(480, 4))
y = rng.integers(0, 2, 12)[g]
res = cv_gap(KNeighborsClassifier(5), X, y, discover_groups(X, 2.0), accuracy_score)
print(f"hidden groups   random KFold {res['random_kfold']:.3f}  group KFold {res['group_kfold']:.3f}")

# 2. Covariate shift between train and test.
a = rng.normal(size=(500, 5)); b = rng.normal(size=(500, 5)); b[:, 3] += 1.5
av = adversarial_validation(a, b)
print(f"adversarial AUC {av['auc']:.3f}  most shifted feature #{av['top_features'][0][0]}")

# 3. A grader that only checks the output has the right shape.
truth = rng.integers(0, 2, 50).tolist()
lazy = lambda pred, t: float(len(pred) == len(t))
print("lazy grader     ", check_grader_baselines(lazy, truth, {"all_zero": [0] * 50})["problems"])

# 4. Metric-aware decoding beats a 0.5 threshold for F1.
p = np.array([0.45, 0.4, 0.35, 0.05, 0.02])
chosen, ef1 = expected_f1_set(p)
print(f"expected-F1 set {chosen} (E[F1]={ef1:.3f}); threshold 0.5 predicts nothing")
