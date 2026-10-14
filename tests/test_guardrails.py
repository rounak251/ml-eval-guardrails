import itertools
import sys

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from guardrails import (
    adversarial_validation,
    check_determinism,
    check_grader_baselines,
    cv_gap,
    discover_groups,
    expected_f1_set,
)


def brute_expected_f1(p, chosen):
    n, total = len(p), 0.0
    for truth in itertools.product([0, 1], repeat=n):
        w = np.prod([pi if t else 1 - pi for pi, t in zip(p, truth)])
        T, k = sum(truth), len(chosen)
        tp = sum(truth[i] for i in chosen)
        f1 = 1.0 if (k == 0 and T == 0) else (2 * tp / (k + T) if k + T else 0.0)
        total += w * f1
    return total


def test_expected_f1_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.uniform(0, 1, rng.integers(1, 7))
        chosen, val = expected_f1_set(p)
        assert abs(val - brute_expected_f1(p, chosen)) < 1e-9
        best = max(
            brute_expected_f1(p, list(s))
            for r in range(len(p) + 1)
            for s in itertools.combinations(range(len(p)), r)
        )
        assert abs(val - best) < 1e-9


def test_determinism_detects_randomness(tmp_path):
    stable = [sys.executable, "-c", "open('{out}/o.txt','w').write('42')"]
    noisy = [sys.executable, "-c", "import os;open('{out}/o.txt','wb').write(os.urandom(8))"]
    assert check_determinism(stable, "o.txt")["deterministic"]
    assert not check_determinism(noisy, "o.txt")["deterministic"]


def test_adversarial_validation_flags_shift():
    rng = np.random.default_rng(0)
    same = adversarial_validation(rng.normal(size=(400, 3)), rng.normal(size=(400, 3)))
    a = rng.normal(size=(400, 3))
    b = rng.normal(size=(400, 3)); b[:, 2] += 2.0
    shifted = adversarial_validation(a, b)
    assert same["auc"] < 0.6
    assert shifted["auc"] > 0.85 and shifted["top_features"][0][0] == 2


def test_hidden_groups_inflate_random_cv():
    rng = np.random.default_rng(0)
    n_groups, per = 12, 40
    g = np.repeat(np.arange(n_groups), per)
    centers = rng.normal(scale=5, size=(n_groups, 4))
    y_group = rng.integers(0, 2, n_groups)
    X = centers[g] + rng.normal(scale=0.3, size=(len(g), 4))  # features identify the group
    y = y_group[g]  # label is a property of the group, not learnable across groups
    found = discover_groups(X, distance_threshold=2.0)
    assert len(set(found)) >= n_groups // 2
    res = cv_gap(KNeighborsClassifier(5), X, y, found, accuracy_score)
    assert res["random_kfold"] > 0.9 and res["gap"] > 0.2


def test_grader_that_only_checks_format_is_caught():
    truth = [1, 0, 1, 1]
    lazy = lambda pred, t: 1.0 if len(pred) == len(t) else 0.0
    honest = lambda pred, t: float(np.mean(np.array(pred) == np.array(t)))
    base = {"all_zero": [0, 0, 0, 0], "all_one": [1, 1, 1, 1]}
    assert check_grader_baselines(lazy, truth, base)["problems"]
    assert not check_grader_baselines(honest, truth, base, ceiling=0.8)["problems"]
