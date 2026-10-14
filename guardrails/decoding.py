"""Turn calibrated probabilities into the output that maximises the metric's
expected value, instead of thresholding at 0.5 or sorting by habit."""
import numpy as np


def expected_f1_set(p):
    """Choose the label set maximising expected F1 for independent label
    probabilities `p` (Jansche / Nan et al. style exact computation, O(n^3)).

    Returns (indices, expected_f1). The empty set is chosen when it is better
    (expected F1 of the empty prediction = P(no positives)).
    """
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(-p)
    q = p[order]
    # dist[k] = P(number of true positives among all labels == k)
    dist = np.zeros(n + 1)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= 1 - pi
    best_val, best_k = dist[0], 0  # predict nothing: F1 = 1 iff truth empty
    for k in range(1, n + 1):
        # E[2 TP / (k + T)] with TP from top-k; condition on top-k membership
        top, rest = q[:k], q[k:]
        a = np.zeros(k + 1); a[0] = 1.0  # count positives inside the chosen set
        for pi in top:
            a[1:] = a[1:] * (1 - pi) + a[:-1] * pi
            a[0] *= 1 - pi
        b = np.zeros(n - k + 1); b[0] = 1.0  # count positives outside it
        for pi in rest:
            b[1:] = b[1:] * (1 - pi) + b[:-1] * pi
            b[0] *= 1 - pi
        i = np.arange(k + 1)[:, None]
        j = np.arange(n - k + 1)[None, :]
        val = float((a[:, None] * b[None, :] * 2 * i / (k + i + j)).sum())
        if val > best_val:
            best_val, best_k = val, k
    return order[:best_k].tolist(), best_val


def expected_mrr_order(p):
    """For a single correct answer among candidates with probabilities `p`,
    sorting by `p` maximises expected reciprocal rank. Returns the order and
    the expected MRR, useful for checking whether calibration is the bottleneck."""
    p = np.asarray(p, float)
    order = np.argsort(-p)
    return order.tolist(), float((p[order] / np.arange(1, len(p) + 1)).sum())
