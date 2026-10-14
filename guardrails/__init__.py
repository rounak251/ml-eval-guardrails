"""Checks that catch the usual ways an ML evaluation lies to you."""
from .determinism import check_determinism
from .adversarial import adversarial_validation
from .groups import discover_groups, cv_gap
from .grader import check_grader_baselines
from .decoding import expected_f1_set, expected_mrr_order

__all__ = [
    "check_determinism",
    "adversarial_validation",
    "discover_groups",
    "cv_gap",
    "check_grader_baselines",
    "expected_f1_set",
    "expected_mrr_order",
]
