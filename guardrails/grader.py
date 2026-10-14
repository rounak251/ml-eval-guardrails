"""A grader you can trust must score trivial submissions near zero (or at the
chance level) and the ground truth at the maximum."""


def check_grader_baselines(grade, truth, baselines, perfect_score=1.0, tol=1e-9, ceiling=None):
    """`grade(pred, truth) -> float`. `baselines` maps a name to a trivial
    prediction (empty, constant, majority class, shuffled truth...).

    Returns per-baseline scores and a list of problems found.
    """
    problems = []
    oracle = grade(truth, truth)
    if abs(oracle - perfect_score) > tol:
        problems.append(f"oracle scores {oracle}, expected {perfect_score}")
    scores = {}
    for name, pred in baselines.items():
        try:
            s = grade(pred, truth)
        except Exception as e:  # a grader that crashes on bad input is also a finding
            problems.append(f"{name}: grader raised {type(e).__name__}: {e}")
            continue
        scores[name] = s
        if s >= oracle - tol:
            problems.append(f"{name} scores {s}, as high as the oracle")
        elif ceiling is not None and s > ceiling:
            problems.append(f"{name} scores {s} > allowed ceiling {ceiling}")
    return {"oracle": oracle, "baselines": scores, "problems": problems}
