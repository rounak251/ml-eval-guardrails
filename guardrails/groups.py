"""Find hidden groups (sessions, patients, sources) and measure how much a
random split inflates the score compared with a group split."""
import numpy as np
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler


def discover_groups(Z, distance_threshold):
    """Cluster nuisance descriptors `Z` (e.g. image mean/std, timestamps,
    embedding of the background) into candidate hidden groups."""
    Zs = StandardScaler().fit_transform(np.asarray(Z, float))
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="average"
    )
    return cl.fit_predict(Zs)


def cv_gap(model, X, y, groups, scorer, n_splits=5, seed=0):
    """Score `model` with shuffled KFold and with GroupKFold on `groups`.

    A large positive gap means the random-split number is mostly the model
    recognising groups, not solving the task.
    """
    def run(splitter, g=None):
        scores = []
        for tr, va in splitter.split(X, y, g):
            m = clone(model).fit(X[tr], y[tr])
            scores.append(scorer(y[va], m.predict(X[va])))
        return float(np.mean(scores))

    random = run(KFold(n_splits, shuffle=True, random_state=seed))
    grouped = run(GroupKFold(n_splits), groups)
    return {"random_kfold": random, "group_kfold": grouped, "gap": random - grouped}
