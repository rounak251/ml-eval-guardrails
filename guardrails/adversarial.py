"""Adversarial validation: can a model tell train rows from test rows?"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def adversarial_validation(X_train, X_test, n_splits=5, seed=0, top_k=10):
    """Return out-of-fold AUC of a train-vs-test classifier and the features
    whose permutation hurts it most.

    AUC near 0.5 means the test set looks like train. AUC well above 0.5 means
    a random split of train will overstate your test score.
    """
    X = np.vstack([X_train, X_test]).astype(float)
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    oof = np.zeros(len(y))
    models = []
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        models.append((m, va))
    auc = roc_auc_score(y, oof)

    rng = np.random.default_rng(seed)
    drops = np.zeros(X.shape[1])
    for m, va in models:
        base = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        for j in range(X.shape[1]):
            Xp = X[va].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[j] += base - roc_auc_score(y[va], m.predict_proba(Xp)[:, 1])
    drops /= len(models)
    order = np.argsort(-drops)[:top_k]
    return {"auc": float(auc), "top_features": [(int(j), float(drops[j])) for j in order]}
