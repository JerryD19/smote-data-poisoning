"""
Label-flipping poisoning of the minority class, and how class rebalancing changes its impact.

Core idea: SMOTE creates synthetic minority samples by interpolating between a minority
point and one of its k nearest minority neighbours. If an attacker relabels a few
majority-class records as minority ("poison"), every synthetic sample built from a
poisoned point (as the base or as the neighbour) inherits that poison.

This module provides:
  - smote_with_parents: SMOTE that also returns the two parents of every synthetic sample,
    so the spread of poison can be measured directly.
  - poison_minority: relabel k majority-class training records as minority.
  - run_experiment: the full grid (poison level x rebalancing method x model x repeat).

Author: Jeremiah Dibie
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_DROP = ["child_ID", "group", "asd_risk"]  # group leaks the label (see autism-risk-detection-ml)


def load_qchat(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load Q-CHAT data and return features X and binary target y (1 = high risk)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    y = (df["asd_risk"] == 2).astype(int)
    X = df.drop(columns=[c for c in FEATURES_DROP if c in df.columns])
    return X, y


def smote_with_parents(X_min: np.ndarray, n_new: int, k: int = 5, rng=None):
    """SMOTE (Chawla et al., 2002) that records each synthetic sample's parents.

    Same sampling scheme as imbalanced-learn: pick a minority sample uniformly at random,
    pick one of its k nearest minority neighbours uniformly, interpolate at a uniform gap.

    Returns
    -------
    X_new : (n_new, d) synthetic samples
    base, nbr : (n_new,) indices into X_min of the two parents
    """
    rng = np.random.default_rng(rng)
    if n_new <= 0:
        return np.empty((0, X_min.shape[1])), np.empty(0, int), np.empty(0, int)
    k_eff = min(k, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    base = rng.integers(0, len(X_min), n_new)
    nbr = neigh[base, rng.integers(0, k_eff, n_new)]
    gap = rng.random((n_new, 1))
    X_new = X_min[base] + gap * (X_min[nbr] - X_min[base])
    return X_new, base, nbr


def poison_minority(y_train: np.ndarray, order: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Relabel the first k indices of `order` (majority-class rows) as minority.

    Using a fixed random order per repeat makes poison sets nested (k=5 contains k=2),
    so results across poison levels are paired.
    """
    y_p = y_train.copy()
    idx = order[:k]
    y_p[idx] = 1
    poisoned = np.zeros(len(y_train), bool)
    poisoned[idx] = True
    return y_p, poisoned


def make_model(name: str, balanced: bool, seed: int):
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=5000, class_weight=cw, random_state=seed)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight=cw, random_state=seed)
    raise ValueError(name)


def _scores(model, X):
    return model.decision_function(X)


def fit_and_score(X_tr, y_tr, poisoned, X_te, y_te, model_name, method, seed, k_nn=5):
    """Train one model with one rebalancing method; return metrics and SMOTE spread stats."""
    scaler = StandardScaler().fit(X_tr)
    Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
    out = {}

    if method == "smote":
        min_idx = np.flatnonzero(y_tr == 1)
        n_new = int((y_tr == 0).sum() - len(min_idx))
        X_new, base, nbr = smote_with_parents(Xs_tr[min_idx], n_new, k=k_nn, rng=seed)
        pois_min = poisoned[min_idx]
        from_base = pois_min[base]
        from_any = from_base | pois_min[nbr]
        Xs_fit = np.vstack([Xs_tr, X_new])
        y_fit = np.concatenate([y_tr, np.ones(n_new, int)])
        k_p = int(pois_min.sum())
        out.update({
            "n_minority_real": len(min_idx),
            "n_synthetic": n_new,
            "synthetic_from_poison_base": int(from_base.sum()),
            "synthetic_from_poison_any": int(from_any.sum()),
            "poison_share_before": k_p / len(min_idx),
            "poison_share_after": (k_p + int(from_any.sum())) / (len(min_idx) + n_new),
        })
        model = make_model(model_name, balanced=False, seed=seed).fit(Xs_fit, y_fit)
    else:
        model = make_model(model_name, balanced=(method == "class_weight"), seed=seed).fit(Xs_tr, y_tr)

    s = _scores(model, Xs_te)
    pred = (s > 0).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum()); fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum()); tn = int(((pred == 0) & (y_te == 0)).sum())
    out.update({
        "roc_auc": roc_auc_score(y_te, s),
        "pr_auc": average_precision_score(y_te, s),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "fpr": fp / (fp + tn),
        "flagged_share": pred.mean(),
    })
    return out


def poison_order(X_tr: np.ndarray, y_tr: np.ndarray, attack: str, seed: int) -> np.ndarray:
    """Order in which majority-class training rows get relabelled.

    random  - a random choice of low-risk children (label noise / unsophisticated attacker)
    outlier - low-risk children furthest from the high-risk centroid first (an attacker who
              picks the most clearly low-risk records, to drag the decision boundary furthest)
    """
    maj = np.flatnonzero(y_tr == 0)
    if attack == "random":
        return np.random.default_rng(10_000 + seed).permutation(maj)
    if attack == "outlier":
        Xs = StandardScaler().fit_transform(X_tr)
        centroid = Xs[y_tr == 1].mean(axis=0)
        dist = np.linalg.norm(Xs[maj] - centroid, axis=1)
        return maj[np.argsort(-dist)]
    raise ValueError(attack)


def run_experiment(X: pd.DataFrame, y: pd.Series, poison_levels=(0, 2, 5, 10, 15, 20, 30),
                   methods=("none", "class_weight", "smote"),
                   models=("Logistic Regression", "SVM"), attacks=("random", "outlier"),
                   n_repeats=20, test_size=0.2):
    """Full grid. The test set is split off FIRST and never poisoned or resampled."""
    rows = []
    Xv, yv = X.to_numpy(float), y.to_numpy()
    for rep in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(Xv, yv, test_size=test_size, stratify=yv, random_state=rep)
        for attack in attacks:
            order = poison_order(X_tr, y_tr, attack, rep)
            for k in poison_levels:
                y_p, poisoned = poison_minority(y_tr, order, k)
                for model_name in models:
                    for method in methods:
                        r = fit_and_score(X_tr, y_p, poisoned, X_te, y_te, model_name, method, seed=rep)
                        rows.append({"repeat": rep, "attack": attack, "poisoned": k, "model": model_name,
                                     "method": method, **r})
    return pd.DataFrame(rows)
