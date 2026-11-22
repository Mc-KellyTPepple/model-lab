"""Scores, intervals and the model line-up."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 1
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}   # two-sided 95% t values by degrees of freedom

# id, display name, simplicity rank (lower = simpler), factory
CLASSIFIERS = [
    ("baseline", "Guess the most common answer (baseline)", 0, lambda: DummyClassifier(strategy="most_frequent")),
    ("tree", "Small decision tree", 1, lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED)),
    ("linear", "Logistic regression", 1, lambda: LogisticRegression(max_iter=300)),
    ("boosting", "Gradient boosting", 2, lambda: HistGradientBoostingClassifier(max_iter=60, early_stopping=False, random_state=SEED)),
    ("forest", "Random forest", 2, lambda: RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=SEED)),
]
REGRESSORS = [
    ("baseline", "Always predict the average (baseline)", 0, lambda: DummyRegressor(strategy="mean")),
    ("tree", "Small decision tree", 1, lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=SEED)),
    ("linear", "Ridge regression", 1, lambda: Ridge(alpha=1.0)),
    ("boosting", "Gradient boosting", 2, lambda: HistGradientBoostingRegressor(max_iter=60, early_stopping=False, random_state=SEED)),
    ("forest", "Random forest", 2, lambda: RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=2, n_jobs=1, random_state=SEED)),
]


def lineup(task: str):
    return CLASSIFIERS if task == "classification" else REGRESSORS


def primary_for(task: str, y: np.ndarray) -> str:
    if task == "regression":
        return "r2"
    share = np.bincount(y).max() / len(y)
    return "balanced_accuracy" if share >= 0.7 else "accuracy"


def score(task: str, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if name == "r2":
        return float(r2_score(y_true, y_pred))
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(balanced_accuracy_score(y_true, y_pred))


def all_metrics(task: str, model, X: np.ndarray, y: np.ndarray) -> dict:
    p = model.predict(X)
    if task == "regression":
        return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5)}
    out = {"accuracy": float(accuracy_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro"))}
    try:
        pr = model.predict_proba(X)
        classes = np.unique(y)
        if pr.shape[1] == 2 and len(classes) == 2:
            out["auc"] = float(roc_auc_score(y, pr[:, 1]))
        elif pr.shape[1] == len(classes):
            out["auc"] = float(roc_auc_score(y, pr, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return out


def fold_interval(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    k = len(a)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    t = T975.get(k - 1, 1.96)
    return {"mean": mean, "se": se, "lo": mean - t * se, "hi": mean + t * se, "folds": [float(v) for v in a]}


def bootstrap_ci(task: str, name: str, y_true: np.ndarray, y_pred: np.ndarray, n: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        if name != "r2" and len(np.unique(y_true[i])) < 2:
            continue
        vals.append(score(task, name, y_true[i], y_pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
