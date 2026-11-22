"""The training engine: profile, check for leaks, compare models fairly, test once, explain, and keep a model to serve.

Order matters, and it is the point of the project:
  1. 20% of the rows are set aside and not touched until a single model has been chosen.
  2. Every model is scored by 5-fold cross-validation on the other 80%, with all preprocessing fitted inside each fold.
  3. A suspected leak (one column that alone nearly gives the answer) is left out, and the score with it is shown for contrast.
  4. The winner is the simplest model within one standard error of the best, and it must beat the baseline to be called useful.
  5. The held-out 20% is scored once, with a bootstrap interval.
"""
from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .csvio import DataError, is_missing, parse_csv, to_float
from .metrics import SEED, all_metrics, bootstrap_ci, fold_interval, lineup, primary_for, score, T975
from .prep import Prep, subset
from .profile import detect_task, profile

MIN_CLASS = 5
MIN_ROWS = 30
LEAK_SCORE = 0.97

PRIMARY_LABEL = {"accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "r2": "R²"}


@dataclass
class Trained:
    prep: Prep
    model: object
    task: str
    target: str
    classes: list[str] | None
    features: list[dict]
    primary: str
    test_primary: float
    typical_error: float | None
    created: float = field(default_factory=time.time)
    n_train: int = 0


def _norm_label(v: str) -> str:
    s = v.strip()
    f = to_float(s)
    return str(int(f)) if f is not None and float(f).is_integer() and not any(c.isalpha() for c in s) else s


def _pct(x: float, primary: str) -> str:
    return f"{x:.2f}" if primary == "r2" else f"{x * 100:.1f}%"


def run(text: str, target: str | None = None, task: str | None = None, include_suspected: bool = False,
        max_train_rows: int = 2000, time_budget: float = 60.0, progress=lambda stage, pct, msg: None) -> tuple[dict, Trained]:
    t_start = time.monotonic()
    warnings.filterwarnings("ignore")
    progress("read", 2, "Reading your file")
    header, rows = parse_csv(text)
    prof = profile(header, rows)
    target = target or prof["suggested_target"]
    if target not in header:
        raise DataError("Choose which column to predict.")
    tcol = next(c for c in prof["columns"] if c["name"] == target)
    task = task or detect_task(tcol)
    if task is None:
        raise DataError(f"'{target}' cannot be predicted: it is {tcol['kind']}. Choose a column of numbers or a few categories.")
    data = {h: np.array([r[i] for r in rows], dtype=object) for i, h in enumerate(header)}
    warns = [w for w in prof["warnings"]]
    dropped = {"missing_target": 0, "rare_classes": [], "subsampled": False}

    # ---- target ----
    if task == "classification":
        lab = np.array([_norm_label(v) for v in data[target]], dtype=object)
        known = np.array([not is_missing(v) for v in lab])
        counts = Counter(lab[known])
        keep = [k for k, c in counts.most_common() if c >= MIN_CLASS]
        dropped["rare_classes"] = [k for k, c in counts.items() if c < MIN_CLASS]
        if len(keep) < 2:
            raise DataError("The column to predict needs at least two values that each appear in 5 or more rows.")
        if len(keep) > 12:
            raise DataError(f"'{target}' has {len(keep)} different values. Pick a column with at most 12 categories, or a numeric column.")
        classes = keep
        cmap = {k: i for i, k in enumerate(classes)}
        valid = known & np.array([v in cmap for v in lab])
        y_all = np.array([cmap.get(v, -1) for v in lab])
    else:
        classes = None
        f = np.array([np.nan if is_missing(v) or to_float(v) is None else to_float(v) for v in data[target]])
        valid = ~np.isnan(f)
        y_all = f
    dropped["missing_target"] = int((~valid).sum())
    if dropped["missing_target"]:
        warns.append({"level": "info", "text": f"{dropped['missing_target']} rows have no usable value for '{target}' and were skipped."})
    if dropped["rare_classes"]:
        warns.append({"level": "info", "text": f"Categories with fewer than {MIN_CLASS} rows were skipped: {', '.join(map(str, dropped['rare_classes'][:6]))}."})
    idx_all = np.flatnonzero(valid)
    if len(idx_all) < MIN_ROWS:
        raise DataError(f"I need at least {MIN_ROWS} rows with a known '{target}'. Found {len(idx_all)}.")
    rows_total = len(idx_all)
    if rows_total > max_train_rows:
        strat = y_all[idx_all] if task == "classification" else None
        idx_all, _ = train_test_split(idx_all, train_size=max_train_rows, stratify=strat, random_state=SEED)
        idx_all = np.sort(idx_all)
        dropped["subsampled"] = True
        warns.append({"level": "info", "text": f"Used a random {max_train_rows:,} of {rows_total:,} rows to keep this quick."})
    y = y_all[idx_all]
    d = subset(data, idx_all)

    # ---- features ----
    kinds, excluded = {}, []
    for c in prof["columns"]:
        if c["name"] == target:
            continue
        if c["kind"] in ("numeric", "categorical") and c["missing_pct"] < 60:
            kinds[c["name"]] = c["kind"]
        else:
            why = {"id": "looks like an ID", "constant": "same value in every row", "empty": "empty", "text": "free text", "highcard": "too many different values"}.get(c["kind"], f"{c['missing_pct']}% empty")
            excluded.append({"name": c["name"], "reason": why})
    if not kinds:
        raise DataError("There are no usable columns to learn from besides the one to predict.")

    # ---- hold out the test rows ----
    progress("split", 6, "Setting aside 20% of the rows as a final test")
    n = len(y)
    strat = y if task == "classification" and min(np.bincount(y)) >= 2 else None
    tr, te = train_test_split(np.arange(n), test_size=0.2, stratify=strat, random_state=SEED)
    prim = primary_for(task, y[tr])
    d_tr, d_te, y_tr, y_te = subset(d, tr), subset(d, te), y[tr], y[te]
    if task == "classification":
        warns += _imbalance(y_tr, classes)

    # ---- leak scan (training rows only) ----
    progress("leaks", 10, "Looking for columns that give the answer away")
    leaks, base_guess = _leak_scan(d_tr, y_tr, kinds, task, prim)
    suspected = [] if include_suspected else [l["column"] for l in leaks]
    for l in leaks:
        if not include_suspected:
            excluded.append({"name": l["column"], "reason": f"suspected leak: alone it predicts {_pct(l['score'], prim)}"})
    use = {k: v for k, v in kinds.items() if k not in suspected}
    if not use:
        raise DataError("After removing suspected leaks, no columns are left to learn from.")

    # ---- cross-validation of every model ----
    k = max(2, min(5, int(np.bincount(y_tr).min()))) if task == "classification" else 5
    splitter = StratifiedKFold(k, shuffle=True, random_state=SEED) if task == "classification" else KFold(k, shuffle=True, random_state=SEED)
    folds = []
    for a, b in splitter.split(np.zeros(len(y_tr)), y_tr if task == "classification" else None):
        p = Prep(use).fit(subset(d_tr, a))
        folds.append((p.transform(subset(d_tr, a)), y_tr[a], p.transform(subset(d_tr, b)), y_tr[b]))
    models, t_models = [], time.monotonic()
    line = lineup(task)
    for i, (mid, name, rank, make) in enumerate(line):
        pct = 14 + int(60 * i / len(line))
        progress("models", pct, f"Testing: {name}")
        if i > 0 and time.monotonic() - t_start > time_budget:
            models.append({"id": mid, "name": name, "rank": rank, "skipped": True, "note": "Skipped to stay within the time limit."})
            continue
        t0 = time.monotonic()
        try:
            va, trn = [], []
            for Xa, ya, Xb, yb in folds:
                m = make().fit(Xa, ya)
                va.append(score(task, prim, yb, m.predict(Xb)))
                trn.append(score(task, prim, ya, m.predict(Xa)))
        except Exception as exc:
            models.append({"id": mid, "name": name, "rank": rank, "skipped": True, "note": f"Could not be trained on this data ({type(exc).__name__})."})
            continue
        models.append({"id": mid, "name": name, "rank": rank, "cv": fold_interval(va), "train_mean": float(np.mean(trn)),
                       "gap": float(np.mean(trn) - np.mean(va)), "secs": round(time.monotonic() - t0, 2)})
    ok = [m for m in models if "cv" in m]
    base = next((m for m in ok if m["id"] == "baseline"), None)
    cands = [m for m in ok if m["id"] != "baseline"]
    if base is None or not cands:
        raise DataError("No model could be trained on this data.")

    # ---- choose: simplest model within one standard error of the best ----
    progress("choose", 78, "Choosing a model")
    best = max(cands, key=lambda m: m["cv"]["mean"])
    near = [m for m in cands if m["cv"]["mean"] >= best["cv"]["mean"] - best["cv"]["se"]]
    winner = min(near, key=lambda m: (m["rank"], -m["cv"]["mean"]))
    reason = (f"It has the highest cross-validated score." if winner is best else
              f"{best['name']} scored slightly higher, but within one standard error, so the simpler {winner['name']} was chosen.")
    diffs = np.array(winner["cv"]["folds"]) - np.array(base["cv"]["folds"])
    kf = len(diffs)
    dse = float(diffs.std(ddof=1) / np.sqrt(kf)) if kf > 1 else 0.0
    t = T975.get(kf - 1, 1.96)
    dm, dlo, dhi = float(diffs.mean()), float(diffs.mean() - t * dse), float(diffs.mean() + t * dse)
    min_gain = 0.03
    beats = dlo > 0 and dm >= min_gain
    for m in models:
        m["winner"] = m is winner

    # ---- the single look at the test rows ----
    progress("test", 82, "Scoring the held-out test rows once")
    make = next(mk for (mid, _, _, mk) in line if mid == winner["id"])
    prep = Prep(use).fit(d_tr)
    Xtr, Xte = prep.transform(d_tr), prep.transform(d_te)
    model = make().fit(Xtr, y_tr)
    pred = model.predict(Xte)
    test_metrics = all_metrics(task, model, Xte, y_te)
    tp = score(task, prim, y_te, pred)
    lo, hi = bootstrap_ci(task, prim, y_te, pred)
    bmodel = next(mk for (mid, _, _, mk) in line if mid == "baseline")().fit(Xtr, y_tr)
    base_test = score(task, prim, y_te, bmodel.predict(Xte))
    base_test_metrics = all_metrics(task, bmodel, Xte, y_te)

    # ---- explanations ----
    progress("explain", 88, "Working out which columns mattered")
    importance = _importance(model, prep, d_te, y_te, task, prim, use)
    extra = {}
    if task == "classification":
        from sklearn.metrics import confusion_matrix
        extra["confusion"] = {"classes": classes, "matrix": confusion_matrix(y_te, pred, labels=list(range(len(classes)))).tolist()}
    else:
        order = np.random.default_rng(SEED).permutation(len(y_te))[:150]
        extra["scatter"] = [[float(y_te[i]), float(pred[i])] for i in order]
    typical = float(np.median(np.abs(y_te - pred))) if task == "regression" else None
    q90 = float(np.quantile(np.abs(y_te - pred), 0.9)) if task == "regression" else None

    # ---- contrast: what the score would have been with the suspected leak left in ----
    contrast = None
    if leaks and not include_suspected:
        progress("contrast", 92, "Measuring how misleading the leak would have been")
        full = {**use, **{l["column"]: kinds[l["column"]] for l in leaks}}
        ml = make
        va = []
        for a, b in splitter.split(np.zeros(len(y_tr)), y_tr if task == "classification" else None):
            p = Prep(full).fit(subset(d_tr, a))
            va.append(score(task, prim, y_tr[b], ml().fit(p.transform(subset(d_tr, a)), y_tr[a]).predict(p.transform(subset(d_tr, b)))))
        contrast = {"model": winner["name"], "cv_with_leak": float(np.mean(va)), "cv_without_leak": winner["cv"]["mean"], "columns": [l["column"] for l in leaks]}

    # ---- warnings about trust ----
    if not beats:
        warns.append({"level": "warn", "text": "No model is clearly better than the baseline here. The columns may not contain a usable signal for this target."})
    if winner["gap"] > 0.2:
        warns.append({"level": "warn", "text": f"{winner['name']} fits the training rows much better than new rows (gap {_pct(winner['gap'], prim) if prim != 'r2' else round(winner['gap'], 2)}). It memorises some noise."})
    if tp < winner["cv"]["mean"] - 2 * max(winner["cv"]["se"], 0.01):
        warns.append({"level": "warn", "text": "The test score is lower than the cross-validated score. The cross-validation was probably a little optimistic."})
    if len(y) < 200:
        warns.append({"level": "warn", "text": f"With {len(y)} usable rows the test score has a wide interval. Do not read small differences between models."})

    # ---- refit on every row for serving ----
    progress("serve", 96, "Preparing the model for the prediction form")
    prep_all = Prep(use).fit(d)
    model_all = make().fit(prep_all.transform(d), y)
    form = []
    for col, kind in use.items():
        s = prep_all.spec[col]
        form.append({"name": col, "kind": kind, **({"min": s["min"], "max": s["max"], "median": s["median"]} if kind == "numeric" else {"levels": s["levels"]})})
    trained = Trained(prep_all, model_all, task, target, classes, form, prim, tp, q90, n_train=int(len(y)))

    result = {
        "task": task, "target": target, "classes": classes, "primary": {"name": prim, "label": PRIMARY_LABEL[prim]},
        "rows": {"total": rows_total, "used": int(n), "train": int(len(tr)), "test": int(len(te))},
        "dropped": dropped, "folds": k,
        "features": [{"name": c, "kind": kd} for c, kd in use.items()], "excluded": excluded,
        "leaks": leaks, "contrast": contrast,
        "models": models,
        "winner": {"id": winner["id"], "name": winner["name"], "reason": reason},
        "verdict": {"beats_baseline": beats, "diff": {"mean": dm, "lo": dlo, "hi": dhi}, "min_gain": min_gain},
        "test": {"n": int(len(y_te)), "primary": {"value": tp, "lo": lo, "hi": hi}, "baseline": base_test, "metrics": test_metrics, "baseline_metrics": base_test_metrics, "typical_error": typical, "error_q90": q90},
        "importance": importance, "warnings": warns, "form": form,
        "profile": {"rows": prof["rows"], "duplicates": prof["duplicates"], "columns": prof["columns"]},
        "seconds": round(time.monotonic() - t_start, 1), **extra,
    }
    result["summary"] = _summary(result)
    progress("done", 100, "Done")
    return result, trained


def _imbalance(y: np.ndarray, classes) -> list[dict]:
    c = np.bincount(y)
    share = c.max() / c.sum()
    if share >= 0.9:
        return [{"level": "warn", "text": f"'{classes[int(c.argmax())]}' is {share * 100:.0f}% of the rows. A model that always says that is already {share * 100:.0f}% accurate, so accuracy alone would mislead. Scores use balanced accuracy."}]
    if share >= 0.7:
        return [{"level": "info", "text": f"One category is {share * 100:.0f}% of the rows, so scores use balanced accuracy instead of plain accuracy."}]
    return []


def _leak_scan(d: dict, y: np.ndarray, kinds: dict, task: str, prim: str) -> tuple[list[dict], float]:
    """One column at a time: if that column alone nearly gives the answer, flag it.

    Numbers are tried with a tree of up to 24 leaves (enough for a noisy copy of a number, or a handful of classes).
    Categories are tried with a plain lookup: the most common answer (or the average) for each category.
    Both are scored by 3-fold cross-validation, so a column cannot pass by memorising rows."""
    cv = 3
    if task == "classification" and np.bincount(y).min() < cv:
        return [], 0.0
    splitter = StratifiedKFold(cv, shuffle=True, random_state=SEED) if task == "classification" else KFold(cv, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y)), y if task == "classification" else None))
    leaf = max(3, int(0.01 * len(y)))
    out = []
    for col, kind in kinds.items():
        vals = []
        try:
            if kind == "categorical":
                lv = np.array(["" if is_missing(v) else str(v).strip() for v in d[col]], dtype=object)
                for a, b in folds:
                    table = {}
                    for k in set(lv[a]):
                        ya = y[a][lv[a] == k]
                        table[k] = int(np.bincount(ya).argmax()) if task == "classification" else float(ya.mean())
                    default = int(np.bincount(y[a]).argmax()) if task == "classification" else float(y[a].mean())
                    pred = np.array([table.get(k, default) for k in lv[b]])
                    vals.append(score(task, prim, y[b], pred))
            else:
                X = Prep({col: kind}).fit(d).transform(d)
                for a, b in folds:
                    m = (DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor)(max_leaf_nodes=24, min_samples_leaf=leaf, random_state=SEED)
                    vals.append(score(task, prim, y[b], m.fit(X[a], y[a]).predict(X[b])))
        except Exception:
            continue
        s = float(np.mean(vals))
        if s >= LEAK_SCORE:
            out.append({"column": col, "score": s, "note": "This column alone almost gives the answer. If it is only known after the outcome happens, a model using it would look perfect and be useless."})
    return sorted(out, key=lambda x: -x["score"]), 0.0


def _importance(model, prep: Prep, d_te: dict, y_te: np.ndarray, task: str, prim: str, use: dict, repeats: int = 3) -> list[dict]:
    rng = np.random.default_rng(SEED)
    base = score(task, prim, y_te, model.predict(prep.transform(d_te)))
    out = []
    for col in use:
        drops = []
        for _ in range(repeats):
            dd = dict(d_te)
            dd[col] = d_te[col][rng.permutation(len(y_te))]
            drops.append(base - score(task, prim, y_te, model.predict(prep.transform(dd))))
        out.append({"column": col, "drop": float(np.mean(drops))})
    out.sort(key=lambda r: -r["drop"])
    return [{**r, "drop": round(r["drop"], 4)} for r in out[:10]]


def _summary(r: dict) -> list[str]:
    p = r["primary"]["name"]
    lab = r["primary"]["label"]
    t = r["test"]["primary"]
    w = r["winner"]["name"]
    lines = [f"Best model: {w}. On the {r['rows']['test']} test rows it scored {_pct(t['value'], p)} {lab} (95% interval {_pct(t['lo'], p)} to {_pct(t['hi'], p)}), against {_pct(r['test']['baseline'], p)} for the baseline."]
    if r["verdict"]["beats_baseline"]:
        lines.append("It is clearly better than the baseline in cross-validation, so the columns carry real signal for this target.")
    else:
        lines.append("It is not clearly better than the baseline, so these columns may not predict this target. Do not rely on it.")
    if r["importance"] and r["importance"][0]["drop"] > 0:
        top = [i["column"] for i in r["importance"][:3] if i["drop"] > 0]
        lines.append(f"The columns that mattered most: {', '.join(top)}.")
    if r["contrast"]:
        c = r["contrast"]
        lines.append(f"Suspected leak ({', '.join(c['columns'])}) was left out. With it, the score would have looked like {_pct(c['cv_with_leak'], p)}, against an honest {_pct(c['cv_without_leak'], p)}.")
    return lines


def predict_one(tr: Trained, values: dict[str, str]) -> dict:
    data = {c: np.array([str(values.get(c, ""))], dtype=object) for c in tr.prep.kinds}
    X = tr.prep.transform(data)
    notes = []
    for f in tr.features:
        v = str(values.get(f["name"], "")).strip()
        if is_missing(v):
            notes.append(f"'{f['name']}' was left empty, so a typical value was assumed.")
        elif f["kind"] == "numeric":
            x = to_float(v)
            if x is None:
                notes.append(f"'{f['name']}' is not a number, so a typical value was assumed.")
            elif x < f["min"] or x > f["max"]:
                notes.append(f"'{f['name']}' = {v} is outside the range the model saw ({f['min']:g} to {f['max']:g}). Treat the answer with caution.")
        elif v not in f["levels"]:
            notes.append(f"'{f['name']}' = {v} is a value the model has not seen.")
    if tr.task == "regression":
        p = float(tr.model.predict(X)[0])
        return {"prediction": p, "typical_error": tr.typical_error, "notes": notes}
    proba = tr.model.predict_proba(X)[0]
    order = np.argsort(-proba)[:3]
    return {"prediction": tr.classes[int(order[0])], "candidates": [{"label": tr.classes[int(i)], "probability": float(proba[i])} for i in order], "notes": notes}


def predict_rows(tr: Trained, data: dict[str, np.ndarray]) -> np.ndarray:
    """Predictions for many rows (labels as class indices for classification)."""
    return tr.model.predict(tr.prep.transform(data))
