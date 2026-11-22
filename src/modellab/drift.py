"""Monitoring: has new data drifted away from what the model was trained on?

Per column we compare the share of rows in each bin (numbers: deciles of the training data;
categories: the levels seen in training) with the Population Stability Index.
Common reading: under 0.10 stable, 0.10 to 0.25 moderate shift, over 0.25 large shift.
The sampling-noise floor (what two samples of the same data would show by chance) is subtracted first, because small files otherwise look drifted.
"""
from __future__ import annotations

import numpy as np

from .csvio import DataError, is_missing, parse_csv, to_float
from .prep import NA, OTHER, _bin_frac

EPS = 1e-4
MIN_ROWS = 30


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def noise_floor(bins: int, n_new: int, n_ref: int) -> float:
    """PSI between two samples of the SAME distribution is not zero; this is about what chance alone gives."""
    return (bins - 1) * (1 / max(n_new, 1) + 1 / max(n_ref, 1))


def label(v: float) -> str:
    return "stable" if v < 0.10 else "moderate" if v < 0.25 else "large"


def check(trained, text: str) -> dict:
    from .engine import predict_rows
    header, rows = parse_csv(text)
    if len(rows) < MIN_ROWS:
        raise DataError(f"I need at least {MIN_ROWS} rows of new data to judge drift. Found {len(rows)}.")
    missing = [c for c in trained.prep.kinds if c not in header]
    if missing:
        raise DataError("The new file is missing columns the model uses: " + ", ".join(missing[:6]))
    data = {h: np.array([r[i] for r in rows], dtype=object) for i, h in enumerate(header)}
    cols = []
    for col, ref in trained.prep.ref.items():
        raw = data[col]
        if ref["kind"] == "numeric":
            x = np.array([np.nan if is_missing(v) or to_float(v) is None else to_float(v) for v in raw])
            miss = float(np.isnan(x).mean())
            x = np.where(np.isnan(x), trained.prep.spec[col]["median"], x)
            p = psi(ref["frac"], _bin_frac(x, np.array(ref["edges"])))
            p = max(0.0, p - noise_floor(len(ref["frac"]), len(x), trained.n_train))
            cols.append({"column": col, "psi": round(p, 3), "level": label(p), "missing_pct": round(miss * 100, 1),
                         "new_mean": round(float(x.mean()), 3), "train_mean": round(trained.prep.spec[col]["mean"], 3)})
        else:
            vals = [NA if is_missing(v) else v.strip() for v in raw]
            known = set(ref["frac"])
            keys = list(ref["frac"])
            cnt = {k: 0 for k in keys}
            for v in vals:
                k = v if v in known else OTHER
                cnt[k] = cnt.get(k, 0) + 1
            if OTHER not in cnt:
                cnt[OTHER] = 0
            keys = list(dict.fromkeys(keys + [OTHER]))
            p = psi([ref["frac"].get(k, 0) for k in keys], [cnt[k] / len(vals) for k in keys])
            p = max(0.0, p - noise_floor(len(keys), len(vals), trained.n_train))
            unseen = cnt[OTHER] / len(vals)
            cols.append({"column": col, "psi": round(p, 3), "level": label(p), "unseen_pct": round(unseen * 100, 1)})
    cols.sort(key=lambda c: -c["psi"])
    out = {"rows": len(rows), "columns": cols, "worst": cols[0]["column"] if cols else None}
    drifted = [c for c in cols if c["level"] == "large"]
    some = [c for c in cols if c["level"] == "moderate"]
    out["verdict"] = ("large" if drifted else "moderate" if some else "stable")
    out["message"] = {
        "large": f"{len(drifted)} column(s) have shifted a lot ({', '.join(c['column'] for c in drifted[:4])}). Predictions on this data are less trustworthy; consider retraining on newer data.",
        "moderate": f"{len(some)} column(s) have shifted somewhat ({', '.join(c['column'] for c in some[:4])}). Keep an eye on it.",
        "stable": "New data looks like the data the model was trained on.",
    }[out["verdict"]]

    # Prediction mix, and real accuracy if the new file includes the answers.
    preds = predict_rows(trained, data)
    if trained.task == "classification":
        from collections import Counter
        c = Counter(trained.classes[int(i)] for i in preds)
        out["prediction_mix"] = {k: round(v / len(rows), 3) for k, v in c.most_common()}
    else:
        out["prediction_summary"] = {"mean": round(float(np.mean(preds)), 3), "min": round(float(np.min(preds)), 3), "max": round(float(np.max(preds)), 3)}
    if trained.target in data:
        from .engine import _norm_label
        from .metrics import score
        if trained.task == "classification":
            cmap = {k: i for i, k in enumerate(trained.classes)}
            lab = [_norm_label(v) for v in data[trained.target]]
            keep = [i for i, v in enumerate(lab) if v in cmap]
            if len(keep) >= MIN_ROWS:
                yt = np.array([cmap[lab[i]] for i in keep])
                out["live_score"] = {"value": float(score("classification", trained.primary, yt, preds[keep])), "n": len(keep), "at_training": trained.test_primary, "primary": trained.primary}
        else:
            y = np.array([np.nan if is_missing(v) or to_float(v) is None else to_float(v) for v in data[trained.target]])
            keep = ~np.isnan(y)
            if keep.sum() >= MIN_ROWS:
                out["live_score"] = {"value": float(score("regression", "r2", y[keep], preds[keep])), "n": int(keep.sum()), "at_training": trained.test_primary, "primary": "r2"}
    return out
