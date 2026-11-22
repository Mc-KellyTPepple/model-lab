"""Built-in practice datasets, generated on demand (nothing is stored in the repo).

Real: wine, breast_cancer, diabetes (scikit-learn's bundled copies of public UCI/Efron data).
Synthetic (labelled as such everywhere): churn, plus two variants made to teach a lesson:
  churn_leaky   has a column that restates the answer, so the leak scan must catch it
  churn_later   the same customers a year on, shifted, for the drift monitor
"""
from __future__ import annotations

import io
import csv

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

SAMPLES = {
    "wine": {"title": "Wine cultivar", "kind": "real", "target": "cultivar", "blurb": "178 wines, 13 lab measurements, 3 grape cultivars (UCI Wine)."},
    "breast_cancer": {"title": "Tumour type", "kind": "real", "target": "diagnosis", "blurb": "569 cell-nucleus measurements, benign or malignant (UCI Wisconsin Diagnostic)."},
    "diabetes": {"title": "Diabetes progression", "kind": "real", "target": "progression", "blurb": "442 patients, 10 baseline measures, disease progression a year later (Efron et al.)."},
    "churn": {"title": "Customer churn", "kind": "synthetic", "target": "churned", "blurb": "1,500 invented subscribers; who cancels depends on tenure, contract and support calls."},
    "churn_leaky": {"title": "Churn with a leak", "kind": "synthetic", "target": "churned", "blurb": "Same data plus an 'account_status' column that gives the answer away. Watch it get caught."},
    "churn_later": {"title": "Churn, a year later", "kind": "synthetic", "target": "churned", "blurb": "Newer customers with a different mix. Use it in Monitor after training on 'churn'."},
}


def _csv(header: list[str], rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def _churn(n: int, seed: int, shift: bool = False, leak: bool = False) -> str:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.35, 0.3, 0.35] if shift else [0.55, 0.25, 0.2])
    tenure = np.clip(rng.gamma(2.0, 11 if not shift else 7, n), 1, 72).round().astype(int)
    charge = (rng.normal(62 if not shift else 78, 18, n)).clip(20, 130).round(2)
    calls = rng.poisson(1.2 if not shift else 1.8, n)
    pay = rng.choice(["card", "bank transfer", "mailed cheque", "e-wallet"], n)
    region = rng.choice(["north", "south", "east", "west"], n)
    z = (-0.6 + 1.3 * (contract == "month-to-month") - 1.0 * (contract == "two-year") - 0.035 * tenure
         + 0.012 * (charge - 60) + 0.35 * calls + 0.4 * (pay == "mailed cheque"))
    churn = rng.random(n) < 1 / (1 + np.exp(-z))
    hdr = ["customer_id", "tenure_months", "monthly_charge", "support_calls", "contract", "payment_method", "region"]
    rows = []
    status = np.where(churn, "closed", "active")
    flip = rng.random(n) < 0.01
    status = np.where(flip, np.where(status == "closed", "active", "closed"), status)
    for i in range(n):
        r = [f"C{seed % 10}{i:05d}", tenure[i], charge[i], calls[i], contract[i], pay[i], region[i]]
        if leak:
            r.append(status[i])
        r.append("yes" if churn[i] else "no")
        rows.append(r)
    return _csv(hdr + (["account_status"] if leak else []) + ["churned"], rows)


def get(name: str) -> str:
    if name == "churn":
        return _churn(1500, 11)
    if name == "churn_leaky":
        return _churn(1500, 11, leak=True)
    if name == "churn_later":
        return _churn(600, 29, shift=True)
    if name == "wine":
        d = load_wine()
        return _csv(list(d.feature_names) + ["cultivar"], [list(map(float, x)) + [d.target_names[t]] for x, t in zip(d.data, d.target)])
    if name == "breast_cancer":
        d = load_breast_cancer()
        return _csv(list(d.feature_names) + ["diagnosis"], [list(map(float, x)) + [d.target_names[t]] for x, t in zip(d.data, d.target)])
    if name == "diabetes":
        d = load_diabetes()
        return _csv(list(d.feature_names) + ["progression"], [list(map(float, x)) + [float(t)] for x, t in zip(d.data, d.target)])
    raise KeyError(name)
