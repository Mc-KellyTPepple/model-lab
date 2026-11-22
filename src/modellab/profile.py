"""What is in the spreadsheet: column kinds, data-quality warnings, a sensible target and task."""
from __future__ import annotations

import re
from collections import Counter

from .csvio import is_missing, to_float

ID_NAME = re.compile(r"(^|[_\s])(id|uuid|guid|index|idx|key|row|number|no)($|[_\s])|id$", re.I)
TARGET_NAMES = re.compile(r"^(target|label|class|outcome|y|churn\w*|survived|price|salary|diagnosis|species|default|fraud\w*|status|result)$", re.I)


def column_profile(name: str, values: list[str]) -> dict:
    n = len(values)
    present = [v for v in values if not is_missing(v)]
    miss = n - len(present)
    nums = [to_float(v) for v in present]
    numeric_share = sum(x is not None for x in nums) / max(len(present), 1)
    uniq = len(set(present))
    out = {"name": name, "missing": miss, "missing_pct": round(100 * miss / n, 1), "unique": uniq, "examples": list(dict.fromkeys(present))[:4]}
    if not present:
        return {**out, "kind": "empty"}
    if uniq == 1:
        return {**out, "kind": "constant"}
    if numeric_share >= 0.95:
        vals = [x for x in nums if x is not None]
        is_int = all(float(x).is_integer() for x in vals)
        sorted_ids = is_int and uniq >= 0.98 * len(present) and ID_NAME.search(name)
        if sorted_ids or (is_int and uniq == len(present) and n > 50 and (max(vals) - min(vals)) < 1.5 * n and ID_NAME.search(name)):
            return {**out, "kind": "id"}
        mean = sum(vals) / len(vals)
        return {**out, "kind": "numeric", "min": min(vals), "max": max(vals), "mean": round(mean, 4), "integer": is_int}
    avg_len = sum(len(v) for v in present) / len(present)
    if uniq >= 0.9 * len(present) and (ID_NAME.search(name) or avg_len > 8):
        return {**out, "kind": "id" if avg_len <= 40 else "text"}
    if avg_len > 40:
        return {**out, "kind": "text"}
    if uniq <= max(30, 0.05 * n):
        top = Counter(present).most_common(5)
        return {**out, "kind": "categorical", "top": [[k, c] for k, c in top]}
    return {**out, "kind": "highcard"}


def profile(header: list[str], rows: list[list[str]]) -> dict:
    cols = [column_profile(h, [r[i] for r in rows]) for i, h in enumerate(header)]
    seen, dup = set(), 0
    for r in rows:
        t = tuple(r)
        dup += t in seen
        seen.add(t)
    warnings = []
    if len(rows) < 100:
        warnings.append({"level": "warn", "text": f"Only {len(rows)} rows. Scores from so few rows are unstable, so treat them as rough."})
    if dup:
        warnings.append({"level": "info", "text": f"{dup} duplicate rows. Duplicates can make a score look better than it is."})
    for c in cols:
        if c["kind"] == "id":
            warnings.append({"level": "info", "text": f"'{c['name']}' looks like an ID. It is left out of training because an ID cannot predict anything new."})
        elif c["kind"] == "constant":
            warnings.append({"level": "info", "text": f"'{c['name']}' has the same value in every row, so it is left out."})
        elif c["kind"] in ("text", "highcard"):
            warnings.append({"level": "info", "text": f"'{c['name']}' has too many different values to use directly, so it is left out."})
        elif c["missing_pct"] >= 60:
            warnings.append({"level": "warn", "text": f"'{c['name']}' is {c['missing_pct']}% empty and is left out."})
    return {"rows": len(rows), "columns": cols, "duplicates": dup, "warnings": warnings, "suggested_target": suggest_target(cols)}


def suggest_target(cols: list[dict]) -> str | None:
    usable = [c for c in cols if c["kind"] in ("numeric", "categorical")]
    for c in usable:
        if TARGET_NAMES.match(c["name"]):
            return c["name"]
    return usable[-1]["name"] if usable else None


def detect_task(col: dict) -> str | None:
    """'classification' or 'regression' for a target column profile, or None if it cannot be a target."""
    if col["kind"] == "categorical":
        return "classification"
    if col["kind"] == "numeric":
        few = col["unique"] <= 10 and col.get("integer", False)
        return "classification" if few else "regression"
    return None
