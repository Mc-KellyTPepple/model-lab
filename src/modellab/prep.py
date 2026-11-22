"""Turning raw spreadsheet text into model inputs. Fitted on training rows only, then reused for prediction and drift checks."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .csvio import is_missing, to_float

OTHER, NA = "__other__", "__missing__"


def subset(data: dict[str, np.ndarray], idx) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def _floats(values) -> np.ndarray:
    return np.array([np.nan if is_missing(v) or (f := to_float(v)) is None else f for v in values], dtype=np.float64)


class Prep:
    def __init__(self, kinds: dict[str, str], max_levels: int = 15):
        self.kinds, self.max_levels = kinds, max_levels
        self.names: list[str] = []
        self.source: list[str] = []
        self.spec: dict[str, dict] = {}
        self.ref: dict[str, dict] = {}

    def fit(self, data: dict[str, np.ndarray]) -> "Prep":
        self.names, self.source, self.spec, self.ref = [], [], {}, {}
        for col, kind in self.kinds.items():
            raw = data[col]
            if kind == "numeric":
                x = _floats(raw)
                ok = x[~np.isnan(x)]
                med = float(np.median(ok)) if len(ok) else 0.0
                sd = float(ok.std()) if len(ok) > 1 and ok.std() > 0 else 1.0
                edges = np.unique(np.quantile(ok, np.linspace(0.1, 0.9, 9))) if len(ok) > 9 else np.array([med])
                spec = {"median": med, "mean": float(ok.mean()) if len(ok) else 0.0, "std": sd, "missing": bool(np.isnan(x).any()),
                        "min": float(ok.min()) if len(ok) else 0.0, "max": float(ok.max()) if len(ok) else 0.0, "edges": edges.tolist()}
                self.spec[col] = spec
                self.names.append(col); self.source.append(col)
                if spec["missing"]:
                    self.names.append(f"{col} (missing)"); self.source.append(col)
                self.ref[col] = {"kind": "numeric", "edges": spec["edges"], "frac": _bin_frac(np.where(np.isnan(x), med, x), edges)}
            else:
                vals = [NA if is_missing(v) else v.strip() for v in raw]
                common = [k for k, _ in Counter(vals).most_common(self.max_levels)]
                levels = [k for k in common if k != NA]
                self.spec[col] = {"levels": levels, "missing": NA in vals}
                for lv in levels:
                    self.names.append(f"{col} = {lv}"); self.source.append(col)
                self.names.append(f"{col} = other/unseen"); self.source.append(col)
                if NA in vals:
                    self.names.append(f"{col} (missing)"); self.source.append(col)
                cnt = Counter(vals)
                tot = len(vals)
                fr = {lv: cnt.get(lv, 0) / tot for lv in levels}
                fr[OTHER] = max(0.0, 1 - sum(fr.values()) - cnt.get(NA, 0) / tot)
                if NA in cnt:
                    fr[NA] = cnt[NA] / tot
                self.ref[col] = {"kind": "categorical", "frac": fr}
        return self

    def transform(self, data: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(data.values())))
        parts = []
        for col, kind in self.kinds.items():
            raw, s = data[col], self.spec[col]
            if kind == "numeric":
                x = _floats(raw)
                miss = np.isnan(x)
                parts.append(((np.where(miss, s["median"], x) - s["mean"]) / s["std"]).reshape(n, 1))
                if s["missing"]:
                    parts.append(miss.astype(np.float64).reshape(n, 1))
            else:
                vals = np.array([NA if is_missing(v) else v.strip() for v in raw], dtype=object)
                block = np.zeros((n, len(s["levels"]) + 1 + (1 if s["missing"] else 0)))
                for j, lv in enumerate(s["levels"]):
                    block[:, j] = vals == lv
                known = np.isin(vals, s["levels"] + [NA])
                block[:, len(s["levels"])] = ~known
                if s["missing"]:
                    block[:, -1] = vals == NA
                parts.append(block)
        return np.hstack(parts) if parts else np.zeros((n, 0))


def _bin_frac(x: np.ndarray, edges: np.ndarray) -> list[float]:
    idx = np.searchsorted(edges, x, side="right")
    cnt = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return (cnt / max(cnt.sum(), 1)).tolist()
