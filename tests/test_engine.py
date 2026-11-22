import io, sys
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from modellab.csvio import DataError, parse_csv
from modellab.engine import predict_one, run


def to_csv(cols: dict) -> str:
    names = list(cols)
    n = len(next(iter(cols.values())))
    out = [",".join(names)]
    for i in range(n):
        out.append(",".join(str(cols[c][i]) for c in names))
    return "\n".join(out)


def test_real_signal_beats_baseline():
    df = load_breast_cancer(as_frame=True).frame
    r, _ = run(df.to_csv(index=False), target="target")
    assert r["verdict"]["beats_baseline"]
    assert r["test"]["primary"]["value"] > r["test"]["baseline"] + 0.2
    assert r["leaks"] == []


def test_noise_target_is_not_called_useful():
    rng = np.random.default_rng(0)
    n = 400
    cols = {f"x{i}": rng.normal(size=n).round(3) for i in range(6)}
    cols["cat"] = rng.choice(["a", "b", "c"], n)
    cols["y"] = rng.choice(["yes", "no"], n)
    r, _ = run(to_csv(cols), target="y")
    assert not r["verdict"]["beats_baseline"]
    assert any("baseline" in w["text"] for w in r["warnings"])


def test_planted_leak_is_caught_and_excluded():
    df = load_breast_cancer(as_frame=True).frame.copy()
    df["refund_issued"] = np.where(df["target"] == 1, "no", "yes")  # restates the answer
    r, _ = run(df.to_csv(index=False), target="target")
    assert [l["column"] for l in r["leaks"]] == ["refund_issued"]
    assert "refund_issued" not in [f["name"] for f in r["features"]]
    assert r["contrast"] and r["contrast"]["cv_with_leak"] > r["contrast"]["cv_without_leak"]


def test_regression_runs_and_reports_error():
    df = load_diabetes(as_frame=True).frame
    r, tr = run(df.to_csv(index=False), target="target")
    assert r["task"] == "regression"
    assert r["test"]["typical_error"] is not None
    out = predict_one(tr, {c: str(df[c].iloc[0]) for c in df.columns if c != "target"})
    assert isinstance(out["prediction"], float)


def test_small_data_warns():
    rng = np.random.default_rng(1)
    n = 40
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    cols = {"x": x.round(3), "z": z.round(3), "y": np.where(x + z + rng.normal(size=n) > 0, "up", "down")}
    r, _ = run(to_csv(cols), target="y")
    assert any("wide interval" in w["text"] for w in r["warnings"])


def test_too_few_rows_rejected():
    with pytest.raises(DataError):
        run("a,b\n1,2\n3,4\n", target="b")


def test_imbalance_uses_balanced_accuracy():
    rng = np.random.default_rng(2)
    n = 500
    cols = {"x": rng.normal(size=n).round(3), "y": np.where(rng.random(n) < 0.05, "rare", "common")}
    r, _ = run(to_csv(cols), target="y")
    assert r["primary"]["name"] == "balanced_accuracy"
    assert any("always says" in w["text"] or "balanced accuracy" in w["text"] for w in r["warnings"])


def test_predict_flags_out_of_range_and_unseen():
    df = load_breast_cancer(as_frame=True).frame
    _, tr = run(df.to_csv(index=False), target="target")
    vals = {c: str(df[c].iloc[0]) for c in df.columns if c != "target"}
    vals["mean radius"] = "9999"
    out = predict_one(tr, vals)
    assert any("outside the range" in n for n in out["notes"])


def test_deterministic():
    df = load_breast_cancer(as_frame=True).frame
    a, _ = run(df.to_csv(index=False), target="target")
    b, _ = run(df.to_csv(index=False), target="target")
    assert a["test"]["primary"] == b["test"]["primary"]
    assert a["winner"]["id"] == b["winner"]["id"]


def _noisy(n=500, seed=5):
    rng = np.random.default_rng(seed)
    cols = {f"x{i}": rng.normal(size=n).round(3) for i in range(4)}
    return rng, n, cols


def test_leak_caught_with_many_classes():
    rng, n, cols = _noisy()
    y = rng.choice(list("abcdefg"), n)
    cols["y"] = y
    flip = rng.random(n) < 0.01
    cols["recorded"] = np.where(flip, rng.permutation(y), y)
    r, _ = run(to_csv(cols), target="y")
    assert "recorded" in [l["column"] for l in r["leaks"]]


def test_leak_caught_for_a_number():
    rng, n, cols = _noisy()
    y = (cols["x0"] * 3 + rng.normal(size=n)).round(3)
    cols["y"] = y
    cols["recorded"] = (y * (1 + rng.normal(0, 0.01, n))).round(3)
    r, _ = run(to_csv(cols), target="y")
    assert "recorded" in [l["column"] for l in r["leaks"]]


def test_strong_but_honest_column_is_not_a_leak():
    rng, n, cols = _noisy()
    x = cols["x0"]
    cols["y"] = np.where(x + rng.normal(0, 0.8, n) > 0, "up", "down")
    r, _ = run(to_csv(cols), target="y")
    assert r["leaks"] == []
