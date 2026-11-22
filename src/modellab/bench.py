"""Does Model Lab's honesty hold up on data it has never seen? Four measurable claims, on public datasets.

For every dataset and split seed, 30% of the rows are kept back by this script (the Lab never sees them), and the
rest is given to the Lab exactly like an upload. Then:

  1. Pick quality     the Lab's model and every candidate are scored on the kept-back rows;
                      regret = best candidate's score minus the Lab's score.
  2. Stated interval  how far the Lab's own test score was from the kept-back score, and how often its
                      95% interval contained it. (The kept-back score is itself a noisy sample, so
                      coverage below 95% is expected; the typical gap is the number to read.)
  3. False alarms     the answer column is shuffled so no signal exists; the Lab should NOT say it beats the baseline.
  4. Leak catching    a copy of the answer (1% of values flipped) is added as a column; the Lab should flag it.
                      On the unmodified data, any flagged column is counted as a false flag.

Datasets that cannot be downloaded are listed as skipped; nothing is filled in.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from .engine import predict_rows, run
from .metrics import SEED, lineup, primary_for, score
from .prep import Prep, subset

OFFLINE = ["wine", "breast_cancer", "diabetes"]
OPENML = [  # (name used in results, openml name, task, target column or None for default)
    ("credit-g", "credit-g", "classification", None),
    ("diabetes-pima", "diabetes", "classification", None),
    ("blood-transfusion", "blood-transfusion-service-center", "classification", None),
    ("phoneme", "phoneme", "classification", None),
    ("spambase", "spambase", "classification", None),
    ("kc1", "kc1", "classification", None),
    ("ilpd", "ilpd", "classification", None),
    ("qsar-biodeg", "qsar-biodeg", "classification", None),
    ("banknote", "banknote-authentication", "classification", None),
    ("vehicle", "vehicle", "classification", None),
    ("segment", "segment", "classification", None),
    ("abalone", "abalone", "regression", None),
    ("california-housing", "__sklearn_california__", "regression", None),
]
MAX_ROWS = 5000
SEEDS = (1, 2, 3)


def _frames_offline(name):
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
    d = {"wine": load_wine, "breast_cancer": load_breast_cancer, "diabetes": load_diabetes}[name](as_frame=True)
    df = d.frame.copy()
    if name != "diabetes":
        df["target"] = [d.target_names[t] for t in d.target]
    return df, "target", "regression" if name == "diabetes" else "classification"


def _frame_openml(spec):
    from sklearn.datasets import fetch_california_housing, fetch_openml
    name, oml, task, _ = spec
    if oml == "__sklearn_california__":
        d = fetch_california_housing(as_frame=True)
        return d.frame, d.frame.columns[-1], task
    d = fetch_openml(oml, version=1, as_frame=True, parser="auto")
    df = d.frame.copy()
    tcol = d.target.name if hasattr(d.target, "name") and d.target.name else df.columns[-1]
    return df, tcol, task


def _to_csv(df) -> str:
    return df.to_csv(index=False)


def _outer_scores(trained, df_train, df_out, target, result, task):
    """Score the Lab's own model and each candidate (refit on the same rows) on the kept-back rows."""
    from .csvio import parse_csv
    header, rows = parse_csv(_to_csv(df_out))
    out = {h: np.array([r[i] for r in rows], dtype=object) for i, h in enumerate(header)}
    prim = trained.primary
    if task == "classification":
        from .engine import _norm_label
        cmap = {k: i for i, k in enumerate(trained.classes)}
        lab = [_norm_label(v) for v in out[target]]
        keep = np.array([v in cmap for v in lab])
        y = np.array([cmap.get(v, -1) for v in lab])[keep]
    else:
        y = np.array([float(v) for v in out[target]])
        keep = np.ones(len(y), bool)
    sub = {c: out[c][keep] for c in trained.prep.kinds}
    lab_score = float(score(task, prim, y, predict_rows(trained, sub)))
    header2, rows2 = parse_csv(_to_csv(df_train))
    tr = {h: np.array([r[i] for r in rows2], dtype=object) for i, h in enumerate(header2)}
    if task == "classification":
        ytr = np.array([cmap.get(_norm_label(v), -1) for v in tr[target]])
        ok = ytr >= 0
    else:
        ytr = np.array([float(v) for v in tr[target]])
        ok = np.ones(len(ytr), bool)
    idx = np.flatnonzero(ok)
    if len(idx) > 2000:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, 2000, replace=False))
    prep = Prep(dict(trained.prep.kinds)).fit(subset(tr, idx))
    Xtr, Xo = prep.transform(subset(tr, idx)), prep.transform(sub)
    cands = {}
    for mid, name, _, make in lineup(task):
        try:
            cands[mid] = float(score(task, prim, y, make().fit(Xtr, ytr[idx]).predict(Xo)))
        except Exception:
            pass
    return lab_score, cands


def one_dataset(name, df, target, task, seeds=SEEDS, log=print):
    rec = {"dataset": name, "task": task, "rows": int(len(df)), "columns": int(df.shape[1]), "runs": []}
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    for seed in seeds:
        t0 = time.time()
        out = df.sample(frac=0.3, random_state=seed)
        trn = df.drop(out.index)
        run_rec = {"seed": seed}
        try:
            res, trained = run(_to_csv(trn), target=target, task=task)
            lab, cands = _outer_scores(trained, trn, out, target, res, task)
            best = max(cands.values())
            t = res["test"]["primary"]
            run_rec.update({
                "winner": res["winner"]["id"], "claimed": t["value"], "lo": t["lo"], "hi": t["hi"],
                "outer": lab, "baseline_outer": cands.get("baseline"), "best_candidate_outer": best,
                "regret": best - lab, "gap": abs(t["value"] - lab), "covered": bool(t["lo"] <= lab <= t["hi"]),
                "says_beats_baseline": bool(res["verdict"]["beats_baseline"]),
                "really_beats_baseline": bool(lab > (cands.get("baseline") or 0) + 1e-9),
                "false_leak_flags": [l["column"] for l in res["leaks"]],
            })
            # 3. shuffled target: no signal exists
            fa = []
            for k in range(3):
                sh = trn.copy()
                sh[target] = np.random.default_rng(100 * seed + k).permutation(sh[target].to_numpy())
                try:
                    r2, _ = run(_to_csv(sh), target=target, task=task)
                    fa.append(bool(r2["verdict"]["beats_baseline"]))
                except Exception:
                    pass
            run_rec["shuffled"] = {"runs": len(fa), "false_alarms": int(sum(fa))}
            # 4. planted leak
            lk = trn.copy()
            rng = np.random.default_rng(seed)
            if task == "classification":
                vals = lk[target].astype(str).to_numpy()
                flip = rng.random(len(vals)) < 0.01
                alt = rng.permutation(vals)
                lk["recorded_outcome"] = np.where(flip, alt, vals)
            else:
                y = lk[target].astype(float).to_numpy()
                lk["recorded_outcome"] = y * (1 + rng.normal(0, 0.01, len(y)))
            try:
                r3, _ = run(_to_csv(lk), target=target, task=task)
                run_rec["leak_caught"] = "recorded_outcome" in [l["column"] for l in r3["leaks"]]
            except Exception as exc:
                run_rec["leak_caught"] = None
                run_rec["leak_error"] = type(exc).__name__
        except Exception as exc:
            run_rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        run_rec["seconds"] = round(time.time() - t0, 1)
        rec["runs"].append(run_rec)
        log(f"  {name} seed {seed}: " + (run_rec.get("error") or f"claimed {run_rec['claimed']:.3f} outer {run_rec['outer']:.3f} regret {run_rec['regret']:.3f} leak {run_rec['leak_caught']}"))
    return rec


def summarise(records: list[dict]) -> dict:
    runs = [r for rec in records for r in rec["runs"] if "claimed" in r]
    if not runs:
        return {}
    sh = [r["shuffled"] for r in runs if r.get("shuffled")]
    lk = [r["leak_caught"] for r in runs if r.get("leak_caught") is not None]
    regs = np.array([r["regret"] for r in runs])
    gaps = np.array([r["gap"] for r in runs])
    return {
        "datasets": len([rec for rec in records if any("claimed" in r for r in rec["runs"])]),
        "runs": len(runs),
        "regret_median": float(np.median(regs)), "regret_mean": float(regs.mean()),
        "regret_within_2pts": float((regs <= 0.02).mean()),
        "gap_median": float(np.median(gaps)), "coverage": float(np.mean([r["covered"] for r in runs])),
        "false_alarms": int(sum(s["false_alarms"] for s in sh)), "shuffled_runs": int(sum(s["runs"] for s in sh)),
        "leak_caught": int(sum(lk)), "leak_runs": len(lk),
        "false_leak_flags": int(sum(1 for r in runs if r["false_leak_flags"])), "clean_runs": len(runs),
        "says_useful_and_is": int(sum(1 for r in runs if r["says_beats_baseline"] and r["really_beats_baseline"])),
        "says_useful_but_isnt": int(sum(1 for r in runs if r["says_beats_baseline"] and not r["really_beats_baseline"])),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="modellab-bench")
    ap.add_argument("--offline", action="store_true", help="use scikit-learn's bundled datasets only (for checking the harness)")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, skipped = [], []
    if args.offline:
        for n in OFFLINE:
            df, t, task = _frames_offline(n)
            records.append(one_dataset(n, df, t, task, seeds))
    else:
        for spec in OPENML:
            try:
                df, t, task = _frame_openml(spec)
            except Exception as exc:
                skipped.append({"dataset": spec[0], "reason": f"{type(exc).__name__}: {str(exc)[:100]}"})
                print(f"  skipped {spec[0]}: {exc}")
                continue
            records.append(one_dataset(spec[0], df, t, task, seeds))
            (out / "datasets.json").write_text(json.dumps({"records": records, "skipped": skipped}, indent=1))
    summary = summarise(records)
    (out / "datasets.json").write_text(json.dumps({"records": records, "skipped": skipped}, indent=1))
    (out / "summary.json").write_text(json.dumps({"summary": summary, "skipped": skipped, "seeds": list(seeds)}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
