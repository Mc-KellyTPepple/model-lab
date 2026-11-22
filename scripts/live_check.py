"""Train each built-in sample on the LIVE service and record how long it takes, as a visitor would experience it.

Usage: python scripts/live_check.py https://your-service.onrender.com [out.json]
Standard library only. Run manually (see .github/workflows/live.yml); it is not scheduled.
"""
import json
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
out = sys.argv[2] if len(sys.argv) > 2 else "evals/results/live.json"
SAMPLES = [("wine", "cultivar"), ("breast_cancer", "diagnosis"), ("diabetes", "progression"), ("churn", "churned"), ("churn_leaky", "churned")]


def call(path, data=None, ctype="application/json", timeout=120):
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": ctype} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


t0 = time.time()
call("/health", timeout=180)
wake = round(time.time() - t0, 1)
rows = []
for name, target in SAMPLES:
    t = time.time()
    rec = {"sample": name}
    try:
        job = call(f"/train?target={target}", json.dumps({"sample": name}).encode())["job"]
        while True:
            j = call(f"/jobs/{job}")
            if j["status"] in ("done", "error"):
                break
            if time.time() - t > 600:
                raise TimeoutError("over 10 minutes")
            time.sleep(1)
        rec["seconds"] = round(time.time() - t, 1)
        rec["status"] = j["status"]
        if j["status"] == "done":
            rec["server_seconds"] = j["result"]["seconds"]
            rec["rows"] = j["result"]["rows"]["used"]
            rec["skipped_models"] = [m["name"] for m in j["result"]["models"] if m.get("skipped")]
        else:
            rec["error"] = j.get("error")
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - t, 1))
    rows.append(rec)
    print(rec, flush=True)
    time.sleep(2)
json.dump({"service": base, "first_health_seconds": wake, "runs": rows, "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}, open(out, "w"), indent=1)
