import json, threading, time, urllib.request
from modellab import samples
from modellab.server import App, Limits, Server, make_handler


def _serve(app):
    srv = Server(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _req(url, data=None, ctype="application/json"):
    r = urllib.request.Request(url, data=data, headers={"Content-Type": ctype} if data is not None else {})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, json.loads(f.read() or b"{}") if "json" in f.headers.get("Content-Type", "") else f.read()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait(base, jid):
    for _ in range(300):
        code, j = _req(f"{base}/jobs/{jid}")
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("job never finished")


def test_full_flow_train_predict_monitor():
    app = App(Limits(50, 500))
    srv, base = _serve(app)
    try:
        assert _req(f"{base}/health")[1] == {"ok": True}
        code, out = _req(f"{base}/train?target=churned", json.dumps({"sample": "churn_leaky"}).encode())
        assert code == 202
        j = _wait(base, out["job"])
        assert j["status"] == "done", j
        res = j["result"]
        assert res["leaks"][0]["column"] == "account_status"
        vals = {f["name"]: (f["levels"][0] if f["kind"] == "categorical" else str(f["median"])) for f in res["form"]}
        code, p = _req(f"{base}/predict", json.dumps({"session": res["session"], "values": vals}).encode())
        assert code == 200 and p["prediction"] in ("yes", "no")
        code, m = _req(f"{base}/monitor?session={res['session']}", json.dumps({"sample": "churn_later"}).encode())
        assert code == 200 and m["verdict"] == "large"
        assert _req(f"{base}/predict", json.dumps({"session": "nope", "values": {}}).encode())[0] == 410
    finally:
        srv.shutdown()


def test_bad_inputs_and_limits():
    app = App(Limits(1, 500))
    srv, base = _serve(app)
    try:
        assert _req(f"{base}/train", b"a,b\n1,2\n", "text/csv")[0] == 202   # accepted, fails in the job
        code, out = _req(f"{base}/train", b"a,b\n1,2\n", "text/csv")
        assert code == 429
        assert _req(f"{base}/train", json.dumps({"sample": "../etc"}).encode())[0] == 400
        assert _req(f"{base}/jobs/unknown")[0] == 404
        code, body = _req(f"{base}/samples/wine.csv")
        assert code == 200 and body.startswith(b"alcohol")
    finally:
        srv.shutdown()
