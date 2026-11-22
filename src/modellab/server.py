"""Small HTTP service (standard library + the engine).

  GET  /health
  GET  /samples                         list of practice datasets
  GET  /samples/<id>.csv                one of them
  POST /profile                         body = CSV            -> columns, warnings, suggested target
  POST /train?target=&task=&leaks=keep  body = CSV or {"sample": id}  -> {"job": id}
  GET  /jobs/<id>                       -> status, stage, percent, message, and the result when done
  POST /predict                         {"session": id, "values": {...}}
  POST /monitor?session=id              body = CSV or {"sample": id}  -> drift report

Uploaded files are held in memory only, never written to disk, and forgotten after an hour. Nothing is logged.
One training runs at a time (the host is small); others wait in a queue and see their position.
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from collections import OrderedDict, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from . import samples
from .csvio import MAX_BYTES, DataError, parse_csv
from .drift import check as drift_check
from .engine import predict_one, run
from .profile import detect_task, profile

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
SESSION_TTL = 3600
MAX_SESSIONS = 20
MAX_JOBS = 60
QUEUE_LIMIT = 8


class Limits:
    def __init__(self, per_ip_per_hour: int = 10, daily_cap: int = 300, clock=time.time):
        self.per_ip, self.daily_cap, self.clock = per_ip_per_hour, daily_cap, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.count = "", 0
        self.lock = threading.Lock()

    def allow(self, ip: str) -> str | None:
        with self.lock:
            now = self.clock()
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.daily_cap:
                return "The daily limit for this demo is used up. Please try again tomorrow."
            q = self.hits[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.per_ip:
                return f"Limit: at most {self.per_ip} training runs per hour per visitor."
            q.append(now)
            self.count += 1
            return None


class App:
    def __init__(self, limits: Limits | None = None, budget: float | None = None, max_rows: int | None = None):
        self.limits = limits or Limits()
        self.budget = budget or float(os.environ.get("MODELLAB_BUDGET", 75))
        self.max_rows = max_rows or int(os.environ.get("MODELLAB_MAX_ROWS", 2000))
        self.sessions: OrderedDict[str, object] = OrderedDict()
        self.jobs: OrderedDict[str, dict] = OrderedDict()
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- jobs ----
    def _worker(self) -> None:
        while True:
            jid, text, kw = self.q.get()
            job = self.jobs.get(jid)
            if job is None:
                continue
            job.update(status="running", stage="read", pct=1, message="Starting")

            def progress(stage, pct, msg, job=job):
                job.update(stage=stage, pct=pct, message=msg)

            try:
                result, trained = run(text, progress=progress, time_budget=self.budget, max_train_rows=self.max_rows, **kw)
                sid = secrets.token_urlsafe(9)
                with self.lock:
                    self.sessions[sid] = trained
                    while len(self.sessions) > MAX_SESSIONS:
                        self.sessions.popitem(last=False)
                result["session"] = sid
                job.update(status="done", pct=100, result=result)
            except DataError as exc:
                job.update(status="error", error=str(exc))
            except Exception as exc:
                print(f"train failed: {type(exc).__name__}", flush=True)
                job.update(status="error", error="Something went wrong while training. Try a smaller or cleaner file.")
            finally:
                job["finished"] = time.time()
                job.pop("text", None)

    def _resolve_text(self, body: bytes, ctype: str) -> tuple[str | None, str | None]:
        """CSV text from either a raw CSV body or {"sample": id}."""
        if "json" in ctype:
            try:
                obj = json.loads(body or b"{}")
            except ValueError:
                return None, "Bad request."
            name = obj.get("sample") if isinstance(obj, dict) else None
            if name not in samples.SAMPLES:
                return None, "Unknown sample."
            return samples.get(name), None
        if len(body) > MAX_BYTES:
            return None, f"File too large (max {MAX_BYTES // 1_000_000} MB)."
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return body.decode(enc), None
            except UnicodeDecodeError:
                continue
        return None, "Could not read that file as text."

    def train(self, body: bytes, ctype: str, query: dict, ip: str):
        text, err = self._resolve_text(body, ctype)
        if err:
            return 400, {"error": err}
        if self.q.qsize() >= QUEUE_LIMIT:
            return 503, {"error": "Lots of people are training models right now. Try again in a minute."}
        refused = self.limits.allow(ip)
        if refused:
            return 429, {"error": refused}
        kw = {"target": (query.get("target") or [None])[0], "task": (query.get("task") or [None])[0],
              "include_suspected": (query.get("leaks") or [""])[0] == "keep"}
        if kw["task"] not in (None, "classification", "regression"):
            return 400, {"error": "Unknown task."}
        jid = secrets.token_urlsafe(9)
        with self.lock:
            self.jobs[jid] = {"status": "queued", "stage": "queued", "pct": 0, "message": "Waiting for a turn", "created": time.time()}
            while len(self.jobs) > MAX_JOBS:
                self.jobs.popitem(last=False)
        self.q.put((jid, text, kw))
        return 202, {"job": jid}

    def job(self, jid: str):
        j = self.jobs.get(jid)
        if j is None:
            return 404, {"error": "That run has expired. Start a new one."}
        out = {k: j[k] for k in ("status", "stage", "pct", "message") if k in j}
        if j["status"] == "queued":
            pos = 1 + sum(1 for k, v in self.jobs.items() if v["status"] in ("queued", "running") and v["created"] < j["created"])
            out["position"] = pos
        if j["status"] == "done":
            out["result"] = j["result"]
        if j["status"] == "error":
            out["error"] = j["error"]
        return 200, out

    # ---- other routes ----
    def profile(self, body: bytes, ctype: str):
        text, err = self._resolve_text(body, ctype)
        if err:
            return 400, {"error": err}
        try:
            header, rows = parse_csv(text)
            p = profile(header, rows)
        except DataError as exc:
            return 400, {"error": str(exc)}
        for c in p["columns"]:
            c["task"] = detect_task(c)
        return 200, p

    def _session(self, sid: str | None):
        with self.lock:
            t = self.sessions.get(sid)
            if t is None or time.time() - t.created > SESSION_TTL:
                self.sessions.pop(sid, None)
                return None
            self.sessions.move_to_end(sid)
            return t

    def predict(self, obj):
        t = self._session(obj.get("session") if isinstance(obj, dict) else None)
        if t is None:
            return 410, {"error": "This model has expired (models are kept for an hour). Train it again."}
        vals = obj.get("values")
        if not isinstance(vals, dict):
            return 400, {"error": "Send the values to predict from."}
        return 200, predict_one(t, {str(k): str(v)[:100] for k, v in vals.items()})

    def monitor(self, body: bytes, ctype: str, sid: str | None):
        t = self._session(sid)
        if t is None:
            return 410, {"error": "This model has expired (models are kept for an hour). Train it again."}
        text, err = self._resolve_text(body, ctype)
        if err:
            return 400, {"error": err}
        try:
            return 200, drift_check(t, text)
        except DataError as exc:
            return 400, {"error": str(exc)}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            route = self.path.partition("?")[0]
            if route == "/health":
                return self._json(200, {"ok": True})
            if route == "/samples":
                return self._json(200, {"samples": [{"id": k, **v} for k, v in samples.SAMPLES.items()]})
            if route.startswith("/samples/") and route.endswith(".csv"):
                name = route[len("/samples/"):-4]
                if name in samples.SAMPLES:
                    return self._send(200, samples.get(name).encode(), "text/csv; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{name}.csv"'})
            if route.startswith("/jobs/"):
                return self._json(*app.job(route[len("/jobs/"):]))
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            route, _, qs = self.path.partition("?")
            query = parse_qs(qs)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._json(400, {"error": "bad request"})
            if n > MAX_BYTES + 10_000:
                return self._json(413, {"error": f"File too large (max {MAX_BYTES // 1_000_000} MB)."})
            body = self.rfile.read(n)
            ctype = (self.headers.get("Content-Type") or "").lower()
            fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            ip = fwd or self.client_address[0]
            try:
                if route == "/train":
                    return self._json(*app.train(body, ctype, query, ip))
                if route == "/profile":
                    return self._json(*app.profile(body, ctype))
                if route == "/predict":
                    try:
                        obj = json.loads(body or b"{}")
                    except ValueError:
                        return self._json(400, {"error": "bad request"})
                    return self._json(*app.predict(obj))
                if route == "/monitor":
                    return self._json(*app.monitor(body, ctype, (query.get("session") or [None])[0]))
            except Exception as exc:  # never leak internals
                print(f"{route} failed: {type(exc).__name__}", flush=True)
                return self._json(500, {"error": "Something went wrong. Try again."})
            self._json(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    return Handler


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def serve(app: App | None = None, host: str = "0.0.0.0", port: int | None = None) -> None:
    port = port or int(os.environ.get("PORT", "7860"))
    print(f"listening on {host}:{port}", flush=True)
    Server((host, port), make_handler(app or App())).serve_forever()
