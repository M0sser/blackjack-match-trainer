"""The local server: a small JSON API over the practice match and the review, plus the page.

It listens on this computer only (127.0.0.1) and makes no network requests.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import traceback
import urllib.request
import webbrowser
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trainer.brain import Brain, BrainConfig
from trainer.errors import TrainerError
from trainer.history import save_review_form, timestamp, write_record
from trainer.opponents import DEFAULT_RULE_WEIGHTS, DEFAULT_STYLE_WEIGHTS
from trainer.practice import PracticeMatch
from trainer.review import analyze_match, check_rounds, recommend_bet, recommend_play, save_review
from trainer.settings import MODES, Settings, load_settings, population_from, save_settings

WEB_DIR = Path(__file__).resolve().parent / "web"
PORTS = range(8721, 8727)
MAX_BODY = 2_000_000
NOT_READY = "Still working out the best bets. This takes a few minutes the first time."
CRASHED = "Something went wrong inside the trainer."
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                 ".ico": "image/x-icon"}


class _NotReady(Exception):
    pass


def _step(body: dict):
    """The step the page was showing when the click was made, if it said."""
    step = body.get("step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
        raise TrainerError("That request was not understood.")
    return step


def _amount(body: dict, name: str, default=None) -> float:
    value = body.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainerError("A bet has to be a number.")
    return float(value)


class App:
    """Everything the server holds: the settings, the brain (once solved) and the match in play."""

    def __init__(self, data_dir, config: BrainConfig = BrainConfig()):
        self.data_dir = Path(data_dir)
        self.config = config
        self.settings = load_settings(self.data_dir)
        self.lock = threading.RLock()
        self.brain: Brain | None = None
        self.match: PracticeMatch | None = None
        self._practice_record: dict | None = None
        self.status = {"ready": False, "progress": 0.0, "message": "Getting ready"}
        self._solve_number = 0

    # ---- the brain --------------------------------------------------------------

    def set_brain(self, brain: Brain) -> None:
        with self.lock:
            self.brain = brain
            self.status = {"ready": True, "progress": 1.0, "message": "Ready"}

    def start_brain(self) -> None:
        """Solve (or load) the brain for the current settings in the background."""
        with self.lock:
            self._solve_number += 1
            number, settings = self._solve_number, self.settings
            self.status = {"ready": False, "progress": 0.0, "message": "Getting ready"}
        threading.Thread(target=self._solve, args=(number, settings), daemon=True).start()

    def _solve(self, number: int, settings: Settings) -> None:
        def progress(fraction: float, message: str) -> None:
            with self.lock:
                if number == self._solve_number:
                    self.status = {"ready": False, "progress": round(float(fraction), 3), "message": message}

        try:
            brain = Brain.prepare(settings.rules, population_from(settings), self.data_dir / "cache",
                                  self.config, progress)
        except Exception:
            traceback.print_exc()
            with self.lock:
                if number == self._solve_number:
                    self.status = {"ready": False, "progress": 0.0,
                                   "message": "Could not work out the best bets. See the trainer's window."}
            return
        with self.lock:
            if number == self._solve_number:        # a newer solve has not replaced this one
                self.brain = brain
                self.status = {"ready": True, "progress": 1.0, "message": "Ready"}

    def _ready_brain(self) -> Brain:
        if self.brain is None:
            raise _NotReady()
        return self.brain

    # ---- requests ---------------------------------------------------------------

    def handle(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        """Answer one API request as (status code, JSON-ready dict)."""
        try:
            if body is not None and not isinstance(body, dict):
                raise TrainerError("That request was not understood.")
            return self._route(method, path, body or {})
        except TrainerError as err:
            return 400, {"error": str(err)}
        except _NotReady:
            return 503, {"error": NOT_READY}
        except Exception:
            traceback.print_exc()
            return 500, {"error": CRASHED}

    def _route(self, method: str, path: str, body: dict) -> tuple[int, dict]:
        route = (method, path)
        if route == ("GET", "/api/status"):
            with self.lock:
                return 200, {"brain": dict(self.status), "settings": self.settings.to_dict(),
                             "default_rule_weights": dict(DEFAULT_RULE_WEIGHTS),
                             "default_style_weights": dict(DEFAULT_STYLE_WEIGHTS)}
        if route == ("POST", "/api/practice/start"):
            return 200, self._start_match(body)
        if route == ("GET", "/api/practice/state"):
            with self.lock:
                if self.match is None:
                    return 404, {"error": "No match is running."}
                return 200, self._save_practice(self.match.state())
        if route == ("POST", "/api/practice/bet"):
            main, pp, fct = _amount(body, "main"), _amount(body, "pp", 0), _amount(body, "fct", 0)
            with self.lock:
                return 200, self._save_practice(self._match().bet(main, pp, fct, step=_step(body)), "bet", body)
        if route == ("POST", "/api/practice/action"):
            with self.lock:
                return 200, self._save_practice(self._match().act(str(body.get("action", "")), step=_step(body)), "action", body)
        if route == ("POST", "/api/practice/next"):
            with self.lock:
                return 200, self._save_practice(self._match().next_round(step=_step(body)), "next", body)
        if route == ("POST", "/api/review/save"):
            with self.lock:
                rounds = self._rounds(body)
                rules = self._rules()
                checks = check_rounds(rules, rounds)
                complete = False
                for i, check in enumerate(checks):
                    if not check["ok"] or not check["complete"]:
                        break
                    if i + 1 == rules.rounds or rules.floor_bet(check["chips_after"]) < rules.min_bet:
                        complete = True
                        break
                path = save_review_form(self.data_dir, body.get("review_id"), body.get("form"),
                                        complete=complete, settings=self.settings.to_dict())
            return 200, {"saved": True, "file": "data/reviews/" + path.name}
        if route == ("POST", "/api/review/check"):
            return 200, {"rounds": check_rounds(self._rules(), self._rounds(body))}
        if route == ("POST", "/api/review/best-bet"):
            with self.lock:
                brain = self._ready_brain()
            return 200, recommend_bet(brain.rules, brain, self._rounds(body))
        if route == ("POST", "/api/review/best-play"):
            with self.lock:
                brain = self._ready_brain()
            return 200, recommend_play(brain.rules, brain, self._rounds(body), body.get("current"))
        if route == ("POST", "/api/review/analyze"):
            with self.lock:
                brain = self._ready_brain()
            report = analyze_match(brain.rules, brain, self._rounds(body))
            with self.lock:
                save_review(self.data_dir, body, report)
            return 200, report
        if route == ("POST", "/api/settings"):
            return 200, self._change_settings(body)
        return 404, {"error": "Not found."}

    def _rules(self):
        with self.lock:
            return self.brain.rules if self.brain is not None else self.settings.rules

    @staticmethod
    def _rounds(body: dict) -> list:
        rounds = body.get("rounds")
        if not isinstance(rounds, list):
            raise TrainerError("The match was not understood: it needs a list of rounds.")
        return rounds

    def _match(self) -> PracticeMatch:
        if self.match is None:
            raise TrainerError("Start a match first.")
        return self.match

    def _start_match(self, body: dict) -> dict:
        mode = body.get("mode", self.settings.default_mode)
        if mode not in MODES:
            raise TrainerError("The mode has to be Coach or Match.")
        seed = body.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            seed = secrets.randbits(32)
        with self.lock:
            brain = self._ready_brain()
            # the opponent comes from the mix this brain was solved for
            self.match = PracticeMatch(brain.rules, brain, brain.population, mode, seed)
            self._practice_record = {"kind": "practice", "id": str(uuid4()), "created_at": timestamp(),
                                     "seed": seed, "mode": mode, "settings": self.settings.to_dict(), "events": []}
            return self._save_practice(self.match.state(), "start")

    def _save_practice(self, snapshot: dict, event=None, request=None) -> dict:
        """Keep every action and round, including matches left before the summary screen."""
        record = self._practice_record
        if record is None:
            return snapshot
        now = timestamp()
        previous = record.get("state", {})
        if event or snapshot["phase"] != previous.get("phase"):
            record["events"].append({"at": now, "event": event or "clock", "request": request,
                                     "state": dict(snapshot)})
        finished = snapshot["phase"] == "match_over" or (snapshot["phase"] == "round_over" and
            (snapshot["round"] >= snapshot["rounds"] or snapshot["chips"] < self.match.rules.min_bet))
        record.update(updated_at=now, status="complete" if finished else "in_progress", state=dict(snapshot))
        path = self.data_dir / "matches" / (record["id"] + ".json")
        try:
            write_record(path, record)
            snapshot["saved_file"] = "data/matches/" + path.name
        except OSError:
            traceback.print_exc()
            snapshot["save_error"] = "This match could not be saved to disk. Check that the trainer's data folder is writable."
        return snapshot

    def _change_settings(self, body: dict) -> dict:
        new = Settings.from_dict(body)
        with self.lock:
            resolving = not new.same_brain_as(self.settings)
            self.settings = new
            save_settings(self.data_dir, new)
            if resolving:
                self.brain = None
                self.match = None
        if resolving:
            self.start_brain()
            with self.lock:
                if self.brain is None and self.status.get("ready"):
                    self.status = {"ready": False, "progress": 0.0, "message": "Getting ready"}
        return {"settings": new.to_dict(), "resolving": resolving}


# ---- HTTP ------------------------------------------------------------------------

def _handler_for(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MatchTrainer"

        def log_message(self, format, *args) -> None:
            if os.environ.get("TRAINER_LOG_REQUESTS"):      # quiet unless asked, for tracking down page problems
                print("request:", format % args, flush=True)

        def _send(self, code: int, content: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _static(self, path: str) -> None:
            name = "index.html" if path == "/" else path[len("/static/"):] if path.startswith("/static/") else ""
            target = WEB_DIR / name
            if (not name or "/" in name or "\\" in name or ".." in name or not target.is_file()
                    or target.suffix not in CONTENT_TYPES):
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            self._send(200, target.read_bytes(), CONTENT_TYPES[target.suffix])

        def _addressed_to_this_computer(self) -> bool:
            """False for a page that reached the trainer under some other name than this computer's own."""
            host = (self.headers.get("Host") or "").strip().lower()
            name = host if host.endswith("]") else host.rsplit(":", 1)[0]
            if name in ("127.0.0.1", "localhost", "[::1]"):
                return True
            self._send(403, b"Not allowed", "text/plain; charset=utf-8")
            return False

        def do_GET(self) -> None:
            if not self._addressed_to_this_computer():
                return
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/"):
                self._send_json(*app.handle("GET", path, None))
            else:
                self._static(path)

        def do_POST(self) -> None:
            if not self._addressed_to_this_computer():
                return
            path = self.path.split("?", 1)[0]
            # Only the trainer's own page sends JSON; a form or a plain request from another
            # web page cannot, so it is turned away before anything changes.
            kind = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if kind != "application/json":
                self._send_json(415, {"error": "That request was not understood."})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 <= length <= MAX_BODY:
                    raise ValueError("unreasonable length")
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": "That request was not understood."})
                return
            self._send_json(*app.handle("POST", path, body))

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False    # on Windows "reuse" would let a second trainer share the port
    request_queue_size = 64     # a browser opens several connections at once; the default line holds 5


def make_server(app: App, port: int = 8721) -> ThreadingHTTPServer:
    return _Server(("127.0.0.1", port), _handler_for(app))


def running_trainer(port: int) -> str | None:
    """The address of a trainer already answering on this port, or None."""
    address = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(address + "/api/status", timeout=1.5) as response:
            answer = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return address if isinstance(answer, dict) and "brain" in answer else None


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for port in PORTS:                  # opened twice? show the one that is already running
        address = running_trainer(port)
        if address:
            print(f"Match Trainer is already running at {address}")
            if not os.environ.get("TRAINER_NO_BROWSER"):
                webbrowser.open(address)
            return
    app = App(data_dir)
    server = None
    for port in PORTS:
        try:
            server = make_server(app, port)
            break
        except OSError:
            continue
    if server is None:
        print("Match Trainer could not start: ports 8721 to 8726 are all in use. "
              "Close other copies of the trainer and try again.")
        sys.exit(1)
    app.start_brain()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Match Trainer is running at {address}")
    print("Leave this window open while you use it. Close it to stop the trainer.")
    if not os.environ.get("TRAINER_NO_BROWSER"):
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
