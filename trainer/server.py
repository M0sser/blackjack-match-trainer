"""The local server: a small JSON API over the practice match and the review, plus the page.

It listens on this computer only (127.0.0.1) and makes no network requests.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trainer.brain import Brain, BrainConfig
from trainer.errors import TrainerError
from trainer.opponents import DEFAULT_RULE_WEIGHTS, DEFAULT_STYLE_WEIGHTS
from trainer.practice import PracticeMatch
from trainer.review import analyze_match, check_rounds, save_review
from trainer.settings import MODES, Settings, load_settings, population_from, save_settings

WEB_DIR = Path(__file__).resolve().parent / "web"
NOT_READY = "Still working out the best bets. This takes a few minutes the first time."
CRASHED = "Something went wrong inside the trainer."
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                 ".ico": "image/x-icon"}


class _NotReady(Exception):
    pass


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
                return 200, self.match.state()
        if route == ("POST", "/api/practice/bet"):
            main, pp, fct = _amount(body, "main"), _amount(body, "pp", 0), _amount(body, "fct", 0)
            with self.lock:
                return 200, self._match().bet(main, pp, fct)
        if route == ("POST", "/api/practice/action"):
            with self.lock:
                return 200, self._match().act(str(body.get("action", "")))
        if route == ("POST", "/api/practice/next"):
            with self.lock:
                return 200, self._match().next_round()
        if route == ("POST", "/api/review/check"):
            return 200, {"rounds": check_rounds(self._rules(), self._rounds(body))}
        if route == ("POST", "/api/review/analyze"):
            with self.lock:
                brain = self._ready_brain()
            report = analyze_match(brain.rules, brain, self._rounds(body))
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
            return self.match.state()

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

        def log_message(self, *args) -> None:       # keep the console quiet
            pass

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

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/"):
                self._send_json(*app.handle("GET", path, None))
            else:
                self._static(path)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"error": "That request was not understood."})
                return
            self._send_json(*app.handle("POST", path, body))

    return Handler


def make_server(app: App, port: int = 8721) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler_for(app))
    server.daemon_threads = True
    return server


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    app = App(data_dir)
    server = None
    for port in range(8721, 8727):
        try:
            server = make_server(app, port)
            break
        except OSError:
            continue
    if server is None:
        print("Match Trainer could not start: ports 8721 to 8726 are all in use. "
              "Close other copies of the trainer and try again.")
        return
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
