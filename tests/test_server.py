import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.test_review import MATCH
from trainer.server import App, make_server


@pytest.fixture
def app(tmp_path, quick_brain):
    a = App(tmp_path)
    a.set_brain(quick_brain)
    return a


def test_status(app):
    code, body = app.handle("GET", "/api/status", None)
    assert code == 200 and body["brain"]["ready"] is True and body["settings"]["default_mode"] == "coach"
    assert body["settings"]["rules"]["min_bet"] == 5.0 and "Flat 100" in body["default_rule_weights"]


def test_practice_flow(app):
    assert app.handle("GET", "/api/practice/state", None)[0] == 404
    code, s = app.handle("POST", "/api/practice/start", {"mode": "match", "seed": 5})
    assert code == 200 and s["phase"] == "bet" and s["chips"] == 1000.0
    code, s = app.handle("POST", "/api/practice/bet", {"main": 100})
    assert code == 200 and s["phase"] in ("insurance", "play", "round_over")
    assert app.handle("GET", "/api/practice/state", None)[1]["round"] == 1


def test_a_wrong_move_is_a_400_with_a_plain_message(app):
    app.handle("POST", "/api/practice/start", {"mode": "coach", "seed": 5})
    code, body = app.handle("POST", "/api/practice/action", {"action": "hit"})
    assert code == 400 and "bet" in body["error"].lower()
    assert app.handle("POST", "/api/practice/bet", {"main": "lots"})[0] == 400
    assert app.handle("POST", "/api/practice/start", {"mode": "turbo"})[0] == 400
    assert app.handle("GET", "/api/practice/state", None)[1]["phase"] == "bet"


def test_moves_before_any_match_are_refused(app):
    for path, body in (("/api/practice/bet", {"main": 100}), ("/api/practice/action", {"action": "hit"}),
                       ("/api/practice/next", {})):
        code, answer = app.handle("POST", path, body)
        assert code == 400 and "start" in answer["error"].lower()


def test_a_brain_that_is_still_solving_says_so(tmp_path):
    a = App(tmp_path)
    for path, body in (("/api/practice/start", {"mode": "coach"}), ("/api/review/analyze", {"rounds": MATCH})):
        code, answer = a.handle("POST", path, body)
        assert code == 503 and "working out" in answer["error"].lower()
    assert a.handle("GET", "/api/status", None)[1]["brain"]["ready"] is False
    assert a.handle("POST", "/api/review/check", {"rounds": MATCH})[0] == 200


def test_review_check_and_analyze(app, tmp_path):
    code, body = app.handle("POST", "/api/review/check", {"rounds": MATCH})
    assert code == 200 and body["rounds"][4]["chips_after"] == 1700.0
    code, report = app.handle("POST", "/api/review/analyze", {"rounds": MATCH, "opponent_final": 1450, "won": True})
    assert code == 200 and report["final_chips"] == 1700.0 and len(list((tmp_path / "reviews").glob("*.json"))) == 1
    code, body = app.handle("POST", "/api/review/analyze", {"rounds": MATCH[:2]})
    assert code == 400 and "Round 3" in body["error"]
    assert app.handle("POST", "/api/review/check", {"rounds": "nonsense"})[0] == 400


def test_settings_are_saved_and_a_rule_change_resolves(app, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(app, "start_brain", lambda: started.append(True))
    current = app.handle("GET", "/api/status", None)[1]["settings"]
    code, body = app.handle("POST", "/api/settings", {**current, "default_mode": "match"})
    assert code == 200 and body["resolving"] is False and not started
    changed = {**current, "rules": {**current["rules"], "min_bet": 10.0}}
    code, body = app.handle("POST", "/api/settings", changed)
    assert code == 200 and body["resolving"] is True and started == [True]
    assert json.loads((tmp_path / "settings.json").read_text())["rules"]["min_bet"] == 10.0
    assert app.handle("GET", "/api/status", None)[1]["brain"]["ready"] is False
    assert app.handle("POST", "/api/settings", {**current, "rules": {**current["rules"], "min_bet": -5}})[0] == 400


@pytest.mark.parametrize("bad", [
    {"rules": {"min_bet": 7}}, {"rules": {"tie_credit": 2}}, {"rules": {"clock_seconds": 5}},
    {"rule_weights": {"Flat 100": -1}}, {"rule_weights": {"Flat 100": 0}}, {"style_weights": {"book": "many"}},
    {"default_mode": "turbo"}, {"rules": "nonsense"},
])
def test_bad_settings_are_refused_in_plain_words(app, bad):
    code, body = app.handle("POST", "/api/settings", bad)
    assert code == 400 and body["error"] and "Traceback" not in body["error"]


def test_settings_survive_a_restart(tmp_path, quick_brain):
    first = App(tmp_path)
    first.set_brain(quick_brain)
    current = first.handle("GET", "/api/status", None)[1]["settings"]
    first.handle("POST", "/api/settings", {**current, "default_mode": "match"})
    assert App(tmp_path).handle("GET", "/api/status", None)[1]["settings"]["default_mode"] == "match"
    (tmp_path / "settings.json").write_text("{ not json")
    assert App(tmp_path).handle("GET", "/api/status", None)[1]["settings"]["default_mode"] == "coach"


def test_unknown_routes_and_crashes(app, monkeypatch):
    assert app.handle("GET", "/api/nope", None)[0] == 404
    monkeypatch.setattr("trainer.server.check_rounds", lambda *a: 1 / 0)
    code, body = app.handle("POST", "/api/review/check", {"rounds": []})
    assert code == 500 and "Traceback" not in body["error"]


def test_many_requests_at_once_are_all_answered(app):
    # a browser asks for the page's seven files at the same moment; none may be turned away
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/static/app.js"
    count = 60
    start, answers = threading.Barrier(count), []

    def fetch():
        start.wait()
        try:
            answers.append(urllib.request.urlopen(url, timeout=20).status)
        except Exception as err:
            answers.append(repr(err))

    threads = [threading.Thread(target=fetch) for _ in range(count)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.shutdown()
        server.server_close()
    assert answers == [200] * count


def test_pages_over_a_real_socket(app):
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Match Trainer" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/status").read())["brain"]["ready"] is True
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/static/../server.py")
        assert err.value.code == 404
        request = urllib.request.Request(base + "/api/practice/start", data=b"{ not json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
