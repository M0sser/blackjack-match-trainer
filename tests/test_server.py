import http.client
import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.test_review import MATCH
from trainer.server import App, make_server, running_trainer


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


def test_review_autosaves_incomplete_matches_without_the_brain(tmp_path):
    from uuid import uuid4
    app = App(tmp_path)
    draft = {"revision": 1, "rounds": [{"main": ""} for _ in range(5)]}
    identifier = str(uuid4())
    code, answer = app.handle("POST", "/api/review/save", {
        "review_id": identifier, "form": draft, "rounds": [None] * 5})
    assert code == 200 and answer["saved"]
    path = tmp_path / "reviews" / f"review-{identifier}.json"
    saved = json.loads(path.read_text())
    assert saved["status"] == "in_progress" and saved["form"] == draft
    assert App(tmp_path).data_dir.joinpath("reviews", path.name).is_file()


def test_completed_review_is_saved_before_analysis_and_analysis_uses_the_same_file(app, tmp_path):
    from uuid import uuid4
    identifier = str(uuid4())
    draft = {"revision": 1, "rounds": MATCH}
    assert app.handle("POST", "/api/review/save", {
        "review_id": identifier, "form": draft, "rounds": MATCH})[0] == 200
    path = tmp_path / "reviews" / f"review-{identifier}.json"
    assert json.loads(path.read_text())["status"] == "complete"
    code, report = app.handle("POST", "/api/review/analyze", {
        "review_id": identifier, "review_form": draft, "rounds": MATCH})
    assert code == 200 and report["final_chips"] == 1700
    assert json.loads(path.read_text())["report"] == report
    assert len(list((tmp_path / "reviews").glob("*.json"))) == 1


def test_practice_saves_every_action_and_keeps_separate_matches(app, tmp_path):
    code, state = app.handle("POST", "/api/practice/start", {"mode": "coach", "seed": 5})
    assert code == 200 and "saved_file" in state
    path = next((tmp_path / "matches").glob("*.json"))
    while state["phase"] != "match_over":
        phase = state["phase"]
        if phase == "bet":
            route, body = "/api/practice/bet", {"main": 100}
        elif phase in ("insurance", "play"):
            route, body = "/api/practice/action", {"action": "no_insurance" if phase == "insurance" else "stand"}
        else:
            if state["round"] == 5:
                assert json.loads(path.read_text())["status"] == "complete"
            route, body = "/api/practice/next", {}
        code, state = app.handle("POST", route, body)
        assert code == 200
        saved = json.loads(path.read_text())
        assert saved["state"]["step"] == state["step"]
    assert saved["status"] == "complete" and saved["state"]["summary"] == state["summary"]
    assert any(event["event"] == "action" for event in saved["events"])
    assert saved["seed"] == 5 and saved["settings"]
    app.handle("POST", "/api/practice/start", {"mode": "match", "seed": 6})
    assert len(list((tmp_path / "matches").glob("*.json"))) == 2
    assert json.loads(path.read_text())["status"] == "complete"


def test_practice_timeouts_are_saved_on_the_state_poll(app, tmp_path):
    app.handle("POST", "/api/practice/start", {"mode": "match", "seed": 5})
    app.match._used = app.match.rules.clock_seconds
    app.match._since = None
    code, state = app.handle("GET", "/api/practice/state", None)
    assert code == 200 and state["summary"]["outcome"] == "forfeit"
    saved = json.loads(next((tmp_path / "matches").glob("*.json")).read_text())
    assert saved["status"] == "complete" and saved["state"]["summary"]["outcome"] == "forfeit"


def test_practice_save_failure_is_visible_and_can_be_retried(app, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk failure")
    with monkeypatch.context() as patch:
        patch.setattr("trainer.server.write_record", fail)
        code, state = app.handle("POST", "/api/practice/start", {"mode": "coach", "seed": 5})
        assert code == 200 and state["save_error"] and state["phase"] == "bet"
    code, state = app.handle("GET", "/api/practice/state", None)
    assert code == 200 and "saved_file" in state and "save_error" not in state
    assert len(list((tmp_path / "matches").glob("*.json"))) == 1


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
        request = urllib.request.Request(base + "/api/practice/start", data=b"{ not json", method="POST",
                                         headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(request)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


# ---- found by the final review ------------------------------------------------------

def _serve(app):
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _raw(port, method, path, headers, body=b""):
    """One request with exactly these headers (no automatic Host)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for name, value in headers.items():
        conn.putheader(name, value)
    conn.endheaders(body)
    response = conn.getresponse()
    response.read()
    conn.close()
    return response.status


def test_two_trainers_cannot_share_a_port(app):
    first = make_server(app, port=0)
    try:
        with pytest.raises(OSError):
            make_server(app, port=first.server_address[1])
    finally:
        first.server_close()


def test_a_trainer_that_is_already_running_is_found(app):
    server, port = _serve(app)
    try:
        assert running_trainer(port) == f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
    assert running_trainer(port) is None


def test_a_click_that_arrives_twice_is_a_400(app):
    start = app.handle("POST", "/api/practice/start", {"mode": "coach", "seed": 5})[1]
    assert app.handle("POST", "/api/practice/bet", {"main": 100, "step": start["step"]})[0] == 200
    code, body = app.handle("POST", "/api/practice/bet", {"main": 100, "step": start["step"]})
    assert code == 400 and "twice" in body["error"]
    assert app.handle("POST", "/api/practice/action", {"action": "hit", "step": "soon"})[0] == 400


def test_requests_from_other_web_pages_are_refused(app):
    server, port = _serve(app)
    start = b'{"mode": "coach", "seed": 5}'
    ours = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"}
    try:
        assert _raw(port, "POST", "/api/practice/start", {**ours, "Content-Length": str(len(start))}, start) == 200
        # a form or a plain fetch from another page cannot send JSON without asking first
        assert _raw(port, "POST", "/api/settings", {**ours, "Content-Type": "text/plain", "Content-Length": "2"}, b"{}") == 415
        # a page that reached this computer under some other name
        assert _raw(port, "GET", "/api/status", {"Host": "evil.example"}) == 403
        assert _raw(port, "GET", "/", {"Host": "evil.example"}) == 403
        assert _raw(port, "POST", "/api/settings", {**ours, "Host": "evil.example:80", "Content-Length": "2"}, b"{}") == 403
        assert _raw(port, "GET", "/api/status", {"Host": f"localhost:{port}"}) == 200
        # nonsense lengths are refused at once instead of tying up a thread
        assert _raw(port, "POST", "/api/practice/next", {**ours, "Content-Length": "-5"}) == 400
        assert _raw(port, "POST", "/api/practice/next", {**ours, "Content-Length": "lots"}) == 400
    finally:
        server.shutdown()
        server.server_close()
