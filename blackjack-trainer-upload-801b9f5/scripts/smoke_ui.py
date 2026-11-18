"""Drive the real page in headless Microsoft Edge and check what ends up on screen.

    .venv/Scripts/python.exe scripts/smoke_ui.py practice
    .venv/Scripts/python.exe scripts/smoke_ui.py practice-coach
    .venv/Scripts/python.exe scripts/smoke_ui.py review
    .venv/Scripts/python.exe scripts/smoke_ui.py settings

Starts the trainer on a spare port with a small, quick brain, opens the page with a test
hook in the address (for example /#autoplay), waits for the hook to finish, and checks the
page. Prints SMOKE OK, or what was missing, and where the screenshot is.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.conftest import QUICK, QUICK_RULE_WEIGHTS, QUICK_STYLE_WEIGHTS   # noqa: E402
from trainer.server import App, make_server                                  # noqa: E402

EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

# check name -> (address hook, text that must be on the page, any-of groups, text that must not be)
CHECKS = {
    "practice": ("#autoplay", ["<title>autoplay done</title>", "finished"], [], ["error-strip visible"]),
    "practice-coach": ("#autoplay-coach", ["<title>autoplay done</title>", "finished"], [], ["error-strip visible"]),
    "review": ("#review-demo", ["<title>review done</title>", "1,700"], [["Biggest miss", "No misses"]],
               ["error-strip visible"]),
    "settings": ("#settings-demo", ["<title>settings done</title>", "Flat 100"], [], ["error-strip visible"]),
}


def find_edge() -> Path | None:
    return next((path for path in EDGE_PATHS if path.exists()), None)


def start_trainer():
    data_dir = ROOT / "data" / "smoke"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "settings.json").write_text(json.dumps(
        {"rule_weights": QUICK_RULE_WEIGHTS, "style_weights": QUICK_STYLE_WEIGHTS}), encoding="utf-8")
    app = App(data_dir, QUICK)
    app.start_brain()
    deadline = time.time() + 180
    while not app.status["ready"]:
        if time.time() > deadline:
            raise SystemExit("SMOKE FAILED: the quick brain was not ready after three minutes")
        time.sleep(0.5)
    for port in range(8799, 8810):
        try:
            server = make_server(app, port)
            break
        except OSError:
            continue
    else:
        raise SystemExit("SMOKE FAILED: no free port between 8799 and 8809")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def run_edge(edge: Path, profile: Path, url: str, *extra: str) -> str:
    command = [str(edge), "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
               f"--user-data-dir={profile}", "--window-size=1280,1700", "--virtual-time-budget=120000",
               *extra, url]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=240)
    return done.stdout


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "practice"
    if name not in CHECKS:
        print(f"unknown check {name!r}; use one of {', '.join(CHECKS)}")
        return 2
    edge = find_edge()
    if edge is None:
        print("SMOKE SKIPPED: Microsoft Edge was not found")
        return 0
    hook, needed, any_of, forbidden = CHECKS[name]
    server, base = start_trainer()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            dom = run_edge(edge, Path(scratch) / "profile-dom", base + hook, "--dump-dom")
            tries = 1
            if needed[0] not in dom:
                # Headless Edge on a brand-new profile now and then drops a script request while it
                # starts up. One more go tells that apart from a page that is really broken.
                dom = run_edge(edge, Path(scratch) / "profile-dom-2", base + hook, "--dump-dom")
                tries = 2
            # Picture the page exactly as it was dumped: the same markup with its scripts taken
            # out (so nothing re-runs) and its styles still loaded from the running trainer.
            frozen = re.sub(r"<script\b[^>]*>.*?</script>", "", dom, flags=re.S)
            frozen = frozen.replace("<head>", f'<head><base href="{base}">', 1)
            still = ROOT / "data" / "smoke" / f"{name}-frozen.html"
            still.write_text(frozen, encoding="utf-8")
            shot = ROOT / "data" / "smoke" / f"{name}.png"
            run_edge(edge, Path(scratch) / "profile-shot", still.as_uri(), f"--screenshot={shot}")
    finally:
        server.shutdown()
        server.server_close()

    problems = [f"missing: {text}" for text in needed if text not in dom]
    problems += [f"missing all of: {group}" for group in any_of if not any(text in dom for text in group)]
    problems += [f"should not be there: {text}" for text in forbidden if text in dom]
    (ROOT / "data" / "smoke" / f"{name}.html").write_text(dom, encoding="utf-8")
    if problems:
        print("SMOKE FAILED (" + name + "): " + "; ".join(problems))
        print(f"page saved to {ROOT / 'data' / 'smoke' / (name + '.html')}")
        return 1
    print(f"SMOKE OK ({name}){' on the second try' if tries == 2 else ''}; screenshot: {shot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
