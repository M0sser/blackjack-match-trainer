"""Durable per-match records. Updates replace the file atomically."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from trainer.errors import TrainerError


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def review_path(data_dir, review_id) -> Path:
    try:
        identifier = str(UUID(str(review_id)))
    except (ValueError, TypeError, AttributeError):
        raise TrainerError("The saved review needs a valid match ID.") from None
    return Path(data_dir) / "reviews" / f"review-{identifier}.json"


def read_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A stopped process leaves either the old record or the complete new one.
    handle, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_review_form(data_dir, review_id, form, *, complete=False, settings=None) -> Path:
    path = review_path(data_dir, review_id)
    if (not isinstance(form, dict) or not isinstance(form.get("rounds"), list)
            or len(form["rounds"]) != 5 or not all(isinstance(r, dict) for r in form["rounds"])):
        raise TrainerError("The saved review needs all five round entries.")
    revision = form.get("revision", 0)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise TrainerError("The saved review version was not understood.")
    previous = read_record(path)
    if revision < previous.get("revision", -1):
        return path                         # a delayed older autosave must not replace newer edits
    meaningful = lambda f: {k: v for k, v in f.items() if k not in ("current", "id", "revision")}
    unchanged = meaningful(form) == meaningful(previous.get("form", {}))
    now = timestamp()
    record = previous | {
        "kind": "review", "id": str(UUID(str(review_id))), "revision": revision,
        "created_at": previous.get("created_at", now), "updated_at": now,
        "status": "complete" if complete else "in_progress", "form": form,
    }
    if settings is not None:
        record["settings"] = settings
    if not unchanged:
        record.pop("report", None)
        record.pop("payload", None)
    write_record(path, record)
    return path


def save_review_analysis(data_dir, payload, report) -> Path:
    path = review_path(data_dir, payload["review_id"])
    previous = read_record(path)
    form = payload.get("review_form")
    if form is not None:
        path = save_review_form(data_dir, payload["review_id"], form, complete=True)
        previous = read_record(path)
        if form.get("revision", 0) < previous.get("revision", -1):
            return path
    now = timestamp()
    write_record(path, previous | {
        "kind": "review", "id": str(UUID(str(payload["review_id"]))), "created_at": previous.get("created_at", now),
        "updated_at": now, "status": "complete", "payload": payload, "report": report,
    })
    return path
