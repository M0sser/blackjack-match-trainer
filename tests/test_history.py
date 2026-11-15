import copy
import json
from uuid import uuid4

import pytest

from trainer.errors import TrainerError
from trainer.history import read_record, review_path, save_review_form, write_record
from trainer.review import save_review


def form(revision=1):
    return {"revision": revision, "current": 0, "rounds": [{"main": ""} for _ in range(5)]}


def test_each_review_has_one_file_and_older_autosaves_cannot_replace_edits(tmp_path):
    identifier = str(uuid4())
    draft = form()
    path = save_review_form(tmp_path, identifier, draft)
    newer = copy.deepcopy(draft)
    newer["revision"] = 2
    newer["rounds"][0]["main"] = 150
    save_review_form(tmp_path, identifier, newer)
    save_review_form(tmp_path, identifier, draft)
    assert read_record(path)["form"] == newer
    assert len(list((tmp_path / "reviews").glob("*.json"))) == 1
    save_review_form(tmp_path, str(uuid4()), form())
    assert len(list((tmp_path / "reviews").glob("*.json"))) == 2


def test_analysis_updates_the_same_file_and_edits_remove_obsolete_analysis(tmp_path):
    identifier = str(uuid4())
    draft = form()
    path = save_review_form(tmp_path, identifier, draft)
    save_review(tmp_path, {"review_id": identifier, "review_form": draft}, {"final_chips": 1700})
    assert read_record(path)["report"]["final_chips"] == 1700
    selected_round = draft | {"revision": 2, "current": 1}
    save_review_form(tmp_path, identifier, selected_round, complete=True)
    assert "report" in read_record(path)
    edited = copy.deepcopy(selected_round)
    edited["revision"] = 3
    edited["rounds"][0]["main"] = 200
    save_review_form(tmp_path, identifier, edited)
    assert "report" not in read_record(path)
    # An analysis of the old form finishing late must not overwrite the edit.
    save_review(tmp_path, {"review_id": identifier, "review_form": draft}, {"final_chips": 1700})
    assert read_record(path)["form"] == edited and "report" not in read_record(path)
    assert len(list((tmp_path / "reviews").glob("*.json"))) == 1


@pytest.mark.parametrize("identifier", ["../outside", "", None, "not-a-match"])
def test_review_ids_cannot_be_used_as_paths(tmp_path, identifier):
    with pytest.raises(TrainerError, match="match ID"):
        review_path(tmp_path, identifier)


def test_interrupted_replacement_preserves_previous_record(tmp_path, monkeypatch):
    path = tmp_path / "match.json"
    write_record(path, {"round": 1})
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr("trainer.history.os.replace", fail)
    with pytest.raises(OSError):
        write_record(path, {"round": 2})
    assert json.loads(path.read_text()) == {"round": 1}
    assert list(tmp_path.iterdir()) == [path]
