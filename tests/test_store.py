import hashlib
import json
import os
import stat

from store import RecordStore
from sync import Checkpoint, SessionRecord


def make_record(session="claude-session"):
    return SessionRecord(session, "/tmp/work", [Checkpoint("abc", "m0")], interrupted=False)


def expected_path(root, session_id):
    name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32] + ".json"
    return root / name


def test_save_load_round_trip(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    record = make_record()
    store.save("sess-1", record)
    loaded = store.load("sess-1")
    assert loaded == record
    assert loaded is not None
    assert loaded.owner_pid == os.getpid()
    assert record.owner_pid == os.getpid()


def test_load_missing_returns_none(tmp_path):
    store = RecordStore(tmp_path / "records")
    assert store.load("absent") is None


def test_load_corrupt_returns_none(tmp_path):
    root = tmp_path / "records"
    root.mkdir()
    expected_path(root, "sess-1").write_text("{not json", encoding="utf-8")
    store = RecordStore(root)
    assert store.load("sess-1") is None


def test_saved_file_is_private_and_root_is_private(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    store.save("sess-1", make_record())
    file_mode = stat.S_IMODE(expected_path(root, "sess-1").stat().st_mode)
    assert file_mode == 0o600
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    store.save("sess-1", make_record())
    store.save("sess-1", make_record("claude-session-2"))
    assert [entry.name for entry in root.iterdir()] == [expected_path(root, "sess-1").name]
    loaded = store.load("sess-1")
    assert loaded is not None
    assert loaded.claude_session_id == "claude-session-2"


def test_delete_removes_the_record_and_tolerates_missing(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    store.save("sess-1", make_record())
    store.delete("sess-1")
    assert store.load("sess-1") is None
    store.delete("sess-1")


def test_log_edit_appends_json_lines_with_timestamps(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    store.log_edit({"kind": "edited", "preview": "hi"})
    store.log_edit({"kind": "rewind"})
    lines = (root / "edits.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "edited"
    assert first["preview"] == "hi"
    assert first["ts"].endswith("+00:00")
    assert json.loads(lines[1])["kind"] == "rewind"


def test_log_edit_keeps_existing_lines(tmp_path):
    root = tmp_path / "records"
    store = RecordStore(root)
    store.save("sess-1", make_record())
    store.log_edit({"a": 1})
    store.log_edit({"a": 2})
    lines = (root / "edits.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]
