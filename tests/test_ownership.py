import os
import subprocess
import sys

import pytest
from helpers import FAKE, fake_env

import errors
from host import SessionHost, SessionRequest
from store import RecordStore


def request(tmp_path, messages):
    return SessionRequest(home=str(tmp_path / "home"), session_id="s", model="haiku", effort=None,
                          thinking_disabled=False, system="test", messages=messages, tools=[])


def test_another_process_cannot_claim_a_live_lease(tmp_path):
    store = RecordStore(tmp_path / "records")
    lease = store.acquire("s")
    code = """
import sys
from pathlib import Path
from store import RecordStore
try:
    fd = RecordStore(Path(sys.argv[1])).acquire('s')
except BlockingIOError:
    print('busy')
else:
    raise AssertionError('acquired another process live lease')
"""
    try:
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path / "records")],
                               capture_output=True, text=True, check=True)
        assert child.stdout.strip() == "busy"
    finally:
        store.release(lease)
    store.release(store.acquire("s"))


def test_process_exit_releases_lease_without_removing_lock_file(tmp_path):
    code = """
import os, sys
from pathlib import Path
from store import RecordStore
fd = RecordStore(Path(sys.argv[1])).acquire('s')
os._exit(0)
"""
    root = tmp_path / "records"
    subprocess.run([sys.executable, "-c", code, str(root)], check=True)
    store = RecordStore(root)
    store.release(store.acquire("s"))
    assert len(list(root.glob("*.lock"))) == 1


def test_second_host_cannot_resume_live_transcript_but_can_after_shutdown(tmp_path):
    script = [[{"blocks": [{"text": "first"}]}]]
    env = fake_env(tmp_path, script)
    first = SessionHost(cli_path=str(FAKE), env_override=env)
    second = SessionHost(cli_path=str(FAKE), env_override=env)
    m = [{"role": "user", "content": "hello"}]
    try:
        out = first.submit(first.run(request(tmp_path, m), lambda *_: None)).result(15)
        m += [out.message, {"role": "user", "content": "next"}]
        with pytest.raises(errors.ClaudeAPIError, match="owned by another") as caught:
            second.submit(second.run(request(tmp_path, m), lambda *_: None)).result(15)
        assert caught.value.status_code == 409
        first.shutdown()
        resumed = second.submit(second.run(request(tmp_path, m), lambda *_: None)).result(15)
        assert resumed.end.kind == "stop"
        record = RecordStore(tmp_path / "home" / "claude-agent-session" / "records").load("s")
        assert record is not None and record.owner_pid == os.getpid()
    finally:
        first.shutdown()
        second.shutdown()
