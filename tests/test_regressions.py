import threading
import time
from types import SimpleNamespace

import pytest
from helpers import FAKE, fake_env, fake_log
from test_host import SYSTEM, make_host, req, run, text_of, users_sent

import errors
from host import SessionHost
from store import RecordStore
from sync import Checkpoint, SessionRecord, groups


def test_two_errors_after_tool_results_do_not_brick_session(tmp_path):
    script = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}, {"error": 503}],
              [{"error": 503}], [{"blocks": [{"text": "recovered"}]}]]
    host = make_host(tmp_path, script)
    try:
        m = [SYSTEM, {"role": "user", "content": "call echo"}]
        first = run(host, req(tmp_path, m))
        m += [first.message, {"role": "tool", "tool_call_id": first.message["tool_calls"][0]["id"], "content": "tool ok"}]
        for _ in range(2):
            with pytest.raises(errors.ClaudeAPIError) as caught:
                run(host, req(tmp_path, m))
            assert caught.value.status_code == 503
        assert run(host, req(tmp_path, m)).message["content"].strip() == "recovered"
    finally:
        host.shutdown()


def test_cancel_ignored_by_cli_returns_promptly_and_releases_session(tmp_path):
    script = [[{"blocks": [{"text": " ".join(["slow"] * 500)}], "slow": 0.1}]]
    host = make_host(tmp_path, script, FAKE_CLAUDE_IGNORE_INTERRUPT="1")
    started = threading.Event()
    m = [SYSTEM, {"role": "user", "content": "count"}]
    try:
        future = host.submit(host.run(req(tmp_path, m), lambda *_: started.set()))
        assert started.wait(10)
        start = time.monotonic()
        host.interrupt(str(tmp_path / "home"), "s1")
        with pytest.raises(RuntimeError, match=errors.CANCELLED):
            future.result(5)
        assert time.monotonic() - start < 5
        entry = host._entries[(str(tmp_path / "home"), "s1")]
        assert entry.live is None and not entry.lock.locked()
        store = RecordStore(tmp_path / "home" / "claude-agent-session" / "records")
        record = store.load("s1")
        assert record is not None and record.interrupted
        host._env_override = fake_env(tmp_path, [[{"blocks": [{"text": "resumed"}]}]])
        # Retrying the exact same completion must remain possible, not produce local HTTP 400.
        assert run(host, req(tmp_path, m)).message["content"].strip() == "resumed"
    finally:
        host.shutdown()


def test_missing_resume_binding_is_discarded_for_full_catch_up(tmp_path):
    failing = tmp_path / "broken-claude"
    failing.write_text('#!/bin/sh\necho "No conversation found with session ID: dead" >&2\nexit 1\n')
    failing.chmod(0o755)
    m = [SYSTEM, {"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"}]
    store = RecordStore(tmp_path / "home" / "claude-agent-session" / "records")
    record = SessionRecord("dead", str(tmp_path), [Checkpoint(g.hash, None) for g in groups(m)])
    store.save("s1", record)
    m += [{"role": "user", "content": "new question"}]
    host = SessionHost(cli_path=str(failing), env_override=fake_env(tmp_path, [[{"blocks": [{"text": "recovered"}]}]]))
    try:
        with pytest.raises(errors.ClaudeAPIError) as caught:
            run(host, req(tmp_path, m))
        assert caught.value.status_code == 503
        assert store.load("s1") is None
        host._cli_path = str(FAKE)
        assert run(host, req(tmp_path, m)).message["content"].strip() == "recovered"
        catch_up = text_of(users_sent(tmp_path)[-1])
        assert "old question" in catch_up and "old answer" in catch_up and "new question" in catch_up
        argv = next(e["argv"] for e in fake_log(tmp_path) if "argv" in e)
        assert not any(a.startswith("--resume") for a in argv)
    finally:
        host.shutdown()


def test_midturn_process_error_is_retryable_and_marks_record(tmp_path):
    host = make_host(tmp_path, [[{"exit": 1}]])
    m = [SYSTEM, {"role": "user", "content": "hello"}]
    try:
        with pytest.raises(errors.ClaudeAPIError) as caught:
            run(host, req(tmp_path, m))
        assert caught.value.status_code == 503
        store = RecordStore(tmp_path / "home" / "claude-agent-session" / "records")
        record = store.load("s1")
        assert record is not None and record.interrupted
        host._env_override = fake_env(tmp_path, [[{"blocks": [{"text": "back"}]}]])
        assert run(host, req(tmp_path, m)).message["content"].strip() == "back"
    finally:
        host.shutdown()


def test_rewind_replays_groups_between_clean_fork_and_matching_prefix(tmp_path):
    old = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
           {"role": "user", "content": "important steering"}, {"role": "assistant", "content": "old tail"}]
    record = SessionRecord("native", str(tmp_path),
                           [Checkpoint(g.hash, mid) for g, mid in zip(groups(old), ["u0", "a1", None, "a3"])])
    messages = old[:3] + [{"role": "assistant", "content": "replacement tail"}, {"role": "user", "content": "next"}]
    host = SessionHost()
    host._sdk = SimpleNamespace(fork_session=lambda *_a, **_kw: SimpleNamespace(session_id="fork"))
    request = req(tmp_path, messages)
    plan = host._plan(request, RecordStore(tmp_path / "records"), record, groups(messages), host._entry(request))
    assert plan.resume == "fork"
    assert len(plan.history_base) == 2
    assert any("important steering" in b["text"] for b in plan.blocks)


def test_missing_fork_binding_is_discarded_for_retry(tmp_path):
    old = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
           {"role": "user", "content": "old"}]
    record = SessionRecord("missing", str(tmp_path),
                           [Checkpoint(g.hash, mid) for g, mid in zip(groups(old), ["u0", "a1", "u2"])])
    store = RecordStore(tmp_path / "records")
    store.save("s1", record)

    def fail_fork(*_a, **_kw):
        raise ValueError("Session not found")

    host = SessionHost()
    host._sdk = SimpleNamespace(fork_session=fail_fork)
    messages = old[:2] + [{"role": "user", "content": "replacement"}]
    request = req(tmp_path, messages)
    with pytest.raises(errors.ClaudeAPIError) as caught:
        host._plan(request, store, record, groups(messages), host._entry(request))
    assert caught.value.status_code == 503
    assert store.load("s1") is None


def test_iteration_limit_summary_uses_disposable_session():
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

    import client
    kwargs = {"hermes_session_id": "s", "tools": [{"type": "function"}],
              "messages": [{"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}]}
    assert client.wants_oneoff(kwargs)


async def test_failed_disconnect_kills_and_reaps_native_child():
    from bridge import PendingTools
    from live import LiveSession

    calls = []
    process = SimpleNamespace(returncode=None)

    def kill():
        calls.append("kill")
        process.returncode = -9

    async def wait():
        calls.append("reap")
        return -9

    async def disconnect():
        raise RuntimeError("transport wedged")

    process.kill = kill
    process.wait = wait
    live = LiveSession(sdk=None, options=None, offered=[], pending=PendingTools(), model="haiku", effort=None)
    live._client = SimpleNamespace(_transport=SimpleNamespace(_process=process), disconnect=disconnect)
    await live.close()
    assert calls == ["kill", "reap"]
    assert live.closed and live._client is None
