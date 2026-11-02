import json
import threading
from dataclasses import replace

import pytest
from helpers import ECHO, FAKE, fake_env, fake_log

import errors
from host import SessionHost, SessionRequest
from store import RecordStore

SYSTEM = {"role": "system", "content": "You are a test."}
TOOL = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}, {"blocks": [{"text": "done"}]}]]


def req(tmp_path, messages, session="s1", tools=(ECHO,), **kw):
    return SessionRequest(home=str(tmp_path / "home"), session_id=session, model="haiku", effort=None,
                          thinking_disabled=False, system="You are a test.", messages=messages, tools=list(tools),
                          settings=kw.pop("settings", {}), **kw)


def make_host(tmp_path, script, **env):
    return SessionHost(cli_path=str(FAKE), env_override=fake_env(tmp_path, script, **env))


def run(host, request, events=None):
    sink = events if events is not None else []
    return host.submit(host.run(request, lambda kind, text: sink.append((kind, text)))).result(30)


def users_sent(tmp_path):
    return [e["in"] for e in fake_log(tmp_path) if "in" in e and e["in"].get("type") == "user"]


def argvs(tmp_path):
    return [e["argv"] for e in fake_log(tmp_path) if "argv" in e]


def text_of(message):
    content = message["message"]["content"]
    return " ".join(b.get("text", "") for b in content) if isinstance(content, list) else content


@pytest.fixture
def host_cleanup():
    hosts = []
    yield hosts
    for h in hosts:
        h.shutdown()


def test_fresh_then_continue_on_one_process(tmp_path, host_cleanup):
    script = [[{"blocks": [{"text": "first"}]}], [{"blocks": [{"text": "second"}]}]]
    host = make_host(tmp_path, script)
    host_cleanup.append(host)
    m1 = [SYSTEM, {"role": "user", "content": "hi"}]
    out1 = run(host, req(tmp_path, m1))
    assert out1.end.kind == "stop" and out1.message["content"].strip() == "first"
    m2 = m1 + [out1.message, {"role": "user", "content": "again"}]
    out2 = run(host, req(tmp_path, m2))
    assert out2.message["content"].strip() == "second"
    assert len(argvs(tmp_path)) == 1
    sent = users_sent(tmp_path)
    assert [text_of(u) for u in sent] == ["hi", "again"]
    record = RecordStore(tmp_path / "home" / "claude-agent-session" / "records").load("s1")
    assert record and len(record.history) == 4 and not record.interrupted


def test_tool_loop_resolves_parked_turn_without_new_user_message(tmp_path, host_cleanup):
    host = make_host(tmp_path, TOOL)
    host_cleanup.append(host)
    m1 = [SYSTEM, {"role": "user", "content": "call echo"}]
    out1 = run(host, req(tmp_path, m1))
    assert out1.end.kind == "tool_calls"
    call = out1.message["tool_calls"][0]
    m2 = m1 + [out1.message, {"role": "tool", "tool_call_id": call["id"], "content": "echo ok"}]
    out2 = run(host, req(tmp_path, m2))
    assert out2.end.kind == "stop" and out2.message["content"].strip() == "done"
    assert len(users_sent(tmp_path)) == 1


def test_steer_after_tool_results_goes_to_running_turn(tmp_path, host_cleanup):
    host = make_host(tmp_path, TOOL)
    host_cleanup.append(host)
    m1 = [SYSTEM, {"role": "user", "content": "call echo"}]
    out1 = run(host, req(tmp_path, m1))
    call = out1.message["tool_calls"][0]
    m2 = m1 + [out1.message, {"role": "tool", "tool_call_id": call["id"], "content": "echo ok"},
               {"role": "user", "content": "STEERED"}]
    out2 = run(host, req(tmp_path, m2))
    assert "STEERED" in out2.message["content"]
    steer = [u for u in users_sent(tmp_path) if u.get("priority") == "next"]
    assert len(steer) == 1


def test_new_host_resumes_claude_session(tmp_path, host_cleanup):
    script = [[{"blocks": [{"text": "first"}]}]]
    host = make_host(tmp_path, script)
    m1 = [SYSTEM, {"role": "user", "content": "hi"}]
    out1 = run(host, req(tmp_path, m1))
    host.shutdown()
    record = RecordStore(tmp_path / "home" / "claude-agent-session" / "records").load("s1")
    host2 = make_host(tmp_path, script)
    host_cleanup.append(host2)
    run(host2, req(tmp_path, m1 + [out1.message, {"role": "user", "content": "again"}]))
    assert f"--resume={record.claude_session_id}" in argvs(tmp_path)[-1]
    assert text_of(users_sent(tmp_path)[-1]) == "again"


def test_edited_old_message_is_logged_and_not_resent(tmp_path, host_cleanup):
    script = [[{"blocks": [{"text": "a1"}]}], [{"blocks": [{"text": "a2"}]}], [{"blocks": [{"text": "a3"}]}]]
    host = make_host(tmp_path, script)
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "one"}]
    o1 = run(host, req(tmp_path, m))
    m += [o1.message, {"role": "user", "content": "two"}]
    o2 = run(host, req(tmp_path, m))
    m += [o2.message, {"role": "user", "content": "three"}]
    m[1] = {"role": "user", "content": "one (redacted)"}
    run(host, req(tmp_path, m))
    assert text_of(users_sent(tmp_path)[-1]) == "three"
    log = (tmp_path / "home" / "claude-agent-session" / "records" / "edits.jsonl").read_text().splitlines()
    assert json.loads(log[0])["group"] == 0


def test_compaction_starts_new_claude_session(tmp_path, host_cleanup):
    host = make_host(tmp_path, [[{"blocks": [{"text": "a"}]}], [{"blocks": [{"text": "b"}]}]])
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "one"}]
    o1 = run(host, req(tmp_path, m))
    compacted = [SYSTEM, {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary"},
                 {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    run(host, req(tmp_path, compacted))
    sessions = [a for a in argvs(tmp_path)]
    assert len(sessions) == 2 and any(x.startswith("--session-id") for x in sessions[1])
    last = text_of(users_sent(tmp_path)[-1])
    assert "<conversation-checkpoint>" in last and "summary" in last and last.endswith("next")
    assert o1.end.kind == "stop"


def test_tool_list_change_relaunches_with_resume(tmp_path, host_cleanup):
    host = make_host(tmp_path, [[{"blocks": [{"text": "a"}]}]])
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "one"}]
    o1 = run(host, req(tmp_path, m))
    other = {"type": "function", "function": {"name": "other", "description": "x", "parameters": {"type": "object"}}}
    run(host, req(tmp_path, m + [o1.message, {"role": "user", "content": "two"}], tools=(ECHO, other)))
    assert len(argvs(tmp_path)) == 2 and any(a.startswith("--resume=") for a in argvs(tmp_path)[1])


def test_api_error_raises_status_and_retry_continues_without_duplicate_user(tmp_path, host_cleanup):
    host = make_host(tmp_path, [[{"error": 429}], [{"blocks": [{"text": "ok now"}]}]])
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "hi"}]
    with pytest.raises(errors.ClaudeAPIError) as caught:
        run(host, req(tmp_path, m))
    assert caught.value.status_code == 429
    out = run(host, req(tmp_path, m))
    assert out.message["content"].strip() == "ok now"
    assert [text_of(u) for u in users_sent(tmp_path)] == ["hi", "Continue from where the conversation above left off."]


def test_cancel_marks_interrupted_and_next_turn_continues(tmp_path, host_cleanup):
    slow = [[{"blocks": [{"text": " ".join(str(i) for i in range(500))}], "slow": 0.01}],
            [{"blocks": [{"text": "fine"}]}]]
    host = make_host(tmp_path, slow)
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "count"}]
    events = []
    started = threading.Event()

    def emit(kind, text):
        events.append(text)
        started.set()

    future = host.submit(host.run(req(tmp_path, m), emit))
    assert started.wait(10)
    host.interrupt(str(tmp_path / "home"), "s1")
    with pytest.raises(RuntimeError, match=errors.CANCELLED):
        future.result(20)
    record = RecordStore(tmp_path / "home" / "claude-agent-session" / "records").load("s1")
    assert record.interrupted
    partial = {"role": "assistant", "content": "".join(events)}
    out = run(host, req(tmp_path, m + [partial, {"role": "user", "content": "go on"}]))
    assert out.message["content"].strip() == "fine"
    assert text_of(users_sent(tmp_path)[-1]) == "go on"


def test_lru_cap_retires_least_recent_session(tmp_path, host_cleanup):
    host = make_host(tmp_path, [[{"blocks": [{"text": "a"}]}]])
    host_cleanup.append(host)
    run(host, req(tmp_path, [SYSTEM, {"role": "user", "content": "one"}], session="s1",
                  settings={"max_live_processes": 1}))
    run(host, req(tmp_path, [SYSTEM, {"role": "user", "content": "two"}], session="s2",
                  settings={"max_live_processes": 1}))
    entries = host._entries
    assert entries[(str(tmp_path / "home"), "s1")].live is None
    assert entries[(str(tmp_path / "home"), "s2")].live is not None


def test_idle_timer_stops_process(tmp_path, host_cleanup):
    host = make_host(tmp_path, [[{"blocks": [{"text": "a"}]}]])
    host_cleanup.append(host)
    run(host, req(tmp_path, [SYSTEM, {"role": "user", "content": "one"}], settings={"idle_seconds": 1}))
    entry = host._entries[(str(tmp_path / "home"), "s1")]
    assert entry.live is not None
    import time
    for _ in range(40):
        if entry.live is None:
            break
        time.sleep(0.1)
    assert entry.live is None


def test_effort_change_at_tool_boundary_reaches_same_process(tmp_path, host_cleanup):
    host = make_host(tmp_path, TOOL)
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "call echo"}]
    first = run(host, replace(req(tmp_path, m), effort="low"))
    m += [first.message, {"role": "tool", "tool_call_id": first.message["tool_calls"][0]["id"], "content": "ok"}]
    run(host, replace(req(tmp_path, m), effort="high"))
    controls = [e["in"]["request"] for e in fake_log(tmp_path)
                if "in" in e and e["in"].get("type") == "control_request"]
    assert {"subtype": "apply_flag_settings", "settings": {"effortLevel": "high"}} in controls
    assert len(argvs(tmp_path)) == 1


def test_concurrent_open_does_not_exceed_process_cap(tmp_path, host_cleanup):
    script = [[{"blocks": [{"text": " ".join(["slow"] * 100)}], "slow": 0.01}]]
    host = make_host(tmp_path, script)
    host_cleanup.append(host)
    started = threading.Event()
    m = [SYSTEM, {"role": "user", "content": "hi"}]
    first = host.submit(host.run(req(tmp_path, m, settings={"max_live_processes": 1}),
                                 lambda *_: started.set()))
    assert started.wait(10)
    with pytest.raises(errors.ClaudeAPIError, match="process limit") as caught:
        run(host, req(tmp_path, m, session="s2", settings={"max_live_processes": 1}))
    assert caught.value.status_code == 503
    assert len(argvs(tmp_path)) == 1
    assert first.result(10).end.kind == "stop"


def test_retry_after_error_following_tool_result_continues(tmp_path, host_cleanup):
    script = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}, {"error": 503}],
              [{"blocks": [{"text": "recovered"}]}]]
    host = make_host(tmp_path, script)
    host_cleanup.append(host)
    m = [SYSTEM, {"role": "user", "content": "call echo"}]
    first = run(host, req(tmp_path, m))
    m += [first.message, {"role": "tool", "tool_call_id": first.message["tool_calls"][0]["id"], "content": "tool ok"}]
    with pytest.raises(errors.ClaudeAPIError) as caught:
        run(host, req(tmp_path, m))
    assert caught.value.status_code == 503
    recovered = run(host, req(tmp_path, m))
    assert recovered.message["content"].strip() == "recovered"
    assert text_of(users_sent(tmp_path)[-1]) == "Continue from where the conversation above left off."
    assert len(argvs(tmp_path)) == 1
