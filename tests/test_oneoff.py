import asyncio
import json
import os
from pathlib import Path

import pytest
from helpers import ECHO, FAKE, fake_env, fake_log

import client
import errors
from host import SessionHost


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-profile"))
    hosts = []

    def install(script):
        host = SessionHost(cli_path=str(FAKE), env_override=fake_env(tmp_path, script))
        hosts.append(host)
        monkeypatch.setattr(client, "_HOST", host)
        return host

    yield install
    for host in hosts:
        host.shutdown()


def request(**extra):
    return {"model": "haiku", "messages": [{"role": "user", "content": "Summarize this."}], **extra}


def assert_disposed(tmp_path, host):
    assert not host._entries
    assert not (tmp_path / "empty-profile" / "claude-agent-session").exists()
    for row in fake_log(tmp_path):
        if "argv" not in row:
            continue
        assert "--no-session-persistence" in row["argv"]
        assert not Path(row["cwd"]).exists()
        with pytest.raises(ProcessLookupError):
            os.kill(row["pid"], 0)


def test_text_auxiliary_is_disposable_without_other_provider(tmp_path, monkeypatch, run):
    import providers

    def no_other_provider(*_args, **_kwargs):
        raise AssertionError("One-off requests must not look up another provider")

    monkeypatch.setattr(providers, "get_provider_profile", no_other_provider)
    host = run([[{"blocks": [{"text": "summary"}]}]])
    response = client.Client().chat.completions.create(**request(hermes_auxiliary=True))
    assert response.choices[0].message.content.strip() == "summary"
    assert response.usage.prompt_tokens == 114
    assert_disposed(tmp_path, host)


def test_no_session_streams_and_async_calls_work(tmp_path, run):
    host = run([[{"blocks": [{"text": "streamed text"}]}]])

    async def invoke():
        stream = await client.Client().chat.completions.create(**request(stream=True))
        return [chunk async for chunk in stream]

    chunks = asyncio.run(invoke())
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks).strip() == "streamed text"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert_disposed(tmp_path, host)


SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"],
          "additionalProperties": False}
FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}


def test_schema_output_is_validated_and_streamed_as_json(tmp_path, run):
    host = run([[{"blocks": [{"text": "explanation not returned as JSON"}], "structured_output": {"answer": 7}}]])
    chunks = list(client.Client().chat.completions.create(**request(response_format=FORMAT, stream=True)))
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert json.loads(text) == {"answer": 7}
    assert json.loads(chunks[-1]._response.choices[0].message.content) == {"answer": 7}
    assert "--json-schema" in fake_log(tmp_path)[0]["argv"]
    assert_disposed(tmp_path, host)


@pytest.mark.parametrize("output", [None, {"answer": "wrong"}])
def test_missing_or_invalid_native_schema_result_fails(tmp_path, run, output):
    host = run([[{"blocks": [{"text": "ordinary text"}], "structured_output": output}]])
    with pytest.raises(errors.ClaudeAPIError) as caught:
        client.Client().chat.completions.create(**request(response_format=FORMAT))
    assert caught.value.status_code == 502
    assert_disposed(tmp_path, host)


@pytest.mark.parametrize("status", [401, 429, 503])
def test_native_errors_keep_status_and_dispose(tmp_path, run, status):
    host = run([[{"error": status}]])
    with pytest.raises(errors.ClaudeAPIError) as caught:
        client.Client().chat.completions.create(**request())
    assert caught.value.status_code == status
    assert_disposed(tmp_path, host)


def test_oneoff_tool_boundary_returns_to_hermes_then_closes(tmp_path, run):
    host = run([[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}]])
    response = client.Client().chat.completions.create(**request(tools=[ECHO]))
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "echo"
    assert_disposed(tmp_path, host)


def test_timeout_reaps_child_and_removes_temp_files(tmp_path, run):
    host = run([[{"blocks": [{"text": "one two three four five"}], "slow": 1}]])
    with pytest.raises(TimeoutError):
        client.Client().chat.completions.create(**request(timeout=0.3))
    assert_disposed(tmp_path, host)


def test_cancel_reaps_child_even_when_native_ignores_interrupt(tmp_path, run):
    host = run([[{"blocks": [{"text": "one two three four five"}], "slow": 2}]])
    host._env_override["FAKE_CLAUDE_IGNORE_INTERRUPT"] = "1"
    c = client.Client()
    stream = c.chat.completions.create(**request(stream=True))
    assert next(stream).choices[0].delta.content
    c.cancel()
    with pytest.raises(RuntimeError, match=errors.CANCELLED):
        list(stream)
    assert_disposed(tmp_path, host)


def test_auxiliary_does_not_touch_parked_main_session(tmp_path, run):
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

    host = run([[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]},
                 {"blocks": [{"text": "done"}]}]])
    c = client.Client()
    main = request(hermes_session_id="main", tools=[ECHO])
    first = c.chat.completions.create(**main)
    entry = next(iter(host._entries.values()))
    native = entry.live
    records = list((tmp_path / "empty-profile" / "claude-agent-session" / "records").glob("*.json"))
    assert len(records) == 1
    record = records[0]
    before = record.read_bytes()
    aux = request(hermes_session_id="main", messages=[{"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}])
    (tmp_path / "script.json").write_text(json.dumps([[{"blocks": [{"text": "summary"}]}]]))
    summary = c.chat.completions.create(**aux)
    assert summary.choices[0].message.content.strip() == "summary"
    assert entry.live is native and native.alive and native.waiting
    assert record.read_bytes() == before
    assert len(host._entries) == 1
    call = first.choices[0].message.tool_calls[0]
    main["messages"] += [first.choices[0].message.model_dump(),
                         {"role": "tool", "tool_call_id": call.id, "content": "ok"}]
    final = c.chat.completions.create(**main)
    assert final.choices[0].message.content.strip() == "done"


def test_auxiliary_processes_do_not_use_persistent_capacity(tmp_path, run):
    host = run([[{"blocks": [{"text": "one two three four five six"}], "slow": 0.25}]])
    c = client.Client()
    first = c.chat.completions.create(**request(stream=True))
    second = c.chat.completions.create(**request(stream=True))
    assert next(first).choices[0].delta.content
    assert next(second).choices[0].delta.content
    assert sum(entry.live is not None for entry in host._entries.values()) == 2
    (tmp_path / "script.json").write_text(json.dumps([[{"blocks": [{"text": "main reply"}]}]]))
    response = c.chat.completions.create(**request(hermes_session_id="main"))
    assert response.choices[0].message.content.strip() == "main reply"
    assert list(first)[-1].choices[0].finish_reason == "stop"
    assert list(second)[-1].choices[0].finish_reason == "stop"


def test_host_auth_profile_metadata_is_not_sent_to_native(tmp_path, run):
    host = run([[{"blocks": [{"text": "summary"}]}]])
    response = client.Client().chat.completions.create(
        **request(extra_body={"metadata": {"auth_profile": "routing-only"}}))
    assert response.choices[0].message.content.strip() == "summary"
    assert "routing-only" not in json.dumps(fake_log(tmp_path))
    assert_disposed(tmp_path, host)


def test_json_object_mode_uses_validated_text_without_native_schema(tmp_path, run):
    host = run([[{"blocks": [{"text": '{"ok":true}'}]}]])
    stream = client.Client().chat.completions.create(
        **request(stream=True, response_format={"type": "json_object"}))
    chunks = list(stream)
    assert json.loads("".join(c.choices[0].delta.content or "" for c in chunks)) == {"ok": True}
    assert "--json-schema" not in fake_log(tmp_path)[0]["argv"]
    assert_disposed(tmp_path, host)


@pytest.mark.parametrize("text", ["not json", "[]"])
def test_json_object_mode_rejects_non_object_results(tmp_path, run, text):
    host = run([[{"blocks": [{"text": text}]}]])
    with pytest.raises(errors.ClaudeAPIError) as caught:
        client.Client().chat.completions.create(**request(response_format={"type": "json_object"}))
    assert caught.value.status_code == 502
    assert_disposed(tmp_path, host)
