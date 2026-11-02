import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from helpers import ECHO, FAKE, fake_env

import client as client_module
from host import SessionHost

ROOT = Path(__file__).resolve().parents[1]
TOOL = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}, {"blocks": [{"text": "done"}]}]]


@pytest.fixture
def fake_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    def install(script):
        host = SessionHost(cli_path=str(FAKE), env_override=fake_env(tmp_path, script))
        monkeypatch.setattr(client_module, "_HOST", host)
        return host

    yield install
    if client_module._HOST is not None:
        client_module._HOST.shutdown()


def kwargs(messages, **extra):
    return dict(model="haiku", messages=messages, tools=[ECHO], stream=True, hermes_session_id="sess", **extra)


def test_stream_yields_deltas_then_final_chunk(fake_host):
    fake_host([[{"blocks": [{"text": "hello there"}]}]])
    c = client_module.Client()
    chunks = list(c.chat.completions.create(**kwargs([{"role": "system", "content": "s"},
                                                       {"role": "user", "content": "hi"}])))
    deltas = "".join(ch.choices[0].delta.content or "" for ch in chunks[:-1])
    assert deltas.strip() == "hello there"
    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop"
    assert last.usage.prompt_tokens == 114
    assert last._response.choices[0].message.content.strip() == "hello there"


def test_tool_calls_finish_and_non_stream_collect(fake_host):
    fake_host(TOOL)
    c = client_module.Client()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "call"}]
    response = c.chat.completions.create(**{**kwargs(messages), "stream": False})
    assert response.choices[0].finish_reason == "tool_calls"
    call = response.choices[0].message.model_dump()["tool_calls"][0] if hasattr(
        response.choices[0].message, "model_dump") else response.choices[0].message.tool_calls[0]
    call_id = call["id"] if isinstance(call, dict) else call.id
    messages += [response.choices[0].message.model_dump(),
                 {"role": "tool", "tool_call_id": call_id, "content": "echo ok"}]
    final = c.chat.completions.create(**{**kwargs(messages), "stream": False})
    assert final.choices[0].message.content.strip() == "done"


def test_close_leaves_session_running(fake_host):
    host = fake_host([[{"blocks": [{"text": "a"}]}]])
    c = client_module.Client()
    list(c.chat.completions.create(**kwargs([{"role": "user", "content": "hi"}])))
    c.close()
    assert any(e.live is not None for e in host._entries.values())


def test_async_caller_gets_async_stream(fake_host):
    fake_host([[{"blocks": [{"text": "async ok"}]}]])
    c = client_module.Client()

    async def go():
        stream = await c.chat.completions.create(**kwargs([{"role": "user", "content": "hi"}]))
        return [chunk async for chunk in stream]

    chunks = asyncio.run(go())
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.parametrize("extra,expected", [
    ({"hermes_auxiliary": True}, True),
    ({"hermes_session_id": None}, True),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}}, True),
    ({"tool_choice": "required"}, True),
    ({}, False),
])
def test_routing(extra, expected):
    base = {"model": "haiku", "messages": [], "hermes_session_id": "s"}
    assert client_module.wants_directsdk({**base, **extra}) is expected


def test_reasoning_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    req = client_module.build_request({"model": "claude-opus-5-5", "messages": [], "hermes_session_id": "s",
                                       "extra_body": {"reasoning": {"enabled": False}}})
    assert (req.thinking_disabled, req.effort) == (False, "low")  # Opus 5.5 can't turn thinking off
    req = client_module.build_request({"model": "sonnet", "messages": [], "hermes_session_id": "s",
                                       "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}})
    assert req.effort == "medium" and req.model.startswith("claude-sonnet")
    assert req.system == ""


def test_profile_registers_and_routes(monkeypatch):
    spec = importlib.util.spec_from_file_location("claude_agent_session_plugin", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["claude_agent_session_plugin"] = module
    spec.loader.exec_module(module)
    from providers import get_provider_profile
    profile = get_provider_profile("claude-agent-session-experimental")
    assert profile is not None
    extra, top = profile.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"},
                                                 session_id="abc")
    assert extra == {"reasoning": {"enabled": True, "effort": "high"}} and top == {"hermes_session_id": "abc"}
    _, top = profile.build_api_kwargs_extras(session_id="abc", task="title")
    assert top == {"hermes_auxiliary": True}
    assert type(profile.create_client()).__name__ == "Client"
