import asyncio
import json

import pytest
from helpers import fake_log, open_live

from bridge import ToolResult
from live import ToolServerUnavailable

TEXT = [[{"blocks": [{"thinking": "let me think"}, {"text": "hello world"}]}]]
TOOL = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}]}, {"blocks": [{"text": "done"}]}]]
TOOLS2 = [[{"blocks": [{"tool_use": {"name": "echo", "input": {"x": 1}}}, {"tool_use": {"name": "echo", "input": {"x": 2}}}]},
           {"blocks": [{"text": "done"}]}]]


def collector():
    events = []
    return events, lambda kind, text: events.append((kind, text))


def result(text="echo ok"):
    return ToolResult({"content": [{"type": "text", "text": text}], "isError": False})


async def test_text_turn_streams_and_stops(tmp_path):
    live = await open_live(tmp_path, TEXT)
    try:
        events, emit = collector()
        await live.send_user("hi", uuid="u-1")
        end = await live.step(emit, timeout=20)
        assert end.kind == "stop"
        assert end.text.strip() == "hello world"
        assert end.thinking.strip() == "let me think"
        assert ("text", "hello ") in events
        assert end.reply_ids and not end.tool_calls
        assert end.upstream_requests == 1
        assert end.usage["prompt_tokens"] == 114
        assert not live.in_turn
    finally:
        await live.close()


async def test_tool_step_parks_then_resumes_same_turn(tmp_path):
    live = await open_live(tmp_path, TOOL)
    try:
        _, emit = collector()
        await live.send_user("call echo", uuid="u-1")
        first = await live.step(emit, timeout=20)
        assert first.kind == "tool_calls"
        assert [c["function"]["name"] for c in first.tool_calls] == ["echo"]
        assert json.loads(first.tool_calls[0]["function"]["arguments"]) == {"x": 1}
        assert live.waiting == [first.tool_calls[0]["id"]]
        assert live.in_turn
        live.resolve({first.tool_calls[0]["id"]: result()})
        second = await live.step(emit, timeout=20)
        assert second.kind == "stop"
        assert second.text.strip() == "done"
        assert len(second.tool_result_ids) == 1
        assert not live.waiting
        users = [e["in"] for e in fake_log(tmp_path) if "in" in e and e["in"].get("type") == "user"]
        assert len(users) == 1  # tool results never become a new user message
    finally:
        await live.close()


async def test_two_tool_calls_in_one_message(tmp_path):
    live = await open_live(tmp_path, TOOLS2)
    try:
        _, emit = collector()
        await live.send_user("call twice", uuid="u-1")
        first = await live.step(emit, timeout=20)
        assert first.kind == "tool_calls" and len(first.tool_calls) == 2
        live.resolve({c["id"]: result(f"r{i}") for i, c in enumerate(first.tool_calls)})
        second = await live.step(emit, timeout=20)
        assert second.kind == "stop" and len(second.tool_result_ids) == 2
    finally:
        await live.close()


async def test_steer_while_parked_reaches_running_turn(tmp_path):
    live = await open_live(tmp_path, TOOL)
    try:
        _, emit = collector()
        await live.send_user("call echo", uuid="u-1")
        first = await live.step(emit, timeout=20)
        await live.send_user("STEERED", uuid="u-2", priority="next")
        await asyncio.sleep(0.2)
        live.resolve({first.tool_calls[0]["id"]: result()})
        second = await live.step(emit, timeout=20)
        assert "STEERED" in second.text
        queued = [e["in"] for e in fake_log(tmp_path) if "in" in e and e["in"].get("priority") == "next"]
        assert queued and queued[0]["uuid"] == "u-2"
    finally:
        await live.close()


async def test_interrupt_during_generation(tmp_path):
    slow = [[{"blocks": [{"text": " ".join(str(i) for i in range(400))}], "slow": 0.01}]]
    live = await open_live(tmp_path, slow)
    try:
        events, emit = collector()
        await live.send_user("count", uuid="u-1")
        task = asyncio.create_task(live.step(emit, timeout=20))
        while not events:
            await asyncio.sleep(0.01)
        live.interrupt_soon()
        end = await task
        assert end.kind == "interrupted"
        assert not live.in_turn
    finally:
        await live.close()


async def test_interrupt_and_drain_while_parked(tmp_path):
    live = await open_live(tmp_path, TOOL)
    try:
        _, emit = collector()
        await live.send_user("call echo", uuid="u-1")
        await live.step(emit, timeout=20)
        await live.interrupt_and_drain(timeout=10)
        assert not live.waiting and not live.in_turn
        # The process is still usable for a fresh turn afterwards.
        await live.send_user("again", uuid="u-2")
        end = await live.step(emit, timeout=20)
        assert end.kind == "stop"
    finally:
        await live.close()


async def test_missing_tool_server_fails_closed(tmp_path):
    live = await open_live(tmp_path, TEXT, FAKE_CLAUDE_NO_MCP="1")
    try:
        _, emit = collector()
        await live.send_user("hi", uuid="u-1")
        with pytest.raises(ToolServerUnavailable):
            await live.step(emit, timeout=20)
    finally:
        await live.close()


async def test_api_error_ends_step_as_error(tmp_path):
    live = await open_live(tmp_path, [[{"error": 429}]])
    try:
        _, emit = collector()
        await live.send_user("hi", uuid="u-1")
        end = await live.step(emit, timeout=20)
        assert end.kind == "error"
        assert end.result.api_error_status == 429
    finally:
        await live.close()


async def test_model_and_effort_change_live(tmp_path):
    live = await open_live(tmp_path, TEXT + TEXT, effort="low")
    try:
        _, emit = collector()
        await live.send_user("hi", uuid="u-1")
        await live.step(emit, timeout=20)
        await live.set_model("sonnet")
        await live.set_effort("high")
        controls = [e["in"]["request"] for e in fake_log(tmp_path)
                    if "in" in e and e["in"].get("type") == "control_request"]
        subtypes = [c["subtype"] for c in controls]
        assert "set_model" in subtypes
        assert {"subtype": "apply_flag_settings", "settings": {"effortLevel": "high"}} in controls
        assert live.model == "sonnet" and live.effort == "high"
    finally:
        await live.close()


async def test_resume_passes_session_id(tmp_path):
    live = await open_live(tmp_path, TEXT, resume="22222222-2222-4222-8222-222222222222")
    try:
        _, emit = collector()
        await live.send_user("hi", uuid="u-1")
        end = await live.step(emit, timeout=20)
        argv = next(e["argv"] for e in fake_log(tmp_path) if "argv" in e)
        assert "--resume=22222222-2222-4222-8222-222222222222" in argv
        assert end.result.session_id == "22222222-2222-4222-8222-222222222222"
    finally:
        await live.close()
