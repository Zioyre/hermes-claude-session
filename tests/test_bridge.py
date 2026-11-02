import asyncio
import json

import pytest

from bridge import (
    MAX_DESCRIPTION,
    PREFIX,
    PendingTools,
    ToolResult,
    describe,
    original_name,
    short_name,
    tool_result_from_openai,
)


def test_short_name_leaves_valid_short_names_alone():
    assert short_name("search") == "search"
    assert short_name("read_file-2") == "read_file-2"
    assert len(short_name("a" * 50)) == 50


def test_short_name_hashes_long_or_odd_names():
    odd = "subagent " + "x" * 80 + "!"
    short = short_name(odd)
    assert len(short) <= 50
    assert short != odd
    assert short == short_name(odd)
    assert len(short_name("b" * 51)) <= 50


def test_original_name_round_trips_offered_names():
    offered = ["search", "subagent " + "x" * 80 + "!"]
    for name in offered:
        assert original_name(PREFIX + short_name(name), offered) == name
        assert original_name(short_name(name), offered) == name


def test_original_name_rejects_unoffered_names():
    assert original_name(PREFIX + "nope", ["search"]) is None
    assert original_name(PREFIX + short_name("nope"), ["search"]) is None


def test_describe_truncates_at_the_limit():
    assert describe("short") == "short"
    exact = "d" * MAX_DESCRIPTION
    assert describe(exact) == exact
    cut = describe("d" * (MAX_DESCRIPTION + 100))
    assert len(cut) == MAX_DESCRIPTION
    assert cut.endswith("…")


def test_tool_result_from_string_content():
    result = tool_result_from_openai("done")
    assert result.mcp == {"content": [{"type": "text", "text": "done"}], "isError": False}
    assert result.replacement is None


def test_tool_result_from_none_and_empty_content():
    for content in (None, ""):
        result = tool_result_from_openai(content)
        assert result.mcp == {"content": [{"type": "text", "text": ""}], "isError": False}


def test_tool_result_from_text_and_image_parts():
    content = [
        {"type": "text", "text": "see"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    result = tool_result_from_openai(content)
    assert result.mcp == {
        "content": [
            {"type": "text", "text": "see"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ],
        "isError": False,
    }
    assert result.replacement is None


def test_tool_result_from_unknown_part_uses_a_placeholder_and_replacement():
    part = {"type": "file", "file": {"filename": "a.pdf"}}
    result = tool_result_from_openai([{"type": "text", "text": "see"}, part])
    assert result.mcp["content"][0] == {"type": "text", "text": "see"}
    assert result.mcp["content"][1] == {"type": "text", "text": "Attached content follows."}
    assert result.replacement is not None
    assert result.replacement[1] == {"type": "text", "text": json.dumps(part, sort_keys=True, separators=(",", ":"))}


async def test_pending_tools_happy_path():
    pending = PendingTools()
    pending.expect("t1", "search")
    pending.before("t1", "search")
    assert pending.waiting() == ["t1"]

    mcp = {"content": [{"type": "text", "text": "found"}], "isError": False}
    replacement = [{"type": "text", "text": "real"}]
    task = asyncio.create_task(pending.call("search"))
    await asyncio.sleep(0)
    assert not task.done()

    pending.resolve("t1", ToolResult(mcp, replacement))
    assert await task == mcp
    assert pending.waiting() == []
    assert pending.after("t1") == replacement
    assert pending.after("t1") is None


async def test_call_without_before_raises():
    pending = PendingTools()
    pending.expect("t1", "search")
    with pytest.raises(RuntimeError, match="without PreToolUse id"):
        await pending.call("search")


async def test_call_with_mismatched_name_raises():
    pending = PendingTools()
    pending.expect("t1", "search")
    pending.before("t1", "search")
    with pytest.raises(RuntimeError, match="mismatched tool call t1"):
        await pending.call("other")


async def test_call_without_expect_raises():
    pending = PendingTools()
    pending.before("t1", "search")
    with pytest.raises(RuntimeError):
        await pending.call("search")


async def test_resolve_unknown_id_raises_key_error():
    pending = PendingTools()
    with pytest.raises(KeyError):
        pending.resolve("nope", ToolResult({"content": [], "isError": False}))


async def test_clear_unblocks_waiting_calls_with_an_error():
    pending = PendingTools()
    pending.expect("t1", "search")
    pending.before("t1", "search")
    task = asyncio.create_task(pending.call("search"))
    await asyncio.sleep(0)
    pending.clear()
    result = await task
    assert result == {"content": [{"type": "text", "text": "Cancelled by user."}], "isError": True}
    assert pending.waiting() == []
    assert pending.after("t1") is None


async def test_expect_is_idempotent_for_the_same_name_and_rejects_a_different_one():
    pending = PendingTools()
    pending.expect("t1", "search")
    pending.expect("t1", "search")
    assert pending.waiting() == ["t1"]
    with pytest.raises(RuntimeError):
        pending.expect("t1", "other")


async def test_waiting_lists_unresolved_ids_and_drops_after_resolve():
    pending = PendingTools()
    pending.expect("t1", "a")
    pending.expect("t2", "b")
    assert pending.waiting() == ["t1", "t2"]
    pending.resolve("t1", ToolResult({"content": [], "isError": False}))
    assert pending.waiting() == ["t2"]
