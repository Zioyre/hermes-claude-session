from sync import Checkpoint, catch_up_text, groups, rebuild_checkpoints


def user(text):
    return {"role": "user", "content": text}


def assistant(text=None, tool_calls=None):
    message = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


def tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


HEADER = (
    "<conversation-checkpoint>\n"
    "The messages below are earlier conversation that happened while you were not attached. "
    "Treat them as prior context, not as a new request.\n"
)


def test_renders_user_assistant_with_tool_call_and_result():
    messages = [
        user("hello"),
        assistant("Let me look", [call("c1", "search", '{"q": "x"}')]),
        tool("c1", "found"),
    ]
    expected = (
        HEADER
        + "<user>hello</user>\n"
        + '<assistant>Let me look\n<tool-call>{"id":"c1","name":"search","input":{"q":"x"}}</tool-call></assistant>\n'
        + '<tool-result id="c1">found</tool-result>\n'
        + "</conversation-checkpoint>"
    )
    assert catch_up_text(messages, 0, 3) == expected


def test_skips_system_rows_and_respects_the_range():
    messages = [
        user("before"),
        {"role": "system", "content": "ignore me"},
        user("inside"),
        assistant("reply"),
        user("after"),
    ]
    assert catch_up_text(messages, 1, 4) == HEADER + "<user>inside</user>\n<assistant>reply</assistant>\n</conversation-checkpoint>"


def test_empty_or_unrenderable_range_returns_empty_string():
    messages = [user("u"), {"role": "system", "content": "s"}]
    assert catch_up_text(messages, 1, 2) == ""
    assert catch_up_text(messages, 0, 0) == ""
    assert catch_up_text([], 0, 0) == ""


def test_images_render_as_placeholders():
    messages = [
        user(
            [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]
        )
    ]
    assert catch_up_text(messages, 0, 1) == HEADER + "<user>look\n[image]</user>\n</conversation-checkpoint>"


def test_invalid_tool_arguments_render_verbatim():
    messages = [assistant(None, [call("c1", "x", "not json")])]
    assert (
        catch_up_text(messages, 0, 1)
        == HEADER + '<assistant><tool-call>{"id":"c1","name":"x","input":"not json"}</tool-call></assistant>\n</conversation-checkpoint>'
    )


def test_rebuild_carries_ids_for_matching_hashes_and_drops_changed_ones():
    messages = [user("u1"), assistant("a1"), user("u2")]
    g = groups(messages)
    previous = [Checkpoint(g[0].hash, "m0"), Checkpoint(g[1].hash, "m1"), Checkpoint("stale", "m2")]
    rebuilt = rebuild_checkpoints(g, previous, {})
    assert [entry.hash for entry in rebuilt] == [group.hash for group in g]
    assert [entry.claude_message_id for entry in rebuilt] == ["m0", "m1", None]


def test_rebuild_applies_new_ids_even_over_carried_ones():
    messages = [user("u1"), assistant("a1")]
    g = groups(messages)
    previous = [Checkpoint(g[0].hash, "m0"), Checkpoint(g[1].hash, "m1")]
    rebuilt = rebuild_checkpoints(g, previous, {1: "fresh-id"})
    assert [entry.claude_message_id for entry in rebuilt] == ["m0", "fresh-id"]


def test_rebuild_without_previous_history():
    g = groups([user("u1")])
    rebuilt = rebuild_checkpoints(g, None, {0: "m0"})
    assert rebuilt == [Checkpoint(g[0].hash, "m0")]
