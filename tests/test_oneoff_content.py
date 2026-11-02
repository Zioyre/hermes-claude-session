"""One-off history flattening: nothing silently lost, images kept, markers in order."""
import pytest

from convert import one_off_blocks, user_content_to_blocks

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def text(value):
    return {"type": "text", "text": value}


def image(mime="image/png", data="AAAA"):
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def input_image(mime="image/png", data="AAAA"):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def test_single_user_message_stays_raw():
    assert one_off_blocks([{"role": "user", "content": "just this"}]) == [text("just this")]


def test_single_user_message_after_system_stays_raw():
    messages = [{"role": "system", "content": "system prompt"},
                {"role": "developer", "content": [text("developer note")]},
                {"role": "user", "content": [text("look"), IMAGE]}]
    assert one_off_blocks(messages) == [text("look"), image()]


def test_single_user_message_matches_legacy_conversion_for_supported_parts():
    content = [text("one"), {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZZ"}}, text("two")]
    assert one_off_blocks([{"role": "user", "content": content}]) == user_content_to_blocks(content)


def test_multi_turn_history_keeps_order_markers_and_tool_call_details():
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "first\nline"},
        {
            "role": "assistant",
            "content": "checking",
            "reasoning_content": "private chain of thought",
            "reasoning_signature": "sig-should-vanish",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"x","n":1}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": [text("result"), IMAGE]},
        {"role": "user", "content": "second"},
    ]
    assert one_off_blocks(messages) == [
        text("[user]"),
        text("first\nline"),
        text("[assistant]"),
        text("checking"),
        text('[assistant tool_call call_1: lookup arguments={"q":"x","n":1}]'),
        text("[tool result for call_1]"),
        text("result"),
        image(),
        text("[user]"),
        text("second"),
    ]


def test_reasoning_content_and_signatures_are_never_forwarded():
    blocks = one_off_blocks([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "reasoning_content": "secret",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ])
    flattened = "".join(block.get("text", "") for block in blocks)
    assert "secret" not in flattened
    assert "sig" not in flattened


def test_object_arguments_are_serialized_deterministically():
    blocks = one_off_blocks([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "function": {"name": "f", "arguments": {"b": 2, "a": 1}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ])
    assert text('[assistant tool_call c1: f arguments={"a":1,"b":2}]') in blocks


def test_images_from_user_and_tool_results_are_preserved_in_order():
    blocks = one_off_blocks([
        {"role": "user", "content": [input_image("image/jpeg", "USER1"), text("u")]},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "tool_call_id": "c9", "content": [input_image("image/webp", "TOOL1"), text("t")]},
        {"role": "user", "content": [text("v"), input_image("image/gif", "USER2")]},
    ])
    images = [block for block in blocks if block["type"] == "image"]
    assert images == [image("image/jpeg", "USER1"), image("image/webp", "TOOL1"), image("image/gif", "USER2")]


def test_two_user_turns_without_an_assistant_are_still_marked():
    assert one_off_blocks([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == [
        text("[user]"), text("a"), text("[user]"), text("b")]


def test_empty_and_missing_content_keep_the_marker():
    assert one_off_blocks([{"role": "user", "content": ""}, {"role": "assistant", "content": None}]) == [
        text("[user]"), text(""), text("[assistant]")]


def test_empty_messages_flatten_to_nothing():
    assert one_off_blocks([]) == []


def test_system_and_developer_messages_are_skipped_by_the_flattener():
    blocks = one_off_blocks([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "developer", "content": "dev"},
        {"role": "user", "content": "c"},
    ])
    flattened = "".join(block.get("text", "") for block in blocks)
    assert "sys" not in flattened
    assert "dev" not in flattened


@pytest.mark.parametrize("part", [
    {"type": "file", "file": {"id": "1"}},
    {"type": "input_audio", "input_audio": {"data": "x"}},
    None,
    5,
])
def test_unsupported_user_content_parts_raise(part):
    with pytest.raises(ValueError, match="content part"):
        one_off_blocks([{"role": "user", "content": [part]}])


def test_remote_user_image_raises_instead_of_being_dropped():
    with pytest.raises(ValueError, match="only base64 data image_url inputs are supported"):
        one_off_blocks([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}])


def test_remote_tool_result_image_raises():
    with pytest.raises(ValueError, match="tool content: only base64"):
        one_off_blocks([
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
        ])


def test_assistant_non_text_content_raises():
    with pytest.raises(ValueError, match="assistant content supports text parts only"):
        one_off_blocks([{"role": "user", "content": "q"},
                        {"role": "assistant", "content": [IMAGE]}])


def test_assistant_content_must_be_text_or_null():
    with pytest.raises(ValueError, match="assistant content must be"):
        one_off_blocks([{"role": "user", "content": "q"},
                        {"role": "assistant", "content": 5}])


def test_tool_message_requires_a_call_id():
    with pytest.raises(ValueError, match="tool_call_id"):
        one_off_blocks([{"role": "user", "content": "q"},
                        {"role": "tool", "content": "result"}])


@pytest.mark.parametrize("tool_call", [
    {"id": "c1"},
    {"id": "", "function": {"name": "f", "arguments": "{}"}},
    {"function": {"name": "f", "arguments": "{}"}},
    {"id": "c1", "function": {"arguments": "{}"}},
    {"id": "c1", "function": {"name": "f", "arguments": 4}},
])
def test_malformed_assistant_tool_calls_raise(tool_call):
    with pytest.raises(ValueError, match="tool_call"):
        one_off_blocks([{"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a", "tool_calls": [tool_call]}])


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unsupported message role 'critic'"):
        one_off_blocks([{"role": "user", "content": "q"},
                        {"role": "critic", "content": "nope"}])


def test_non_message_inputs_raise():
    with pytest.raises(ValueError, match="list of message objects"):
        one_off_blocks("not messages")
    with pytest.raises(ValueError, match="list of message objects"):
        one_off_blocks([{"role": "user"}, "oops"])


def test_legacy_conversion_still_skips_unknown_parts():
    # Behavior of the persistent-session converter is unchanged: it stays lenient.
    assert user_content_to_blocks([{"type": "file", "file": {"id": "1"}}]) == []
