from sync import groups, member_hash


def user(text):
    return {"role": "user", "content": text}


def assistant(text=None, tool_calls=None, **extra):
    message = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    message.update(extra)
    return message


def tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def hashes(messages):
    return [group.hash for group in groups(messages)]


def test_whitespace_only_change_keeps_hash():
    assert member_hash(user(" hello ")) == member_hash(user("hello"))
    assert hashes([user("a\n"), assistant("b ")]) == hashes([user("a"), assistant("b")])


def test_reasoning_fields_are_ignored():
    first = assistant("answer", reasoning_content="one", reasoning_details=[{"type": "x"}])
    second = assistant("answer", reasoning_content="two", reasoning="z")
    assert member_hash(first) == member_hash(second)
    assert hashes([first]) == hashes([second])


def test_tool_argument_key_order_does_not_change_hash():
    first = assistant("", [call("c1", "search", '{"a": 1, "b": [2, 3]}')])
    second = assistant("", [call("c1", "search", '{"b": [2, 3], "a": 1}')])
    assert member_hash(first) == member_hash(second)


def test_changed_text_changes_hash():
    assert member_hash(user("hello")) != member_hash(user("hello!"))
    assert member_hash(assistant("one")) != member_hash(assistant("two"))


def test_invalid_tool_arguments_are_hashed_verbatim():
    first = assistant("", [call("c1", "x", "not json")])
    second = assistant("", [call("c1", "x", "not json ")])
    assert member_hash(first) != member_hash(second)


def test_scalar_tool_arguments_are_canonicalised():
    first = assistant("", [call("c1", "x", '"abc"')])
    second = assistant("", [call("c1", "x", '  "abc"  ')])
    assert member_hash(first) == member_hash(second)
    assert member_hash(first) != member_hash(assistant("", [call("c1", "x", '"abd"')]))


def test_image_parts_hash_the_data_url():
    data_url = "data:image/png;base64,AAAA"
    first = user([{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": data_url}}])
    assert member_hash(first) != member_hash(user([{"type": "image_url", "image_url": {"url": data_url}}]))
    assert member_hash(first) != member_hash(
        user([{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}])
    )
    assert member_hash(user([{"type": "input_audio", "input_audio": {"data": "x"}}])) != member_hash(
        user([{"type": "input_audio", "input_audio": {"data": "y"}}])
    )


def test_consecutive_user_rows_merge_and_hash_is_order_independent():
    forward = [user("one"), user("two")]
    backward = [user("two"), user("one")]
    forward_groups = groups(forward)
    backward_groups = groups(backward)
    assert len(forward_groups) == 1
    assert forward_groups[0].role == "user"
    assert (forward_groups[0].start, forward_groups[0].end) == (0, 2)
    assert forward_groups[0].hash == backward_groups[0].hash


def test_consecutive_tool_rows_merge_and_assistant_rows_do_not():
    messages = [
        assistant("", [call("c1", "a", "{}"), call("c2", "b", "{}")]),
        tool("c1", "r1"),
        tool("c2", "r2"),
        assistant("done"),
        assistant("more"),
    ]
    result = groups(messages)
    assert [(g.role, g.start, g.end) for g in result] == [
        ("assistant", 0, 1),
        ("tool", 1, 3),
        ("assistant", 3, 4),
        ("assistant", 4, 5),
    ]
    assert result[0].hash != result[2].hash


def test_role_change_starts_a_new_group():
    result = groups([user("u"), assistant("a"), user("u2"), tool("c1", "r")])
    assert [(g.role, g.start, g.end) for g in result] == [
        ("user", 0, 1),
        ("assistant", 1, 2),
        ("user", 2, 3),
        ("tool", 3, 4),
    ]


def test_system_rows_are_skipped_without_shifting_identity():
    plain = [user("u"), assistant("a"), tool("c1", "r")]
    with_system = [
        {"role": "system", "content": "be nice"},
        user("u"),
        {"role": "developer", "content": "rules"},
        assistant("a"),
        tool("c1", "r"),
    ]
    assert hashes(plain) == hashes(with_system)
    result = groups(with_system)
    assert [(g.start, g.end) for g in result] == [(1, 2), (3, 4), (4, 5)]


def test_empty_and_system_only_messages_have_no_groups():
    assert groups([]) == []
    assert groups([{"role": "system", "content": "x"}]) == []


def test_system_row_between_user_rows_does_not_split_the_group():
    with_system = [user("u1"), {"role": "system", "content": "x"}, user("u2")]
    assert hashes(with_system) == hashes([user("u1"), user("u2")])
    result = groups(with_system)
    assert len(result) == 1
    assert (result[0].start, result[0].end) == (0, 3)


def test_group_hash_is_hex_sha256():
    group = groups([user("u")])[0]
    assert len(group.hash) == 64
    int(group.hash, 16)
