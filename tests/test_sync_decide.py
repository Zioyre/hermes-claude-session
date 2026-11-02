from sync import Checkpoint, Edit, SessionRecord, boundary, decide, groups


def user(text):
    return {"role": "user", "content": text}


def assistant(text=None, tool_calls=None):
    message = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


def tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def call(call_id, name="search", arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def make_record(messages, ids=None, interrupted=False, cwd="/tmp/work"):
    ids = ids or {}
    checkpoints = [Checkpoint(g.hash, ids.get(i)) for i, g in enumerate(groups(messages))]
    return SessionRecord("claude-session", cwd, checkpoints, interrupted=interrupted, owner_pid=7)


def test_no_record_is_fresh():
    decision = decide([user("hi")], None)
    assert decision.kind == "fresh"
    assert decision.start == 0
    assert decision.matched == 0
    assert decision.edits == ()


def test_fully_matched_record_with_user_only_tail_continues():
    seen = [user("u1"), assistant("a1")]
    record = make_record(seen)
    messages = seen + [user("u2"), user("u3")]
    decision = decide(messages, record)
    assert decision.kind == "continue"
    assert decision.matched == 2
    assert decision.start == 2


def test_fully_matched_record_with_assistant_tail_catches_up():
    seen = [user("u1"), assistant("a1")]
    record = make_record(seen)
    messages = seen + [user("u2"), assistant("a2"), tool("c1", "r1")]
    decision = decide(messages, record)
    assert decision.kind == "catch_up"
    assert decision.start == 2


def test_nothing_matched_is_fresh():
    record = make_record([user("other"), assistant("nope")])
    decision = decide([user("u1"), assistant("a1")], record)
    assert decision.kind == "fresh"
    assert decision.start == 0


def test_changed_user_text_before_matching_tail_is_edited():
    old = [user("u1"), assistant("a1"), user("u2"), assistant("a2")]
    record = make_record(old, ids={0: "m0", 1: "m1", 2: "m2", 3: "m3"})
    messages = [user("u1"), assistant("a1"), user("u2 edited"), assistant("a2")]
    decision = decide(messages, record)
    assert decision.kind == "edited"
    assert decision.matched == 2
    assert decision.edits == (
        Edit(group_index=2, old_hash=groups(old)[2].hash, new_hash=groups(messages)[2].hash, preview="u2 edited"),
    )
    assert decision.start == len(messages)
    assert decision.fork_at is None


def test_edited_start_points_at_the_new_tail_when_prompt_grew():
    old = [user("u1"), assistant("a1"), user("u2"), assistant("a2")]
    record = make_record(old)
    messages = [user("u1"), assistant("a1"), user("u2 edited"), assistant("a2"), user("u3")]
    decision = decide(messages, record)
    assert decision.kind == "edited"
    assert decision.start == 4


def test_edit_preview_is_truncated_to_120_chars():
    long_text = "x" * 300
    old = [user("u1"), assistant("a1"), user("u2"), assistant("a2")]
    record = make_record(old)
    messages = [user("u1"), assistant("a1"), user(long_text), assistant("a2")]
    decision = decide(messages, record)
    assert decision.kind == "edited"
    assert decision.edits[0].preview == "x" * 120


def test_edit_preview_is_empty_without_visible_text():
    old = [user("u1"), assistant("a1"), assistant("a2"), assistant("a3")]
    record = make_record(old)
    messages = [
        user("u1"),
        assistant("a1"),
        assistant(None, [call("c1")]),
        assistant("a3"),
    ]
    decision = decide(messages, record)
    assert decision.kind == "edited"
    assert decision.edits[0].preview == ""


def test_truncated_tail_with_new_input_rewinds():
    old = [user("u1"), assistant("a1"), user("u2"), assistant("a2")]
    record = make_record(old, ids={0: "m0", 1: "m1", 2: "m2", 3: "m3"})
    messages = [user("u1"), assistant("a1"), user("different")]
    decision = decide(messages, record)
    assert decision.kind == "rewind"
    assert decision.matched == 2
    assert decision.fork_at == "m1"
    assert decision.start == 2


def test_rewind_falls_back_to_fresh_without_a_boundary():
    old = [user("u1"), assistant("a1"), user("u2")]
    record = make_record(old)
    messages = [user("u1"), assistant("a1"), user("different")]
    decision = decide(messages, record)
    assert decision.kind == "fresh"
    assert decision.start == 0


def test_rewind_when_the_prompt_is_a_prefix_of_saved_history():
    old = [user("u1"), assistant("a1")]
    record = make_record(old, ids={0: "m0", 1: "m1"})
    decision = decide([user("u1")], record)
    assert decision.kind == "rewind"
    assert decision.fork_at == "m0"
    assert decision.start == 1


def test_boundary_skips_a_checkpoint_inside_a_tool_call():
    messages = [user("u1"), assistant("", [call("c1")]), tool("c1", "r1")]
    saved = [Checkpoint(g.hash, mid) for g, mid in zip(groups(messages), ["m0", "m1", "m2"])]
    assert boundary(messages, groups(messages), saved, 3) == "m2"
    assert boundary(messages, groups(messages), saved, 2) == "m0"


def test_boundary_requires_an_id_and_returns_the_last_clean_one():
    messages = [user("u1"), assistant("a1"), user("u2")]
    saved = [Checkpoint(g.hash, mid) for g, mid in zip(groups(messages), ["m0", "m1", None])]
    assert boundary(messages, groups(messages), saved, 3) == "m1"
    assert boundary(messages, groups(messages), saved, 1) == "m0"


def test_boundary_returns_none_when_no_checkpoint_has_an_id():
    messages = [user("u1"), assistant("a1")]
    saved = [Checkpoint(g.hash) for g in groups(messages)]
    assert boundary(messages, groups(messages), saved, 2) is None


def test_compaction_marker_forces_a_fresh_context():
    record = make_record([user("u1"), assistant("a1")])
    messages = [user("  [CONTEXT SUMMARY]:\nsummary"), user("u1"), assistant("a1")]
    decision = decide(messages, record)
    assert decision.kind == "compacted"
    assert decision.start == 0


def test_compaction_marker_already_in_history_does_not_trigger():
    seen = [user("[CONTEXT COMPACTION: summary]"), assistant("a1")]
    record = make_record(seen)
    messages = seen + [user("next")]
    decision = decide(messages, record)
    assert decision.kind == "continue"
    assert decision.start == 2


def test_interrupted_record_resumes_at_the_trailing_user_run():
    messages = [{"role": "system", "content": "s"}, user("u1"), assistant("a1"), user("u2"), user("u3")]
    record = SessionRecord("claude-session", "/tmp", None, interrupted=True)
    decision = decide(messages, record)
    assert decision.kind == "resume_interrupted"
    assert decision.start == 3


def test_interrupted_record_with_assistant_tail_has_nothing_to_send():
    messages = [user("u1"), assistant("a1")]
    record = SessionRecord("claude-session", "/tmp", None)
    decision = decide(messages, record)
    assert decision.kind == "resume_interrupted"
    assert decision.start == 2


def test_record_round_trip():
    record = make_record([user("u1"), assistant("a1")], ids={1: "m1"})
    data = record.to_dict()
    restored = SessionRecord.from_dict(data)
    assert restored == record
    assert restored.to_dict() == data


def test_record_round_trip_with_interrupted_history():
    record = SessionRecord("claude-session", "/tmp", None, interrupted=True, owner_pid=3)
    restored = SessionRecord.from_dict(record.to_dict())
    assert restored.history is None
    assert restored.interrupted is True
    assert restored.owner_pid == 3


def test_from_dict_tolerates_missing_optional_keys():
    restored = SessionRecord.from_dict({"claude_session_id": "s", "cwd": "/w"})
    assert restored.claude_session_id == "s"
    assert restored.cwd == "/w"
    assert restored.history is None
    assert restored.interrupted is False
    assert restored.owner_pid == 0
