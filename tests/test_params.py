"""Boundary validation for one-off SDK calls: accepted shapes and pre-launch rejections."""
import copy

import httpx
import pytest

from params import structured_format, validate_oneoff

MESSAGES = [{"role": "user", "content": "hi"}]
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["n"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "rows"],
    "additionalProperties": False,
}
FORMAT = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": SCHEMA}}


def base(**extra):
    return {"model": "claude-sonnet-5", "messages": MESSAGES, **extra}


def rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_oneoff(kwargs)


def test_minimal_call_is_accepted():
    assert validate_oneoff(base()) is None


def test_hermes_sampling_defaults_are_accepted_and_documented_as_dropped():
    # Subscription models reject sampling controls; validation accepts Hermes' defaults anyway.
    assert validate_oneoff(base(temperature=1, top_p=0.95,
                                extra_body={"temperature": 0.7, "top_p": 1})) is None


def test_host_local_kwargs_are_accepted():
    assert validate_oneoff(base(hermes_auxiliary=True, hermes_session_id="s-1")) is None


def test_stream_and_stream_options_shapes():
    assert validate_oneoff(base(stream=True, stream_options={"include_usage": True})) is None
    assert validate_oneoff(base(stream=True, stream_options={"include_usage": False})) is None
    rejects(base(stream_options={"include_usage": True, "extra": 1}), "unsupported stream_options")


@pytest.mark.parametrize("value", [2, 0, -1, "1", True])
def test_n_must_be_exactly_one(value):
    rejects(base(n=value), "only n=1")


def test_n_one_is_accepted():
    assert validate_oneoff(base(n=1)) is None


@pytest.mark.parametrize("choice", ["required", "none", {"type": "function", "function": {"name": "x"}}])
def test_forced_tool_choice_is_rejected(choice):
    rejects(base(tool_choice=choice), "tool_choice=auto")


def test_auto_tool_choice_is_accepted():
    assert validate_oneoff(base(tool_choice="auto")) is None


def test_parallel_tool_calls_false_is_rejected():
    rejects(base(parallel_tool_calls=False), "parallel_tool_calls=False")
    assert validate_oneoff(base(parallel_tool_calls=True)) is None


@pytest.mark.parametrize("response_format", [
    {"type": "audio"},
    {},
    "text",
])
def test_non_json_schema_response_formats_are_rejected(response_format):
    if response_format == {}:
        # An object without a type is treated as no structured output by the OpenAI contract.
        assert validate_oneoff(base(response_format=response_format)) is None
    else:
        rejects(base(response_format=response_format), "response_format")


def test_json_object_mode_and_host_metadata_are_accepted():
    assert validate_oneoff(base(response_format={"type": "json_object"})) is None
    assert structured_format(base(response_format={"type": "json_object"})) is None
    assert validate_oneoff(base(extra_body={"metadata": {"auth_profile": "account"}})) is None
    rejects(base(extra_body={"metadata": {"api_key": "secret"}}), "host auth_profile")


def test_json_schema_response_format_is_accepted_top_level_and_in_extra_body():
    assert validate_oneoff(base(response_format=FORMAT)) is None
    assert validate_oneoff(base(extra_body={"response_format": FORMAT})) is None
    assert validate_oneoff(base(response_format={"type": "text"})) is None


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"x": {"type": "nonsense"}}},
    {"type": "object", "properties": {"x": {"type": "string", "minLength": "long"}}},
    {"type": "object", "properties": {"x": {"type": "string"}}, "required": "x"},
    {"type": "object", "properties": {"x": {"type": "array", "items": {"type": 7}}}},
    {"type": "object", "properties": {"x": {"anyOf": [{"type": "integer"}, {"type": "bogus"}]}}},
    {"type": "object", "properties": {"x": {"$ref": 3}}},
    {"type": "object", "additionalProperties": "yes"},
])
def test_nested_invalid_schemas_are_rejected_before_launch(schema):
    rejects(base(response_format={"type": "json_schema", "json_schema": {"schema": schema}}),
            "not valid JSON Schema")


@pytest.mark.parametrize("reference", ["https://example.invalid/schema", "file:///private/schema", "relative.json"])
def test_external_schema_references_are_rejected(reference):
    rejects(base(response_format={"type": "json_schema", "json_schema": {"schema": {"$ref": reference}}}),
            "only local JSON Schema references")


def test_schema_wrapper_shape_is_validated():
    rejects(base(response_format={"type": "json_schema"}), "json_schema must be an object")
    rejects(base(response_format={"type": "json_schema", "json_schema": {}}),
            "must be a JSON Schema object")
    rejects(base(response_format={"type": "json_schema", "json_schema": {"schema": []}}),
            "must be a JSON Schema object")
    rejects(base(response_format={"type": "json_schema",
                                  "json_schema": {"name": 3, "schema": SCHEMA}}),
            "name must be a string")
    rejects(base(response_format={"type": "json_schema",
                                  "json_schema": {"strict": "yes", "schema": SCHEMA}}),
            "strict must be boolean")


def test_unknown_top_level_body_and_header_fields_are_rejected():
    rejects(base(logit_bias={}), "unsupported request parameters: logit_bias")
    rejects(base(extra_body={"top_k": 1}), "unsupported extra_body fields: top_k")
    rejects(base(extra_headers={"x-api-key": "secret"}), "traceparent")
    rejects(base(extra_headers={"traceparent": 3}), "traceparent")
    rejects(base(extra_headers=["traceparent"]), "extra_headers must be an object")
    assert validate_oneoff(base(extra_headers={"TraceParent": "00-abc"})) is None
    assert validate_oneoff(base(extra_headers={"traceparent": "00-abc"})) is None


def test_extra_body_must_be_an_object():
    rejects(base(extra_body=[("max_tokens", 5)]), "extra_body must be an object")


def test_reasoning_shape_is_validated():
    assert validate_oneoff(base(extra_body={"reasoning": {"enabled": False}})) is None
    assert validate_oneoff(base(extra_body={"reasoning": {"effort": "high"}})) is None
    rejects(base(extra_body={"reasoning": {"budget": 5}}), "reasoning supports enabled and effort")
    rejects(base(extra_body={"reasoning": {"enabled": "no"}}), "enabled must be boolean")
    rejects(base(extra_body={"reasoning": "off"}), "reasoning supports")


def test_nonempty_stop_is_rejected_and_empty_stop_accepted():
    rejects(base(stop="END"), "stop must be empty")
    rejects(base(stop=["END"]), "stop must be empty")
    rejects(base(extra_body={"stop_sequences": ["END"]}), "stop_sequences")
    rejects(base(stop=5), "stop must be a string")
    assert validate_oneoff(base(stop=[])) is None
    assert validate_oneoff(base(stop="")) is None
    assert validate_oneoff(base(extra_body={"stop_sequences": []})) is None


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": 0}, {"max_tokens": -4}, {"max_tokens": 1.5}, {"max_tokens": True},
    {"max_completion_tokens": 0}, {"max_completion_tokens": "5"},
    {"max_tokens": 10, "max_completion_tokens": 10},
    {"max_tokens": 10, "extra_body": {"max_tokens": 10, "stop_sequences": []},
     "max_completion_tokens": 10},
])
def test_output_token_limits_are_validated(kwargs):
    rejects(base(**kwargs), "max_tokens|max_completion_tokens|positive integer|one output-token")


def test_output_token_limits_accepted():
    assert validate_oneoff(base(max_tokens=100)) is None
    assert validate_oneoff(base(max_completion_tokens=100)) is None
    assert validate_oneoff(base(extra_body={"max_tokens": 100})) is None


@pytest.mark.parametrize("key", ["temperature", "top_p"])
@pytest.mark.parametrize("value", [-0.1, 1.1, "0.5", True, float("nan")])
def test_sampling_ranges_are_validated(key, value):
    rejects(base(**{key: value}), f"{key} must")


@pytest.mark.parametrize("value", [0, -1, True, "5", float("inf"), float("nan")])
def test_timeout_must_be_a_positive_finite_number(value):
    rejects(base(timeout=value), "timeout must")


def test_timeout_is_accepted_for_the_caller_to_map():
    assert validate_oneoff(base(timeout=30)) is None
    assert validate_oneoff(base(timeout=0.5)) is None
    assert validate_oneoff(base(timeout=httpx.Timeout(30, read=10))) is None
    assert validate_oneoff(base(timeout=httpx.Timeout(None))) is None


@pytest.mark.parametrize("messages", ["hi", ["hi"]])
def test_messages_shape_is_validated(messages):
    kwargs = base()
    kwargs["messages"] = messages
    rejects(kwargs, "messages must be a list")


def test_validate_oneoff_rejects_non_dict_kwargs():
    with pytest.raises(ValueError, match="kwargs must be an object"):
        validate_oneoff(["model", "x"])  # type: ignore[arg-type]


def test_structured_format_maps_openai_shape_and_preserves_the_caller_schema():
    kwargs = base(response_format=FORMAT)
    before = copy.deepcopy(kwargs)
    output = structured_format(kwargs)
    assert output == {"type": "json_schema", "schema": SCHEMA}
    assert kwargs == before
    # The returned schema is a deep copy: the caller's contract can't be mutated by the SDK path.
    output["schema"]["properties"]["title"]["type"] = "integer"
    assert kwargs["response_format"]["json_schema"]["schema"] == before["response_format"]["json_schema"]["schema"]


def test_structured_format_reads_extra_body_and_returns_none_for_text_or_absent():
    assert structured_format(base(extra_body={"response_format": FORMAT})) == {
        "type": "json_schema", "schema": SCHEMA}
    assert structured_format(base()) is None
    assert structured_format(base(response_format={"type": "text"})) is None
    assert structured_format(base(extra_body={})) is None


def test_structured_format_rejects_unsupported_formats_and_invalid_schema():
    with pytest.raises(ValueError, match="only json_schema"):
        structured_format(base(response_format={"type": "audio"}))
    with pytest.raises(ValueError, match="not valid JSON Schema"):
        structured_format(base(response_format={"type": "json_schema",
                                                "json_schema": {"schema": {"type": "nope"}}}))
    with pytest.raises(ValueError, match="json_schema must be an object"):
        structured_format(base(response_format={"type": "json_schema"}))
