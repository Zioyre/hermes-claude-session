"""Boundary for one-off (no persistent session) SDK calls.

Hermes hands the client OpenAI-shaped ``chat.completions.create`` kwargs. A disposable SDK
call only accepts a narrow slice of that surface: one choice, automatic tool choice, no
stop sequences, and text, JSON-object or JSON-Schema output. Everything else is rejected
*before* a Claude Code child process starts, so a caller gets a clear error instead of a
silently altered request.

Accepted-but-dropped controls (deliberate, not accidental):
- ``temperature`` / ``top_p``: validated for range/finiteness, then not forwarded. Subscription
  models reject sampling controls; Hermes keeps sending its harmless defaults.
- ``stop`` / ``extra_body.stop_sequences``: only an empty value is accepted, and nothing is
  forwarded (a nonempty stop list cannot be honoured by the SDK path).
- ``timeout``: validated here; mapping it to the SDK's own timeout belongs to the caller.
- ``extra_body.metadata.auth_profile``: host-local routing metadata, never sent to Claude Code.
- ``hermes_auxiliary`` / ``hermes_session_id``: host-local routing kwargs; accepted but never
  forwarded to the SDK request.

``structured_format`` returns the SDK ``output_format`` mapping but keeps the caller's
schema verbatim (only a deep copy is taken). The caller still validates the returned
``structured_output`` against that exact schema.
"""
from __future__ import annotations

import copy
import math

import httpx
import jsonschema

ALLOWED_KWARGS = frozenset({
    "model", "messages", "tools", "stream", "stream_options", "max_tokens",
    "max_completion_tokens", "temperature", "top_p", "stop", "extra_body",
    "extra_headers", "timeout", "tool_choice", "parallel_tool_calls", "n",
    "response_format", "hermes_auxiliary", "hermes_session_id",
})
ALLOWED_EXTRA_BODY = frozenset({
    "max_tokens", "temperature", "top_p", "stop_sequences", "reasoning", "response_format", "metadata",
})
_STREAM_OPTIONS = (None, {}, {"include_usage": True}, {"include_usage": False})


def _error(message: str) -> ValueError:
    return ValueError(f"claude-agent-session one-off request: {message}")


def _finite_number(value: object, name: str, *, upper: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _error(f"{name} must be finite")
    if positive and number <= 0:
        raise _error(f"{name} must be positive")
    if number < 0:
        raise _error(f"{name} must not be negative")
    if upper is not None and number > upper:
        raise _error(f"{name} must be between 0 and {upper:g}")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error(f"{name} must be a positive integer")
    return value


def _validate_headers(headers: object) -> None:
    if headers is None:
        return
    if not isinstance(headers, dict):
        raise _error("extra_headers must be an object")
    for key, value in headers.items():
        if not isinstance(key, str) or key.lower() != "traceparent" or not isinstance(value, str):
            raise _error("extra_headers supports host traceparent metadata only")


def _validate_reasoning(reasoning: object) -> None:
    if reasoning is None:
        return
    if not isinstance(reasoning, dict) or set(reasoning) - {"enabled", "effort"}:
        raise _error("reasoning supports enabled and effort only")
    if "enabled" in reasoning and type(reasoning["enabled"]) is not bool:
        raise _error("reasoning.enabled must be boolean")
    effort = reasoning.get("effort")
    if effort is not None and not isinstance(effort, str):
        raise _error("reasoning.effort must be a string")


def _validate_stop(stop: object, name: str) -> None:
    """Only an empty stop value is harmless; anything nonempty cannot be honoured by the SDK."""
    if stop is None or stop == [] or stop == "":
        return
    if isinstance(stop, str):
        raise _error(f"{name} must be empty; stop sequences are unsupported")
    if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        raise _error(f"{name} must be empty; stop sequences are unsupported")
    raise _error(f"{name} must be a string or a list of strings")


def _validate_schema(schema: object) -> dict:
    """Check the caller's original schema, including every nested subschema, with jsonschema."""
    if not isinstance(schema, dict):
        raise _error("response_format.json_schema.schema must be a JSON Schema object")
    def local_refs(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise _error("only local JSON Schema references are supported")
                local_refs(value)
        elif isinstance(node, list):
            for value in node:
                local_refs(value)

    local_refs(schema)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as error:
        raise _error(f"response_format.json_schema.schema is not valid JSON Schema: {error.message}") from error
    return schema


def _response_format(kwargs: dict) -> object:
    response_format = kwargs.get("response_format")
    if response_format is None:
        extra = kwargs.get("extra_body")
        if isinstance(extra, dict):
            response_format = extra.get("response_format")
    return response_format


def _validate_response_format(response_format: object) -> None:
    if response_format is None:
        return
    if not isinstance(response_format, dict):
        raise _error("response_format must be an object")
    kind = response_format.get("type")
    if kind in (None, "text", "json_object"):
        return
    if kind != "json_schema":
        raise _error(f"response_format type {kind!r} is unsupported; only json_schema is accepted")
    wrapper = response_format.get("json_schema")
    if not isinstance(wrapper, dict):
        raise _error("response_format.json_schema must be an object")
    if "name" in wrapper and not isinstance(wrapper["name"], str):
        raise _error("response_format.json_schema.name must be a string")
    if "strict" in wrapper and type(wrapper["strict"]) is not bool:
        raise _error("response_format.json_schema.strict must be boolean")
    _validate_schema(wrapper.get("schema"))


def validate_oneoff(kwargs: dict) -> None:
    """Raise ``ValueError`` when a one-off call cannot be served; return ``None`` when it can."""
    if not isinstance(kwargs, dict):
        raise _error("request kwargs must be an object")
    unknown = set(kwargs) - ALLOWED_KWARGS
    if unknown:
        raise _error("unsupported request parameters: " + ", ".join(sorted(unknown)))

    if isinstance(kwargs.get("n"), bool) or kwargs.get("n", 1) != 1:
        raise _error("only n=1 is supported")
    if kwargs.get("tool_choice", "auto") not in ("auto", None):
        raise _error("only tool_choice=auto is supported")
    if kwargs.get("parallel_tool_calls") is False:
        raise _error("parallel_tool_calls=False is unsupported")
    if kwargs.get("stream_options") not in _STREAM_OPTIONS:
        raise _error("unsupported stream_options")
    _validate_headers(kwargs.get("extra_headers"))

    messages = kwargs.get("messages")
    if messages is not None and (not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages)):
        raise _error("messages must be a list of message objects")

    extra = kwargs.get("extra_body")
    if extra is None:
        extra = {}
    elif not isinstance(extra, dict):
        raise _error("extra_body must be an object")
    unknown_extra = set(extra) - ALLOWED_EXTRA_BODY
    if unknown_extra:
        raise _error("unsupported extra_body fields: " + ", ".join(sorted(unknown_extra)))

    metadata = extra.get("metadata")
    if metadata is not None and (not isinstance(metadata, dict) or set(metadata) - {"auth_profile"}
                                 or ("auth_profile" in metadata and not isinstance(metadata["auth_profile"], str))):
        raise _error("metadata supports host auth_profile routing only")
    _validate_reasoning(extra.get("reasoning"))

    top_max = kwargs.get("max_tokens")
    if top_max is not None:
        _positive_int(top_max, "max_tokens")
    extra_max = extra.get("max_tokens")
    if extra_max is not None:
        _positive_int(extra_max, "extra_body.max_tokens")
    completion_max = kwargs.get("max_completion_tokens")
    if completion_max is not None:
        _positive_int(completion_max, "max_completion_tokens")
        if top_max is not None or extra_max is not None:
            raise _error("specify only one output-token limit")

    for source, label in ((kwargs, ""), (extra, "extra_body.")):
        for key in ("temperature", "top_p"):
            if source.get(key) is not None:
                _finite_number(source[key], f"{label}{key}", upper=1)

    _validate_stop(kwargs.get("stop"), "stop")
    _validate_stop(extra.get("stop_sequences"), "extra_body.stop_sequences")

    timeout = kwargs.get("timeout")
    if isinstance(timeout, httpx.Timeout):
        timeout = timeout.read
    if timeout is not None:
        _finite_number(timeout, "timeout", positive=True)

    _validate_response_format(_response_format(kwargs))


def structured_format(kwargs: dict) -> dict | None:
    """Map OpenAI ``response_format`` to the SDK ``output_format``; keep the caller's schema intact.

    Returns ``None`` for absent/text/prompt-guided JSON-object formats. The schema is a deep copy of the
    caller's schema, so later validation of ``structured_output`` uses the original contract.
    """
    response_format = _response_format(kwargs)
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise _error("response_format must be an object")
    if response_format.get("type") in (None, "text", "json_object"):
        return None
    if response_format.get("type") != "json_schema":
        raise _error("only json_schema structured output is supported")
    wrapper = response_format.get("json_schema")
    if not isinstance(wrapper, dict):
        raise _error("response_format.json_schema must be an object")
    schema = _validate_schema(wrapper.get("schema"))
    return {"type": "json_schema", "schema": copy.deepcopy(schema)}
