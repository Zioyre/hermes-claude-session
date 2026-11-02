import json

_USAGE_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
)


class UsageTracker:
    """Claude repeats cumulative usage while streaming, so keep the max per message and sum output."""

    def __init__(self) -> None:
        self._by_id: dict[str, dict[str, int]] = {}
        self._last_id: str | None = None

    def update(self, message_id: str, usage: dict) -> None:
        counts = self._by_id.setdefault(message_id, dict.fromkeys(_USAGE_KEYS, 0))
        for key in _USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[key] = max(counts[key], value)
        self._last_id = message_id

    def openai_usage(self) -> dict:
        native = {key: 0 for key in _USAGE_KEYS}
        if self._last_id is not None:
            native.update(self._by_id[self._last_id])
        prompt = native["input_tokens"] + native["cache_read_input_tokens"] + native["cache_creation_input_tokens"]
        completion = sum(counts["output_tokens"] for counts in self._by_id.values())
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": native["cache_read_input_tokens"]},
            "cache_creation_input_tokens": native["cache_creation_input_tokens"],
            "native_usage": dict(native) if self._last_id is not None else {},
        }


def _data_image(url: object) -> tuple[str, str] | None:
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url[len("data:") :].split(";base64,", 1)
    return header, data


def user_content_to_blocks(content: str | list) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    blocks: list[dict] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            raw = part.get("text")
            blocks.append({"type": "text", "text": raw if isinstance(raw, str) else ""})
            continue
        if part.get("type") != "image_url":
            continue
        image = part.get("image_url")
        parsed = _data_image(image.get("url") if isinstance(image, dict) else None)
        if parsed is None:
            raise ValueError("Only base64 data image_url inputs are supported")
        mime, data = parsed
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
    return blocks


# -- loss-aware conversion for a one-off SDK turn -------------------------------------------
_SYSTEM_ROLES = ("system", "developer")


def _marker(text: str) -> dict:
    return {"type": "text", "text": text}


def _strict_content(content: object, where: str) -> list[dict]:
    """Convert user/tool content without dropping anything: unknown parts and remote images raise."""
    if content is None:
        return []
    if isinstance(content, str):
        return [_marker(content)]
    if not isinstance(content, list):
        raise ValueError(f"{where} must be a string or a list of content parts")  # noqa: TRY004 - callers catch ValueError from this boundary
    blocks: list[dict] = []
    for part in content:
        if isinstance(part, str):
            blocks.append(_marker(part))
            continue
        if not isinstance(part, dict):
            raise ValueError(f"{where} content parts must be objects or strings")  # noqa: TRY004 - callers catch ValueError from this boundary
        kind = part.get("type")
        if kind == "text":
            raw = part.get("text")
            if not isinstance(raw, str):
                raise ValueError(f"{where} text parts must carry a string 'text'")
            blocks.append(_marker(raw))
            continue
        if kind == "image_url":
            image = part.get("image_url")
            parsed = _data_image(image.get("url") if isinstance(image, dict) else None)
            if parsed is None:
                raise ValueError(f"{where}: only base64 data image_url inputs are supported")
            mime, data = parsed
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
            continue
        raise ValueError(f"{where}: unsupported content part type {kind!r}")
    return blocks


def _arguments_text(arguments: object) -> str:
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raise ValueError("assistant tool_call arguments must be a JSON string or object")


def _assistant_blocks(message: dict) -> list[dict]:
    blocks: list[dict] = [_marker("[assistant]")]
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                blocks.append(_marker(part))
            elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                blocks.append(_marker(part["text"]))
            else:
                raise ValueError("assistant content supports text parts only")
    elif isinstance(content, str):
        blocks.append(_marker(content))
    elif content is not None:
        raise ValueError("assistant content must be a string, a list of text parts or null")

    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("assistant tool_calls must be a list")  # noqa: TRY004 - callers catch ValueError from this boundary
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ValueError("assistant tool_calls must be function tool call objects")  # noqa: TRY004 - callers catch ValueError from this boundary
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("assistant tool_calls must carry a nonempty id")
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("assistant tool_calls must carry a nonempty function name")
        blocks.append(_marker(
            f"[assistant tool_call {call_id}: {name} arguments={_arguments_text(function.get('arguments'))}]"))
    return blocks


def _tool_blocks(message: dict) -> list[dict]:
    call_id = message.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("tool messages must carry a nonempty tool_call_id")
    return [_marker(f"[tool result for {call_id}]"), *_strict_content(message.get("content"), "tool content")]


def one_off_blocks(messages: list[dict]) -> list[dict]:
    """Flatten an OpenAI message list into ordered text/image blocks for a disposable SDK turn.

    A single user turn (system/developer aside, which the caller assembles itself) is passed
    through as raw blocks. Longer histories keep the original order with explicit ``[user]`` /
    ``[assistant]`` / ``[tool result for <id>]`` markers and each assistant tool call's id,
    name and arguments. Images survive only as base64 blocks on user and tool messages;
    reasoning content, reasoning signatures and remote image URLs are not forwarded (the last
    two raise instead of being dropped).
    """
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("messages must be a list of message objects")
    turns = [message for message in messages if message.get("role") not in _SYSTEM_ROLES]
    if len(turns) == 1 and turns[0].get("role") == "user":
        return _strict_content(turns[0].get("content"), "user content")

    blocks: list[dict] = []
    for message in turns:
        role = message.get("role")
        if role == "user":
            blocks.append(_marker("[user]"))
            blocks.extend(_strict_content(message.get("content"), "user content"))
        elif role == "assistant":
            blocks.extend(_assistant_blocks(message))
        elif role == "tool":
            blocks.extend(_tool_blocks(message))
        else:
            raise ValueError(f"unsupported message role {role!r}")
    return blocks
