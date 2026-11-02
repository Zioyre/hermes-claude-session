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
