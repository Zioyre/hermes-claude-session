import hashlib
import json
from dataclasses import dataclass
from typing import Literal

GroupRole = Literal["user", "assistant", "tool"]

_SKIP_ROLES = frozenset({"system", "developer"})
_GROUP_ROLES = frozenset({"user", "assistant", "tool"})


@dataclass(frozen=True)
class Group:
    role: GroupRole
    start: int
    end: int
    hash: str


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _data_url(url: object) -> str | None:
    if isinstance(url, str) and url.startswith("data:"):
        return url
    return None


def _text_of(part: object) -> str:
    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
        return part["text"]
    return ""


def _user_parts(content: object) -> list[dict]:
    parts = [content] if isinstance(content, str) else content if isinstance(content, list) else []
    hashed: list[dict] = []
    for part in parts:
        if isinstance(part, str):
            stripped = part.strip()
            if stripped:
                hashed.append({"t": stripped})
            continue
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                hashed.append({"t": text.strip()})
            continue
        image = part.get("image_url") if isinstance(part, dict) else None
        url = image.get("url") if isinstance(image, dict) else None
        data_url = _data_url(url)
        if data_url is not None:
            hashed.append({"img": _digest_text(data_url)})
        else:
            hashed.append({"x": _digest(part)})
    return hashed


def _assistant_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(text for text in (_text_of(part) for part in content) if text).strip()
    return ""


def _tool_arguments(arguments: object) -> tuple[object, bool]:
    if not isinstance(arguments, str):
        return arguments, True
    try:
        return json.loads(arguments), True
    except json.JSONDecodeError:
        return arguments, False


def _tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    hashed: list[dict] = []
    for entry in calls:
        if not isinstance(entry, dict):
            continue
        raw_function = entry.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        args, parsed = _tool_arguments(function.get("arguments"))
        hashed.append(
            {
                "id": entry.get("id"),
                "name": function.get("name"),
                "args": _canonical(args) if parsed else args,
            }
        )
    return hashed


def _tool_content(content: object) -> tuple[str, list[dict]]:
    if isinstance(content, str):
        return content.strip(), []
    if not isinstance(content, list):
        return "", []
    texts: list[str] = []
    images: list[dict] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
            continue
        image = part.get("image_url") if isinstance(part, dict) else None
        url = image.get("url") if isinstance(image, dict) else None
        data_url = _data_url(url)
        if data_url is not None:
            images.append({"img": _digest_text(data_url)})
        elif isinstance(part, dict):
            images.append({"x": _digest(part)})
    return "\n".join(texts).strip(), images


def member_hash(message: dict) -> str:
    """Hash only what Claude can observe, so replay noise never looks like an edit."""
    role = message.get("role")
    if role == "user":
        return _digest(_user_parts(message.get("content")))
    if role == "assistant":
        return _digest({"text": _assistant_text(message.get("content")), "calls": _tool_calls(message)})
    if role == "tool":
        text, images = _tool_content(message.get("content"))
        return _digest({"id": message.get("tool_call_id"), "content": text, "images": images})
    return _digest({"role": role, "content": message.get("content")})


def groups(messages: list[dict]) -> list[Group]:
    """Split a prompt into replay units; only the trailing user run may grow without re-forking."""
    result: list[Group] = []
    pending_role: GroupRole | None = None
    pending_start = 0
    pending_end = 0
    pending_hashes: list[str] = []

    def flush(end: int) -> None:
        nonlocal pending_role
        if pending_role is None:
            return
        result.append(
            Group(
                role=pending_role,
                start=pending_start,
                end=end,
                hash=_digest({"role": pending_role, "members": sorted(pending_hashes)}),
            )
        )
        pending_role = None

    for index, message in enumerate(messages):
        role = message.get("role")
        if role in _SKIP_ROLES or role not in _GROUP_ROLES:
            continue
        assert isinstance(role, str)
        if role == "assistant" or pending_role != role:
            flush(pending_end)
            pending_role = role  # type: ignore[assignment]
            pending_start = index
            pending_hashes = [member_hash(message)]
        else:
            pending_hashes.append(member_hash(message))
        pending_end = index + 1
    flush(pending_end)
    return result


@dataclass
class Checkpoint:
    hash: str
    claude_message_id: str | None = None

    def to_dict(self) -> dict:
        return {"hash": self.hash, "claude_message_id": self.claude_message_id}

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(hash=str(data["hash"]), claude_message_id=data.get("claude_message_id"))


@dataclass
class SessionRecord:
    claude_session_id: str
    cwd: str
    history: list[Checkpoint] | None
    interrupted: bool = False
    owner_pid: int = 0

    def to_dict(self) -> dict:
        return {
            "claude_session_id": self.claude_session_id,
            "cwd": self.cwd,
            "history": None if self.history is None else [entry.to_dict() for entry in self.history],
            "interrupted": self.interrupted,
            "owner_pid": self.owner_pid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionRecord":
        stored = data.get("history")
        return cls(
            claude_session_id=str(data.get("claude_session_id", "")),
            cwd=str(data.get("cwd", "")),
            history=None if stored is None else [Checkpoint.from_dict(entry) for entry in stored],
            interrupted=bool(data.get("interrupted", False)),
            owner_pid=int(data.get("owner_pid", 0)),
        )


DecisionKind = Literal["fresh", "continue", "catch_up", "edited", "rewind", "compacted", "resume_interrupted"]


@dataclass(frozen=True)
class Edit:
    group_index: int
    old_hash: str | None
    new_hash: str | None
    preview: str


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    start: int
    matched: int
    edits: tuple[Edit, ...] = ()
    fork_at: str | None = None


COMPACTION_PREFIXES = ("[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:")


def _message_text(message: dict) -> str:
    role = message.get("role")
    content = message.get("content")
    if role == "tool":
        return _tool_content(content)[0]
    if role == "assistant":
        return _assistant_text(content)
    if role == "user":
        parts = [content] if isinstance(content, str) else content if isinstance(content, list) else []
        texts = [part for part in parts if isinstance(part, str)]
        texts.extend(_text_of(part) for part in parts if isinstance(part, dict))
        return "\n".join(text for text in texts if text).strip()
    return ""


def _group_text(messages: list[dict], group: Group) -> str:
    texts = [text for text in (_message_text(m) for m in messages[group.start : group.end]) if text]
    return "\n".join(texts).strip()


def _open_tool_ids(messages: list[dict], scope: list[Group]) -> set[str]:
    open_ids: set[str] = set()
    for group in scope:
        for message in messages[group.start : group.end]:
            if message.get("role") == "assistant":
                for entry in message.get("tool_calls") or []:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        open_ids.add(entry["id"])
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if isinstance(call_id, str):
                    open_ids.discard(call_id)
    return open_ids


def boundary(messages: list[dict], g: list[Group], saved: list[Checkpoint], matched: int) -> str | None:
    """Fork only after a checkpoint whose tool calls were all answered; never strand Claude mid-call."""
    for index in range(min(matched, len(saved)) - 1, -1, -1):
        if not saved[index].claude_message_id:
            continue
        if not _open_tool_ids(messages, g[: index + 1]):
            return saved[index].claude_message_id
    return None


def decide(messages: list[dict], record: SessionRecord | None) -> Decision:
    if record is None:
        return Decision("fresh", 0, 0)

    g = groups(messages)
    saved = record.history
    if record.interrupted or saved is None:
        trailing_start = len(messages)
        if g and g[-1].role == "user":
            trailing_start = g[-1].start
            for group in reversed(g):
                if group.role != "user":
                    break
                trailing_start = group.start
        return Decision("resume_interrupted", trailing_start, 0)

    saved_hashes = {entry.hash for entry in saved}
    for group in g:
        if group.hash in saved_hashes:
            continue
        texts = (_message_text(message).lstrip() for message in messages[group.start : group.end])
        if any(text.startswith(COMPACTION_PREFIXES) for text in texts):
            return Decision("compacted", 0, 0)

    limit = min(len(g), len(saved))
    matched = 0
    while matched < limit and g[matched].hash == saved[matched].hash:
        matched += 1

    if matched == len(saved):
        missed = g[matched:]
        kind: DecisionKind = "catch_up" if any(group.role != "user" for group in missed) else "continue"
        return Decision(kind, missed[0].start if missed else len(messages), matched)

    tail = min(len(g), len(saved))
    if any(g[index].hash == saved[index].hash for index in range(matched + 1, tail)):
        edits = tuple(
            Edit(
                group_index=index,
                old_hash=saved[index].hash,
                new_hash=g[index].hash,
                preview=_group_text(messages, g[index])[:120],
            )
            for index in range(matched, tail)
            if g[index].hash != saved[index].hash
        )
        start = g[len(saved)].start if len(g) > len(saved) else len(messages)
        return Decision("edited", start, matched, edits)

    if matched == 0:
        return Decision("fresh", 0, 0)

    fork_at = boundary(messages, g, saved, matched)
    if fork_at is None:
        return Decision("fresh", 0, 0)
    start = g[matched].start if matched < len(g) else len(messages)
    return Decision("rewind", start, matched, fork_at=fork_at)


_CATCH_UP_HEADER = (
    "<conversation-checkpoint>\n"
    "The messages below are earlier conversation that happened while you were not attached. "
    "Treat them as prior context, not as a new request."
)
_CATCH_UP_FOOTER = "</conversation-checkpoint>"


def _render_parts(content: object) -> str:
    parts = [content] if isinstance(content, str) else content if isinstance(content, list) else []
    lines: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part.strip():
                lines.append(part.strip())
            continue
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
            continue
        image = part.get("image_url") if isinstance(part, dict) else None
        if _data_url(image.get("url") if isinstance(image, dict) else None) is not None:
            lines.append("[image]")
    return "\n".join(lines)


def _render_tool_call(entry: dict) -> str:
    raw_function = entry.get("function")
    function = raw_function if isinstance(raw_function, dict) else {}
    arguments, _parsed = _tool_arguments(function.get("arguments"))
    payload = {
        "id": entry.get("id"),
        "name": function.get("name"),
        "input": arguments,
    }
    return "<tool-call>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</tool-call>"


def _render_message(message: dict) -> str:
    role = message.get("role")
    content = message.get("content")
    if role == "user":
        text = _render_parts(content)
        return f"<user>{text}</user>" if text else ""
    if role == "assistant":
        calls = [_render_tool_call(entry) for entry in message.get("tool_calls") or [] if isinstance(entry, dict)]
        text = _assistant_text(content)
        body = "\n".join([text, *calls]) if text else "\n".join(calls)
        return f"<assistant>{body}</assistant>"
    if role == "tool":
        call_id = message.get("tool_call_id")
        text = _render_parts(content)
        return f'<tool-result id="{call_id}">{text}</tool-result>'
    return ""


def catch_up_text(messages: list[dict], start: int, end: int) -> str:
    """Catch-up text is the only way Claude learns turns it never received as native messages."""
    blocks = [block for block in (_render_message(m) for m in messages[start:end]) if block]
    if not blocks:
        return ""
    return "\n".join([_CATCH_UP_HEADER, *blocks, _CATCH_UP_FOOTER])


def rebuild_checkpoints(
    g: list[Group], previous: list[Checkpoint] | None, new_ids: dict[int, str]
) -> list[Checkpoint]:
    result: list[Checkpoint] = []
    for index, group in enumerate(g):
        carried: str | None = None
        if previous is not None and index < len(previous) and previous[index].hash == group.hash:
            carried = previous[index].claude_message_id
        result.append(Checkpoint(group.hash, new_ids.get(index, carried)))
    return result
