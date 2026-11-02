"""OpenAI-shaped completions over persistent and disposable Claude Agent SDK sessions."""
from __future__ import annotations

import asyncio
import copy
import inspect
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from . import catalog as CATALOG
    from .host import SessionHost, SessionRequest
    from .options import child_env, clamp_effort, setting
    from .params import structured_format, validate_oneoff
except ImportError:
    import catalog as CATALOG
    from host import SessionHost, SessionRequest
    from options import child_env, clamp_effort, setting
    from params import structured_format, validate_oneoff

SETTINGS = ("max_live_processes", "idle_seconds", "parked_idle_seconds", "step_timeout_seconds")
_HOST: SessionHost | None = None
_HOST_LOCK = threading.Lock()
_DONE = object()


def host() -> SessionHost:
    global _HOST
    with _HOST_LOCK:
        if _HOST is None:
            _HOST = SessionHost()
        return _HOST


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:  # noqa: BLE001
        return os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")


def _system_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        if message.get("role") in ("system", "developer"):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "\n\n".join(p for p in parts if p)


def wants_oneoff(kwargs: dict) -> bool:
    """Keep auxiliary work separate from persistent conversation state."""
    if kwargs.get("hermes_auxiliary") is True or not kwargs.get("hermes_session_id"):
        return True
    messages = kwargs.get("messages") or []
    if messages and messages[-1].get("role") == "user":
        # Hermes currently reuses main-loop kwargs for this auxiliary summary, including its session ID.
        try:
            from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST
        except ImportError:
            MAX_ITERATIONS_SUMMARY_REQUEST = None
        if MAX_ITERATIONS_SUMMARY_REQUEST is not None and messages[-1].get("content") == MAX_ITERATIONS_SUMMARY_REQUEST:
            return True
    extra = kwargs.get("extra_body") or {}
    if not isinstance(extra, dict):
        return True
    response_format = kwargs.get("response_format") or extra.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        return True
    return (kwargs.get("n", 1) != 1 or kwargs.get("tool_choice", "auto") not in ("auto", None)
            or kwargs.get("parallel_tool_calls") is False)


def build_request(kwargs: dict, *, one_shot: bool = False) -> SessionRequest:
    model = kwargs.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model is required")
    native = CATALOG.native_model(model)
    extra = kwargs.get("extra_body") or {}
    reasoning = extra.get("reasoning") if isinstance(extra, dict) else None
    effort, thinking_disabled = None, False
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False or str(reasoning.get("effort")).lower() == "none":
            if CATALOG.accepts_thinking_disable(native):
                thinking_disabled = True
            else:
                effort = "low"
        else:
            effort = clamp_effort(reasoning.get("effort"))
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or extra.get("max_tokens")
    max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
    messages = list(kwargs.get("messages") or [])
    timeout = kwargs.get("timeout", 900)
    if timeout is not None and not isinstance(timeout, (int, float)):
        timeout = getattr(timeout, "read", 900)
    return SessionRequest(
        home=_hermes_home(), session_id=str(uuid.uuid4()) if one_shot else str(kwargs["hermes_session_id"]),
        model=native, effort=effort,
        thinking_disabled=thinking_disabled, system=_system_text(messages), messages=messages,
        tools=list(kwargs.get("tools") or []), max_output_tokens=max_tokens,
        env=child_env(max_output_tokens=max_tokens),
        settings={k: v for k in SETTINGS if (v := setting(k)) is not None},
        one_shot=one_shot, output_format=structured_format(kwargs) if one_shot else None,
        timeout=timeout if one_shot else None)


class Obj(SimpleNamespace):
    def model_dump(self, **_: Any) -> Any:
        def unpack(value: Any) -> Any:
            if isinstance(value, Obj):
                return {k: unpack(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return copy.deepcopy(value)
        return unpack(self)


def obj(value: Any) -> Any:
    if isinstance(value, dict):
        return Obj(**{k: copy.deepcopy(v) if k == "native_usage" else obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [obj(v) for v in value]
    return value


def chunk(model: str, delta: dict, finish: str | None = None, usage: dict | None = None) -> Obj:
    return obj({"id": "claude-agent-session", "model": model, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": None, "tool_calls": None, **delta},
                             "finish_reason": finish}], "usage": usage})


class TurnStream:
    """Sync iterator over one step's chunks; the step runs on the session host's loop."""

    def __init__(self, client: Client, req: SessionRequest, model: str) -> None:
        self._client = client
        self._req = req
        self._model = model
        self._queue: queue.Queue = queue.Queue()
        self._finished = False
        self.response: Obj | None = None

        def emit(kind: str, text: str) -> None:
            delta = {"content": text} if kind == "text" else {"reasoning_content": text}
            self._queue.put(chunk(model, delta))

        def complete(_future) -> None:
            key = (req.home, req.session_id)
            client._active.discard(key)
            client._one_shots.discard(key)
            self._queue.put(_DONE)

        self._future = host().submit(host().run(req, emit))
        self._future.add_done_callback(complete)

    def __iter__(self) -> TurnStream:
        return self

    def __next__(self) -> Obj:
        if self._finished:
            raise StopIteration
        item = self._queue.get()
        if item is not _DONE:
            return item
        self._finished = True
        outcome = self._future.result()  # re-raises the step's error
        end, message = outcome.end, outcome.message
        finish = {"tool_calls": "tool_calls", "length": "length"}.get(end.kind, "stop")
        usage = dict(end.usage)
        usage["native_session"] = {"upstream_requests": end.upstream_requests}
        tool_calls = [dict(tc, index=i) for i, tc in enumerate(end.tool_calls)] or None
        self.response = obj({"id": end.reply_ids[-1] if end.reply_ids else "claude-agent-session",
                             "model": self._model, "object": "chat.completion",
                             "choices": [{"index": 0, "finish_reason": finish, "message": message}], "usage": usage})
        last = chunk(self._model, {"content": None, "tool_calls": tool_calls}, finish, usage)
        last._response = self.response
        return last

    def close(self) -> None:
        if not self._finished:
            host().interrupt(self._req.home, self._req.session_id)

    def collect(self) -> Obj:
        for _ in self:
            pass
        assert self.response is not None
        return self.response

    def __enter__(self) -> TurnStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncTurnStream:
    def __init__(self, stream: TurnStream) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncTurnStream:
        return self

    async def __anext__(self) -> Obj:
        def advance() -> Any:
            try:
                return next(self._stream)
            except StopIteration:
                return _DONE
        item = await asyncio.to_thread(advance)
        if item is _DONE:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._stream.close()


class Client:
    def __init__(self, **client_kwargs: Any) -> None:
        self._default_timeout = client_kwargs.get("timeout", 900)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self._active: set[tuple[str, str]] = set()
        self._one_shots: set[tuple[str, str]] = set()

    def create(self, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._create(**kwargs)
        caller = sys._getframe(1).f_code.co_flags
        if caller & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_ITERABLE_COROUTINE):
            return self._acreate(**kwargs)
        return self._create(**kwargs)

    async def _acreate(self, **kwargs: Any) -> Any:
        result = await asyncio.to_thread(self._create, **kwargs)
        return AsyncTurnStream(result) if isinstance(result, TurnStream) else result

    def _create(self, **kwargs: Any) -> Any:
        one_shot = wants_oneoff(kwargs)
        if one_shot:
            kwargs.setdefault("timeout", self._default_timeout)
            validate_oneoff(kwargs)
        req = build_request(kwargs, one_shot=one_shot)
        key = (req.home, req.session_id)
        self._active.add(key)
        if one_shot:
            self._one_shots.add(key)
        stream = TurnStream(self, req, kwargs["model"])
        return stream if kwargs.get("stream") else stream.collect()

    def cancel(self) -> None:
        for home, session_id in list(self._active):
            host().interrupt(home, session_id)

    def close(self) -> None:
        """Hermes recycles clients freely; live sessions belong to the host, not to this object."""
        for home, session_id in list(self._one_shots):
            host().interrupt(home, session_id)
