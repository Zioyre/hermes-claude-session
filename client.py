"""OpenAI-shaped client Hermes calls once per model step; routes to the session host or to DirectSDK."""
from __future__ import annotations

import asyncio
import copy
import importlib.util
import inspect
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from . import errors
    from .host import SessionHost, SessionRequest
    from .options import child_env, clamp_effort, setting
except ImportError:
    import errors
    from host import SessionHost, SessionRequest
    from options import child_env, clamp_effort, setting

logger = logging.getLogger("claude_agent_session")
DIRECTSDK = "claude-subscription-directsdk-experimental"
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


# -- DirectSDK catalog (model aliases, context windows) loaded by path, never as a package import --
def _directsdk_dir() -> Path | None:
    candidates = []
    if os.environ.get("CLAUDE_AGENT_SESSION_DIRECTSDK_DIR"):
        candidates.append(Path(os.environ["CLAUDE_AGENT_SESSION_DIRECTSDK_DIR"]))
    home = os.environ.get("HERMES_HOME")
    if home:
        candidates.append(Path(home) / "plugins" / DIRECTSDK)
    candidates.append(Path.home() / ".hermes" / "plugins" / DIRECTSDK)
    return next((c for c in candidates if (c / "model_catalog.py").is_file()), None)


def load_catalog() -> Any:
    found = _directsdk_dir()
    if found is not None:
        spec = importlib.util.spec_from_file_location("claude_agent_session_catalog", found / "model_catalog.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                return module
            except Exception:
                logger.warning("claude-agent-session: could not load DirectSDK model catalog", exc_info=True)
    return SimpleNamespace(ALIASES={"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"},
                           MODEL_METADATA={}, MODEL_CAPABILITIES={}, native_model=lambda m: m,
                           accepts_thinking_disable=lambda m: False)


CATALOG = load_catalog()


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


def wants_directsdk(kwargs: dict) -> bool:
    """Calls that aren't one Hermes session's main loop stay stateless on DirectSDK."""
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
    response_format = kwargs.get("response_format") or (kwargs.get("extra_body") or {}).get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (None, "text"):
        return True
    return kwargs.get("n", 1) != 1 or kwargs.get("tool_choice", "auto") not in ("auto", None)


def build_request(kwargs: dict) -> SessionRequest:
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
    return SessionRequest(
        home=_hermes_home(), session_id=str(kwargs["hermes_session_id"]), model=native, effort=effort,
        thinking_disabled=thinking_disabled, system=_system_text(messages), messages=messages,
        tools=list(kwargs.get("tools") or []), max_output_tokens=max_tokens,
        env=child_env(max_output_tokens=max_tokens),
        settings={k: v for k in SETTINGS if (v := setting(k)) is not None})


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

        self._future = host().submit(host().run(req, emit))
        self._future.add_done_callback(lambda _f: self._queue.put(_DONE))

    def __iter__(self) -> TurnStream:
        return self

    def __next__(self) -> Obj:
        if self._finished:
            raise StopIteration
        item = self._queue.get()
        if item is not _DONE:
            return item
        self._finished = True
        self._client._active.discard((self._req.home, self._req.session_id))
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
    def __init__(self, **_: Any) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self._active: set[tuple[str, str]] = set()
        self._aux: Any = None
        self._lock = threading.Lock()

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
        if wants_directsdk(kwargs):
            return self._directsdk().chat.completions.create(
                **{k: v for k, v in kwargs.items() if k != "hermes_session_id"})
        req = build_request(kwargs)
        self._active.add((req.home, req.session_id))
        stream = TurnStream(self, req, kwargs["model"])
        return stream if kwargs.get("stream") else stream.collect()

    def _directsdk(self) -> Any:
        with self._lock:
            if self._aux is None:
                try:
                    from providers import get_provider_profile
                    profile = get_provider_profile(DIRECTSDK)
                except Exception:  # noqa: BLE001
                    profile = None
                client = profile.create_client() if profile is not None else None
                if client is None:
                    raise errors.ClaudeAPIError(
                        "claude-agent-session sends auxiliary calls through the "
                        f"{DIRECTSDK} plugin, which isn't installed", 400)
                self._aux = client
            return self._aux

    def cancel(self) -> None:
        for home, session_id in list(self._active):
            host().interrupt(home, session_id)
        if self._aux is not None and callable(getattr(self._aux, "cancel", None)):
            self._aux.cancel()

    def close(self) -> None:
        """Hermes recycles clients freely; live sessions belong to the host, not to this object."""
        if self._aux is not None:
            try:
                self._aux.close()
            except Exception:
                logger.debug("claude-agent-session: closing the DirectSDK client failed", exc_info=True)
            self._aux = None
