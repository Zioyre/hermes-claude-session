"""One long-lived Claude Code process for one Hermes session, driven step by step.

A step is one model call as Hermes sees it: it ends at a top-level ``message_stop`` after an
offered tool call (the native turn stays suspended inside the blocked MCP call) or at the turn's
``ResultMessage``. All methods run on the session host's event loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    from .bridge import PREFIX, PendingTools, ToolResult, original_name
    from .convert import UsageTracker
except ImportError:
    from bridge import PREFIX, PendingTools, ToolResult, original_name
    from convert import UsageTracker

VENDOR = Path(__file__).resolve().parent / "_vendor"
_API_REQUEST = "[API REQUEST] /v1/messages"


def load_sdk():
    """Import the vendored claude-agent-sdk (pure Python, pinned) ahead of anything installed."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import claude_agent_sdk
    return claude_agent_sdk


class ToolServerUnavailable(RuntimeError):
    pass


class SessionClosed(RuntimeError):
    pass


EventKind = Literal["text", "thinking"]
EndKind = Literal["tool_calls", "stop", "length", "error", "interrupted"]


@dataclass
class StepEnd:
    kind: EndKind
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict] = field(default_factory=list)     # OpenAI shape, Hermes tool names
    usage: dict = field(default_factory=dict)
    reply_ids: list[str] = field(default_factory=list)       # transcript uuids of assistant messages
    tool_result_ids: list[str] = field(default_factory=list)  # transcript uuids of tool_result messages
    result: Any = None
    upstream_requests: int = 0
    native_error: str | None = None   # AssistantMessage.error code (rate_limit, billing_error, ...)
    structured_output: Any = None


class LiveSession:
    def __init__(self, *, sdk: Any, options: Any, offered: list[str], pending: PendingTools, model: str,
                 effort: str | None) -> None:
        self.sdk = sdk
        self.options = options
        self.offered = offered
        self.pending = pending
        self.model = model
        self.effort = effort
        self.session_id: str | None = getattr(options, "resume", None) or getattr(options, "session_id", None)
        self.in_turn = False
        self.closed = False
        self._client: Any = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._requests = 0
        self._interrupting = False
        self._checked_init = False
        self.stderr_tail: list[str] = []

    def on_stderr(self, line: str) -> None:
        if _API_REQUEST in line:
            self._requests += 1
        self.stderr_tail = (self.stderr_tail + [line])[-40:]

    async def start(self) -> None:
        self._client = self.sdk.ClaudeSDKClient(self.options)
        await self._client.connect()
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for message in self._client.receive_messages():
                await self._queue.put(message)
        except Exception as error:  # noqa: BLE001 - surfaced to the step that's waiting
            await self._queue.put(error)
        await self._queue.put(SessionClosed("Claude Code exited"))

    @property
    def waiting(self) -> list[str]:
        return self.pending.waiting()

    @property
    def alive(self) -> bool:
        return not self.closed and self._reader is not None and not self._reader.done()

    async def send_user(self, content: str | list, *, uuid: str, priority: str | None = None) -> None:
        message: dict[str, Any] = {"type": "user", "message": {"role": "user", "content": content},
                                   "parent_tool_use_id": None, "uuid": uuid}
        if priority:
            message["priority"] = priority
        else:
            self.in_turn = True

        async def one():
            yield message

        await self._client.query(one(), session_id=self.session_id or "default")

    def resolve(self, results: dict[str, ToolResult]) -> None:
        for tool_use_id, result in results.items():
            self.pending.resolve(tool_use_id, result)

    async def set_model(self, model: str) -> None:
        if model != self.model:
            await self._client.set_model(model)
            self.model = model

    async def set_effort(self, effort: str | None) -> None:
        if effort and effort != self.effort:
            await self._client._query._send_control_request(
                {"subtype": "apply_flag_settings", "settings": {"effortLevel": effort}})
            self.effort = effort

    def _canonical_tool_name(self, name: str) -> str:
        bare = name.removeprefix(PREFIX)
        return PREFIX + bare if original_name(bare, self.offered) is not None else name

    async def step(self, emit: Callable[[EventKind, str], None], timeout: float | None = None) -> StepEnd:
        """Read until the next tool boundary or the end of the native turn."""
        sdk = self.sdk
        usage = UsageTracker()
        end = StepEnd("stop")
        streamed_text: dict[str, str] = {}
        saw_offered_tool = False
        start_requests = self._requests
        message_id = ""
        while True:
            item = await asyncio.wait_for(self._queue.get(), timeout)
            if isinstance(item, Exception):
                self.closed = True
                self.in_turn = False
                raise item
            if isinstance(item, sdk.SystemMessage):
                if item.subtype == "init":
                    self._check_init(item.data)
                continue
            if isinstance(item, sdk.StreamEvent):
                if item.parent_tool_use_id is not None:
                    continue
                event = item.event
                kind = event.get("type")
                if kind == "message_start":
                    message_id = event.get("message", {}).get("id", "")
                    usage.update(message_id, event.get("message", {}).get("usage") or {})
                elif kind == "message_delta":
                    usage.update(message_id, event.get("usage") or {})
                elif kind == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        streamed_text[message_id] = streamed_text.get(message_id, "") + delta.get("text", "")
                        end.text += delta.get("text", "")
                        emit("text", delta.get("text", ""))
                    elif delta.get("type") == "thinking_delta":
                        end.thinking += delta.get("thinking", "")
                        emit("thinking", delta.get("thinking", ""))
                elif kind == "message_stop" and saw_offered_tool:
                    end.kind = "tool_calls"
                    break
                continue
            if isinstance(item, sdk.AssistantMessage):
                if item.parent_tool_use_id is not None:
                    continue
                if getattr(item, "error", None):
                    end.native_error = str(item.error)
                    continue
                if getattr(item, "uuid", None):
                    end.reply_ids.append(item.uuid)
                if item.usage and item.message_id:
                    usage.update(item.message_id, item.usage)
                for block in item.content:
                    if isinstance(block, sdk.ToolUseBlock):
                        name = self._canonical_tool_name(block.name)
                        hermes_name = original_name(name, self.offered)
                        if hermes_name is None:
                            continue
                        self.pending.expect(block.id, name)
                        saw_offered_tool = True
                        end.tool_calls.append({"id": block.id, "type": "function", "function": {
                            "name": hermes_name,
                            "arguments": json.dumps(block.input, separators=(",", ":"), ensure_ascii=False)}})
                    elif isinstance(block, sdk.TextBlock) and item.message_id not in streamed_text:
                        end.text += block.text
                        emit("text", block.text)
                continue
            if isinstance(item, sdk.UserMessage):
                content = item.content if isinstance(item.content, list) else []
                if any(isinstance(b, sdk.ToolResultBlock) for b in content) and getattr(item, "uuid", None):
                    end.tool_result_ids.append(item.uuid)
                continue
            if isinstance(item, sdk.ResultMessage):
                self.in_turn = False
                end.result = item
                end.structured_output = getattr(item, "structured_output", None)
                self.session_id = item.session_id or self.session_id
                if self._interrupting:
                    end.kind = "interrupted"
                elif item.is_error or item.subtype != "success" or (getattr(item, "api_error_status", None) or 0) >= 400:
                    end.kind = "error"
                elif item.stop_reason in ("max_tokens", "model_context_window_exceeded"):
                    end.kind = "length"
                else:
                    end.kind = "stop"
                self._interrupting = False
                break
        end.usage = usage.openai_usage()
        end.upstream_requests = self._requests - start_requests
        return end

    def _check_init(self, data: dict) -> None:
        self.session_id = data.get("session_id") or self.session_id
        servers = data.get("mcp_servers") or []
        if self.offered and not any(s.get("name") == "hermes" and s.get("status") == "connected" for s in servers):
            raise ToolServerUnavailable("The Hermes tool server did not connect to Claude Code")
        self._checked_init = True

    async def interrupt_and_drain(self, timeout: float = 20) -> None:
        """Interrupt, wait for the turn's result (transcript flushed), then cancel blocked tool calls."""
        if self.in_turn and self.alive:
            self._interrupting = True
            await self._client.interrupt()
            try:
                await asyncio.wait_for(self._drain(), timeout)
            except (asyncio.TimeoutError, SessionClosed):
                pass
        self.pending.clear()
        self.in_turn = False
        self._interrupting = False

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            if isinstance(item, Exception):
                raise SessionClosed(str(item))
            if isinstance(item, self.sdk.ResultMessage):
                return

    def interrupt_soon(self) -> None:
        """From the step's own flow (cancel): ask Claude Code to stop; the step sees the result."""
        if self.in_turn and self.alive and not self._interrupting:
            self._interrupting = True
            asyncio.ensure_future(self._client.interrupt())
            self.pending.clear()

    async def close(self, *, force: bool = False) -> None:
        if self.closed and self._client is None:
            return
        self.closed = True
        self.pending.clear()
        process = getattr(getattr(self._client, "_transport", None), "_process", None)
        if force and process is not None and process.returncode is None:
            # Pinned SDK internals. After an ignored interrupt, do not repeat its EOF grace period.
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            if self._client is not None:
                await asyncio.wait_for(self._client.disconnect(), 10)
        except Exception:
            logging.getLogger("claude_agent_session").debug("disconnect failed", exc_info=True)
        if process is not None and process.returncode is None:
            # asyncio cancellation can interrupt the SDK before its own terminate/kill escalation.
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 2)
            except Exception:
                logging.getLogger("claude_agent_session").warning("native process did not reap after kill", exc_info=True)
        if self._reader is not None:
            self._reader.cancel()
        self._client = None
