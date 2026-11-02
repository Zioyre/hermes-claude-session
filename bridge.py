import asyncio
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

PREFIX = "mcp__hermes__"
MAX_DESCRIPTION = 2048
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
_CANCELLED = {"content": [{"type": "text", "text": "Cancelled by user."}], "isError": True}


def short_name(name: str) -> str:
    """Claude caps MCP tool names at 64 chars, so the prefix plus this must stay under 50."""
    if len(name) <= 50 and _SAFE_NAME.fullmatch(name):
        return name
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:39]
    return sanitized + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def original_name(called: str, offered: Iterable[str]) -> str | None:
    stripped = called.removeprefix(PREFIX)
    for name in offered:
        if short_name(name) == stripped:
            return name
    return None


def describe(text: str) -> str:
    if len(text) <= MAX_DESCRIPTION:
        return text
    return text[: MAX_DESCRIPTION - 1] + "…"


@dataclass(frozen=True)
class ToolResult:
    mcp: dict
    replacement: list | None = None


def _data_image(url: object) -> tuple[str, str] | None:
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        return None
    header, data = url[len("data:") :].split(";base64,", 1)
    return header, data


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def tool_result_from_openai(content: str | list | None) -> ToolResult:
    """MCP results only carry text and base64 images; richer parts go through PostToolUse replacement."""
    if content is None or content == "":
        return ToolResult({"content": [_text_block("")], "isError": False})

    if isinstance(content, str):
        return ToolResult({"content": [_text_block(content)], "isError": False})

    mcp_blocks: list[dict] = []
    replacement_blocks: list[dict] = []
    needs_replacement = False
    for part in content:
        if isinstance(part, str):
            mcp_blocks.append(_text_block(part))
            replacement_blocks.append(_text_block(part))
            continue
        if isinstance(part, dict) and part.get("type") == "text":
            raw = part.get("text")
            text = raw if isinstance(raw, str) else ""
            mcp_blocks.append(_text_block(text))
            replacement_blocks.append(_text_block(text))
            continue
        image = part.get("image_url") if isinstance(part, dict) else None
        parsed = _data_image(image.get("url") if isinstance(image, dict) else None)
        if parsed is not None:
            mime, data = parsed
            mcp_blocks.append({"type": "image", "data": data, "mimeType": mime})
            replacement_blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
            )
            continue
        mcp_blocks.append(_text_block("Attached content follows."))
        replacement_blocks.append(
            _text_block(json.dumps(part, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        )
        needs_replacement = True

    return ToolResult(
        {"content": mcp_blocks, "isError": False},
        replacement_blocks if needs_replacement else None,
    )


@dataclass
class _Pending:
    name: str
    future: asyncio.Future[ToolResult]


class PendingTools:
    """One MCP call at a time: Claude's PreToolUse hook parks the id the tools/call handler consumes."""

    def __init__(self) -> None:
        self._entries: dict[str, _Pending] = {}
        self._next: tuple[str, str] | None = None
        self._replacements: dict[str, list | None] = {}

    def expect(self, tool_use_id: str, name: str) -> None:
        entry = self._entries.get(tool_use_id)
        if entry is None:
            self._entries[tool_use_id] = _Pending(name, asyncio.get_running_loop().create_future())
        elif entry.name != name:
            raise RuntimeError(f"mismatched tool call {tool_use_id}")

    def before(self, tool_use_id: str, name: str) -> None:
        self._next = (tool_use_id, name)

    async def call(self, name: str) -> dict:
        parked = self._next
        self._next = None
        if parked is None:
            raise RuntimeError("tool call without PreToolUse id")
        tool_use_id, expected = parked
        if name != expected:
            raise RuntimeError(f"mismatched tool call {tool_use_id}")
        entry = self._entries.get(tool_use_id)
        if entry is None:
            raise RuntimeError(f"tool call without expected id {tool_use_id}")
        result = await entry.future
        self._replacements[tool_use_id] = result.replacement
        self._entries.pop(tool_use_id, None)
        return result.mcp

    def resolve(self, tool_use_id: str, result: ToolResult) -> None:
        entry = self._entries.get(tool_use_id)
        if entry is None:
            raise KeyError(tool_use_id)
        if not entry.future.done():
            entry.future.set_result(result)

    def after(self, tool_use_id: str) -> list | None:
        return self._replacements.pop(tool_use_id, None)

    def clear(self) -> None:
        cancelled = ToolResult({"content": [dict(_CANCELLED["content"][0])], "isError": True})
        for entry in self._entries.values():
            if not entry.future.done():
                entry.future.set_result(cancelled)
        self._entries.clear()
        self._replacements.clear()
        self._next = None

    def waiting(self) -> list[str]:
        return [tool_use_id for tool_use_id, entry in self._entries.items() if not entry.future.done()]


def openai_tool_names(tools: list[dict]) -> list[str]:
    names = []
    for entry in tools or []:
        function = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def make_mcp_server(tools: list[dict], pending: PendingTools, sdk) -> dict:
    """In-process MCP server whose calls block until Hermes runs the tool.

    Built on the SDK's low-level ``build_tool_server`` instead of ``create_sdk_mcp_server`` so the
    SDK doesn't validate arguments against the schema first: a rejected call would never reach
    ``PendingTools`` while Hermes still runs the tool, and the two sides would disagree.
    """
    from claude_agent_sdk._internal._mcp_compat import build_tool_server
    from mcp.types import CallToolResult, Tool

    offered = openai_tool_names(tools)
    wire = []
    for entry in tools or []:
        function = entry.get("function") or {}
        schema = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        schema = {"type": "object", "properties": {}, **schema}
        wire.append(Tool.model_validate({
            "name": short_name(function["name"]),
            "description": describe(function.get("description") or ""),
            "inputSchema": schema,
            "annotations": {"readOnlyHint": False},
            "_meta": {"anthropic/alwaysLoad": True, "anthropic/maxResultSizeChars": 500_000},
        }))

    async def run_tool(tool_name: str, arguments: dict) -> CallToolResult:
        if original_name(tool_name, offered) is None:
            return CallToolResult.model_validate(
                {"content": [{"type": "text", "text": f"No such tool available: {tool_name}"}], "isError": True})
        try:
            result = await pending.call(PREFIX + tool_name)
        except RuntimeError as error:
            return CallToolResult.model_validate({"content": [{"type": "text", "text": str(error)}], "isError": True})
        return CallToolResult.model_validate(result)

    server = build_tool_server("hermes", "1.0.0", wire, run_tool)
    return sdk.McpSdkServerConfig(type="sdk", name="hermes", instance=server)


def make_hooks(pending: PendingTools, sdk) -> dict:
    async def pre_tool(input_data, tool_use_id, context):
        name = input_data.get("tool_name") if isinstance(input_data, dict) else None
        if tool_use_id and isinstance(name, str) and name.startswith(PREFIX):
            pending.before(tool_use_id, name)
        return {}

    async def post_tool(input_data, tool_use_id, context):
        replacement = pending.after(tool_use_id) if tool_use_id else None
        if replacement is None:
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedMCPToolOutput": replacement}}

    return {"PreToolUse": [sdk.HookMatcher(hooks=[pre_tool])], "PostToolUse": [sdk.HookMatcher(hooks=[post_tool])]}
