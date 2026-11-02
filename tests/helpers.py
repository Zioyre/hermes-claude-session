"""Shared builders for tests that drive LiveSession against tests/fake_claude.py."""
import json
import os
from pathlib import Path

from bridge import PendingTools, make_hooks, make_mcp_server, openai_tool_names
from live import LiveSession, load_sdk
from options import build_options

FAKE = Path(__file__).with_name("fake_claude.py")
ECHO = {"type": "function", "function": {"name": "echo", "description": "Echo x.",
                                         "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}


def fake_env(tmp_path: Path, script: list, **extra) -> dict:
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))
    env = dict(os.environ, FAKE_CLAUDE_SCRIPT=str(script_path), FAKE_CLAUDE_LOG=str(tmp_path / "fake.jsonl"),
               CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK="1")
    env.update(extra)
    return env


def fake_log(tmp_path: Path) -> list[dict]:
    path = tmp_path / "fake.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def open_live(tmp_path: Path, script: list, tools=(ECHO,), model="haiku", effort=None, resume=None,
                    session_id="11111111-1111-4111-8111-111111111111", **extra_env) -> LiveSession:
    sdk = load_sdk()
    pending = PendingTools()
    tools = list(tools)
    system = tmp_path / "system.md"
    system.write_text("You are a test.")
    live: LiveSession | None = None

    def stderr(line: str) -> None:
        if live is not None:
            live.on_stderr(line)

    options = build_options(
        sdk=sdk, cli_path=str(FAKE), cwd=tmp_path, model=model, effort=effort, thinking_disabled=False,
        system_prompt_file=system, mcp_server=make_mcp_server(tools, pending, sdk), hooks=make_hooks(pending, sdk),
        env=fake_env(tmp_path, script, **extra_env), resume=resume, session_id=None if resume else session_id,
        stderr=stderr)
    live = LiveSession(sdk=sdk, options=options, offered=openai_tool_names(tools), pending=pending, model=model,
                       effort=effort)
    await live.start()
    return live
