"""Claude Code launch options: isolation, environment and the key that forces a relaunch."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .catalog import supports_adaptive_thinking
except ImportError:
    from catalog import supports_adaptive_thinking

PLUGIN_ID = "claude-agent-session-experimental"

# Claude Code features that duplicate Hermes's own; Hermes's version wins.
ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "DISABLE_AUTO_COMPACT": "1",
    "DISABLE_COMPACT": "1",
    "CLAUDE_CODE_ENABLE_AWAY_SUMMARY": "0",
    "ENABLE_TOOL_SEARCH": "false",
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
    "MAX_MCP_OUTPUT_TOKENS": "200000",
    "CLAUDE_CODE_MAX_RETRIES": "0",
    "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
}
QUIET_TRAFFIC = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
                 "DISABLE_ERROR_REPORTING": "1"}
# CLAUDE_CODE_SIMPLE skips OAuth; the API-key/backend overrides would move billing off the plan;
# an inherited effort level or MCP timeout would override Hermes's settings.
UNSET_ENV = ("CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_EFFORT_LEVEL", "MCP_TOOL_TIMEOUT", "CLAUDE_CODE_EXTRA_BODY",
             "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_FOUNDRY_API_KEY",
             "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
             "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
_OFF = ("false", "0", "no", "off")
_HOME_PREFIXES = (".local/bin", ".claude/local", "bin", ".npm-global/bin", ".bun/bin", ".volta/bin")
_SYSTEM_PREFIXES = ("/opt/homebrew/bin", "/usr/local/bin")


def setting(key: str, default: Any = None) -> Any:
    """Read plugins.entries.<id>.settings.<key> from the active profile, per call."""
    try:
        from hermes_cli.config import load_config_readonly
        value: Any = load_config_readonly()
    except Exception:  # noqa: BLE001 - unreadable config means "not set"
        return default
    for segment in ("plugins", "entries", PLUGIN_ID, "settings", key):
        if not isinstance(value, Mapping) or segment not in value:
            return default
        value = value[segment]
    return value


def telemetry_enabled() -> bool:
    value = setting("claude_code_telemetry", True)
    if isinstance(value, str):
        return value.strip().lower() not in _OFF
    return value is not False and value != 0


def resolve_claude(env: Mapping[str, str]) -> str | None:
    """Explicit override, then PATH, then the usual install dirs (gateways often run with a thin PATH)."""
    override = env.get("CLAUDE_AGENT_SESSION_COMMAND")
    if override:
        return override if os.access(override, os.X_OK) else None
    found = shutil.which("claude", path=env.get("PATH") or os.defpath)
    if found:
        return found
    home = env.get("HOME") or os.path.expanduser("~")
    prefixes = [os.path.join(home, p) for p in _HOME_PREFIXES] + list(_SYSTEM_PREFIXES)
    return shutil.which("claude", path=os.pathsep.join(prefixes))


def _login_identity(env: dict[str, str]) -> None:
    """Without USER a logged-in Claude Code reports loggedIn: false under supervised gateways."""
    name = env.get("USER") or os.environ.get("USER") or ""
    if not name:
        try:
            import pwd
            name = pwd.getpwuid(os.getuid()).pw_name
        except (ImportError, KeyError):
            name = ""
    if name:
        env.setdefault("USER", name)
        env.setdefault("LOGNAME", name)


def child_env(base: Mapping[str, str] | None = None, *, max_output_tokens: int | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for key in UNSET_ENV:
        env.pop(key, None)
    _login_identity(env)
    config_dir = setting("claude_config_dir")
    config_dir = (config_dir.strip() if isinstance(config_dir, str) else "") or env.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(config_dir)
    env.update(ISOLATION_ENV)
    env["CLAUDE_CODE_ENTRYPOINT"] = "glibglorper"
    if not telemetry_enabled():
        env.update(QUIET_TRAFFIC)
    if max_output_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    return env


def clamp_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    effort = str(effort).strip().lower()
    if effort in ("", "none", "off", "minimal"):
        return "low"
    return effort if effort in EFFORTS else "high"


@dataclass(frozen=True)
class LaunchSpec:
    """Everything that can't change on a running Claude Code process."""
    system: str
    tools: tuple[str, ...]          # canonical JSON of each OpenAI tool definition
    max_output_tokens: int | None
    thinking_disabled: bool
    env_fingerprint: str

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps([self.system, self.tools, self.max_output_tokens, self.thinking_disabled,
                                          self.env_fingerprint]).encode()).hexdigest()


def env_fingerprint(env: Mapping[str, str]) -> str:
    keys = ("CLAUDE_CONFIG_DIR", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    return hashlib.sha256(json.dumps({k: env.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:16]


def write_private(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(tmp, path)
    return path


def build_options(*, sdk: Any, cli_path: str, cwd: Path, model: str, effort: str | None, thinking_disabled: bool,
                  system_prompt_file: Path, mcp_server: Any, hooks: dict, env: dict[str, str],
                  resume: str | None = None, session_id: str | None = None,
                  stderr: Callable[[str], None] | None = None, output_format: dict | None = None,
                  one_shot: bool = False) -> Any:
    """ClaudeAgentOptions with every native feature off except the Hermes MCP tools."""
    thinking = {"type": "disabled"} if thinking_disabled else {"type": "adaptive", "display": "summarized"}
    if not thinking_disabled and not supports_adaptive_thinking(model):
        thinking = None
    kwargs: dict[str, Any] = dict(
        cli_path=cli_path, cwd=str(cwd), model=model, env=env, tools=[], setting_sources=[],
        strict_mcp_config=True, skills=[],
        permission_mode="dontAsk", allowed_tools=["mcp__hermes__*"], verbatim_prompts=True,
        include_partial_messages=True, mcp_servers={"hermes": mcp_server}, hooks=hooks, thinking=thinking,
        system_prompt={"type": "file", "path": str(system_prompt_file)},
        extra_args={"debug-to-stderr": None}, stderr=stderr,
    )
    if output_format is not None:
        kwargs["output_format"] = output_format
    if one_shot:
        kwargs["extra_args"]["no-session-persistence"] = None
        kwargs["max_turns"] = 3 if output_format else 1
    if effort and not thinking_disabled:
        kwargs["effort"] = effort
    if resume:
        kwargs["resume"] = resume
    elif session_id:
        kwargs["session_id"] = session_id
    return sdk.ClaudeAgentOptions(**kwargs)
