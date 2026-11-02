"""Disposable account probes: local CLI login state, the live picker and plan usage.

All three read only what Claude Code already knows and never touch a credential file:

* ``setup_status`` runs the CLI's own ``auth status`` and returns only the parsed login state and
  plan label (never email, organization, token or config values).
* ``discover_models`` connects the vendored Agent SDK without sending a user message and reads the
  ``initialize`` handshake's model picker.
* ``fetch_account_usage`` asks the same handshake for the CLI's private ``get_usage`` control
  result, on a disposable child, within a fixed deadline.

Nothing here imports a provider plugin, starts a process at import time, or touches the persistent
session host: every probe owns its own short-lived process.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .bridge import PendingTools
    from .catalog import MODEL_METADATA, native_model
    from .live import LiveSession, load_sdk
    from .options import child_env, resolve_claude, telemetry_enabled
except ImportError:
    from bridge import PendingTools
    from catalog import MODEL_METADATA, native_model
    from live import LiveSession, load_sdk
    from options import child_env, resolve_claude, telemetry_enabled

logger = logging.getLogger("claude_agent_session")

SOURCE = "claude_code_get_usage"
TITLE = "Claude plan limits"
INSTALL_HINT = (
    "Claude Code is not installed (no `claude` on PATH or in the usual install directories). "
    "Install it with `npm install -g @anthropic-ai/claude-code`."
)
LOGIN_HINT = "Claude Code is installed but not logged in. Run `claude auth login`, then try again."
USAGE_UNSUPPORTED = "Claude Code gave no usage answer (needs a version with get_usage)"
USAGE_NO_LIMITS = "Claude Code returned no plan limits"
USAGE_NO_LOGIN = "no Claude subscription login found (run `claude` and log in)"
USAGE_TELEMETRY_OFF = "Claude Code skips the usage fetch with nonessential traffic off (claude_code_telemetry)"

# A `limits[]` kind maps to a fixed label; a model-scoped limit reads "<model> week" and any other
# kind keeps its raw name so a new limit still shows.
LIMIT_LABELS = {"session": "Current session", "weekly_all": "Current week"}
# The older top-level windows, for a response without `limits[]`.
LEGACY_WINDOWS = (
    ("five_hour", "Current session", "account"),
    ("seven_day", "Current week", "account"),
    ("seven_day_opus", "Opus week", "model"),
    ("seven_day_sonnet", "Sonnet week", "model"),
)


# -- small utilities ---------------------------------------------------------------------------
def _run_sync(coro):
    """Run a coroutine from sync code, even when the caller already owns a running event loop.

    ``asyncio.run`` cannot be nested, so the loop case hands the coroutine to a short-lived thread
    with its own loop.  The probe itself is deadline-bounded, so joining it cannot hang forever.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as error:  # noqa: BLE001 - re-raised on the caller's thread
            outcome["error"] = error

    thread = threading.Thread(target=run, name="claude-account-probe", daemon=True)
    try:
        thread.start()
        thread.join()
    except BaseException:
        coro.close()
        raise
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _resolve(command: str | Sequence[str] | None, env: dict[str, str]) -> list[str] | None:
    """An explicit command (never second-guessed) or the CLI ``resolve_claude`` finds."""
    if command:
        parts = [command] if isinstance(command, str) else [str(part) for part in command]
        if not parts:
            return None
        head = parts[0]
        found = head if os.path.isabs(head) and os.access(head, os.X_OK) else shutil.which(
            head, path=env.get("PATH") or os.defpath)
        return [found, *parts[1:]] if found else None
    found = resolve_claude(env)
    return [found] if found else None


def _auth_json(stdout: str) -> dict:
    """Parse ``auth status`` output; tolerate a log line before the JSON object."""
    text = stdout.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except ValueError:
        pass
    for line in reversed(text.splitlines()):
        if not line.strip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        return value if isinstance(value, dict) else {}
    return {}


def _plan_label(raw: Any) -> str:
    """``"pro"`` (auth status) and ``"Claude Pro"`` (handshake) name the same plan."""
    text = str(raw or "").strip()
    if not text:
        return ""
    return text if text.lower().startswith("claude") else "Claude " + text.replace("_", " ").title()


def _snapshot(provider: str, *, windows: tuple = (), details: tuple = (), plan: str | None = None,
              unavailable_reason: str | None = None):
    from agent.account_usage import AccountUsageSnapshot

    return AccountUsageSnapshot(
        provider=provider, source=SOURCE, fetched_at=datetime.now(timezone.utc), title=TITLE,
        plan=plan, windows=tuple(windows), details=tuple(details),
        unavailable_reason=unavailable_reason)


def _unavailable(provider: str, reason: str):
    return _snapshot(provider, unavailable_reason=reason)


# -- login state -------------------------------------------------------------------------------
def setup_status(command: str | Sequence[str] | None = None, env: dict[str, str] | None = None,
                 timeout: float = 20, **_: Any) -> dict:
    """``{available, logged_in, plan, detail, login_command}`` from the CLI's own ``auth status``.

    The response is parsed for ``loggedIn`` and ``subscriptionType`` only; no identity field from
    the CLI's JSON is copied out, and no credential file is opened.
    """
    base = dict(env if env is not None else os.environ)
    child = child_env(base)
    resolved = _resolve(command, child)
    if resolved is None:
        return {"available": False, "logged_in": False, "plan": "", "detail": INSTALL_HINT,
                "login_command": None}
    login_command = resolved + ["auth", "login"]
    try:
        run = subprocess.run(
            resolved + ["auth", "status"], check=False, env=child, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout)
        auth = _auth_json(run.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        auth = {}
    logged_in = auth.get("loggedIn") is True
    plan = _plan_label(auth.get("subscriptionType")) if logged_in else ""
    return {"available": True, "logged_in": logged_in, "plan": plan,
            "detail": "" if logged_in else LOGIN_HINT, "login_command": login_command}


# -- disposable SDK client ---------------------------------------------------------------------
def _sdk_options(sdk: Any, cli: str, cwd: Path, env: dict[str, str]) -> Any:
    """A neutral client: no tools, no MCP servers, no user turn, no persisted session."""
    return sdk.ClaudeAgentOptions(
        cli_path=cli, cwd=str(cwd), env=env, tools=[], setting_sources=[], strict_mcp_config=True,
        skills=[], mcp_servers={}, permission_mode="dontAsk",
        extra_args={"no-session-persistence": None})


@asynccontextmanager
async def _metadata_client(cli: str, env: dict[str, str]) -> AsyncIterator[Any]:
    """Connect a vendored-SDK client for the initialize handshake and dispose of it deterministically."""
    sdk = load_sdk()
    with tempfile.TemporaryDirectory(prefix="claude-account-probe-") as cwd:
        live = LiveSession(sdk=sdk, options=_sdk_options(sdk, cli, Path(cwd), env), offered=[],
                           pending=PendingTools(), model="", effort=None)
        clean = False
        try:
            await live.start()
            yield live._client
            clean = True
        finally:
            # A finished probe closes stdin and lets the CLI exit; a cancelled/failed one is killed.
            await live.close(force=not clean)


async def _server_info(cli: str, env: dict[str, str]) -> dict | None:
    async with _metadata_client(cli, env) as client:
        info = await client.get_server_info()
        return info if isinstance(info, dict) else None


async def _usage_response(cli: str, env: dict[str, str], timeout: float) -> tuple[dict, str | None]:
    async with _metadata_client(cli, env) as client:
        info = await client.get_server_info()
        query = getattr(client, "_query", None)
        if query is None:
            raise RuntimeError("Claude Code connection closed before get_usage")
        # The SDK returns the control response's inner body; an error subtype raises there.
        body = await query._send_control_request(
            {"subtype": "get_usage", "skip_behaviors": True}, timeout=timeout)
        if not isinstance(body, dict):
            raise TypeError("malformed get_usage response")
        account = info.get("account") if isinstance(info, dict) else None
        plan = _plan_label(account.get("subscriptionType")) if isinstance(account, dict) else None
        return body, plan or None


# -- picker ------------------------------------------------------------------------------------
def _catalog_label(model: str) -> str:
    """``claude-opus-4-8`` -> ``Opus 4.8``; a trailing snapshot date is dropped."""
    parts = model.removeprefix("claude-").split("-")
    if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
        parts = parts[:-1]
    family, version = parts[0], [part for part in parts[1:] if part.isdigit()]
    return " ".join(filter(None, [family.capitalize(), ".".join(version)]))


def _picker_rows(info: dict | None) -> list[dict] | None:
    """Initialize ``models`` rows -> ``[{id, label, note}]``, or None when there is no picker."""
    if not isinstance(info, dict):
        return None
    native = info.get("models")
    if not isinstance(native, list) or not native:
        return None
    rows = [row for row in native if isinstance(row, dict)]
    account = info.get("account") if isinstance(info.get("account"), dict) else {}
    plan = str(account.get("subscriptionType") or "").lower()
    credit_billed_on_plan = {"claude-fable-5-1"} if plan and "max" not in plan else set()
    announced = [str(row.get("resolvedModel") or row.get("value") or "") for row in rows]
    # An unpinned model gets ``[1m]`` only from the CLI itself; offered both ways it collapses onto
    # the long-context route like the pinned models do.
    long_context = {model.removesuffix("[1m]") for model in announced if model.endswith("[1m]")}
    routes: dict[str, dict] = {}
    for row, model in zip(rows, announced):
        base = model.removesuffix("[1m]")
        if not base:
            continue
        route = native_model(base)
        pinned = route in MODEL_METADATA
        if not pinned:
            # A value the CLI did not resolve (`default`, `best`) is an alias row, not a model.
            if not row.get("resolvedModel") and not base.startswith("claude-"):
                continue
            if base in long_context:
                route = base + "[1m]"
        label = str(row.get("displayName") or row.get("description") or route).split("·")[0].strip()
        note = "" if pinned else "unpinned"
        if "usage credit" in str(row.get("description") or "").lower() or base in credit_billed_on_plan:
            note = "usage credits" if pinned else "usage credits · unpinned"
        routes.setdefault(route, {"id": route, "label": label, "note": note})
    if not routes:
        return None
    # The picker names only each family's current model; the account still runs older pinned ones.
    for route, meta in MODEL_METADATA.items():
        if route in routes:
            continue
        canonical = meta["canonical_model"]
        routes[route] = {
            "id": route, "label": _catalog_label(canonical),
            "note": "usage credits" if canonical in credit_billed_on_plan else "",
        }
    return list(routes.values())


def discover_models(command: str | Sequence[str] | None = None, env: dict[str, str] | None = None,
                    timeout: float = 40, **_: Any) -> list[dict] | None:
    """The account's live picker as ``[{id, label, note}]``, or None when it cannot be read.

    Connects with no user turn: only the initialize control handshake runs.  Pinned routes the
    picker did not announce are appended after the advertised rows so older models stay visible.
    """
    base = dict(env if env is not None else os.environ)
    if not setup_status(command=command, env=base, timeout=timeout)["logged_in"]:
        return None
    child = child_env(base)
    resolved = _resolve(command, child)
    if resolved is None:
        return None
    try:
        info = _run_sync(asyncio.wait_for(_server_info(resolved[0], child), timeout))
    except Exception:
        logger.debug("claude-agent-session: model picker unavailable", exc_info=True)
        return None
    return _picker_rows(info)


# -- usage -------------------------------------------------------------------------------------
def _finite_number(value: Any) -> float | None:
    """A finite number; bools, NaN, Inf and non-numbers become None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _finite_percent(value: Any, *, fraction: bool = False) -> float | None:
    """A finite percentage, clamped to 0..100; bools, NaN, Inf and non-numbers become None."""
    number = _finite_number(value)
    if number is None:
        return None
    percent = number * 100.0 if fraction and number <= 1 else number
    return max(0.0, min(100.0, percent))


def _reset_at(value: Any) -> datetime | None:
    """A reset timestamp from ISO text or epoch seconds/milliseconds; anything else is unknown."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        seconds = float(value) / 1000.0 if abs(value) >= 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _limit_label(kind: Any, scope: dict) -> str:
    model = scope.get("model") if isinstance(scope.get("model"), dict) else {}
    name = model.get("display_name")
    fixed = LIMIT_LABELS.get(kind) if isinstance(kind, str) else None
    return fixed or (f"{name} week" if isinstance(name, str) and name else str(kind or "unknown"))


def _limits(payload: dict, window_cls: Any) -> list:
    """``limits[]`` rows with finite percentages; an unusable row is skipped, never guessed."""
    windows = []
    entries = payload.get("limits")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        percent = _finite_percent(entry.get("percent"))
        if percent is None:
            continue
        scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
        label = _limit_label(entry.get("kind"), scope)
        windows.append(window_cls(label=label, used_percent=percent,
                                  reset_at=_reset_at(entry.get("resets_at")),
                                  scope="model" if scope.get("model") else "account"))
    return windows


def _legacy_windows(payload: dict, window_cls: Any) -> list:
    windows = []
    for key, label, scope in LEGACY_WINDOWS:
        window = payload.get(key)
        if not isinstance(window, dict):
            continue
        percent = _finite_percent(window.get("utilization"), fraction=True)
        if percent is None:
            continue
        windows.append(window_cls(label=label, used_percent=percent,
                                  reset_at=_reset_at(window.get("resets_at")),
                                  scope=scope))
    return windows


def _extra_details(payload: dict) -> tuple[str, ...]:
    extra = payload.get("extra_usage")
    if not isinstance(extra, dict) or not extra.get("is_enabled"):
        return ()
    spent = _finite_number(extra.get("used_credits"))
    cap = _finite_number(extra.get("monthly_limit"))
    if spent is None or cap is None:
        return ()
    return (f"Extra usage: {spent:.2f} / {cap:.2f} {extra.get('currency') or 'USD'}",)


def fetch_account_usage(provider: str, *, command: str | Sequence[str] | None = None,
                        env: dict[str, str] | None = None, timeout: float = 8.0,
                        **_: Any):
    """Claude Code's own plan limits as a Hermes ``AccountUsageSnapshot``; never raises.

    The private ``get_usage`` control request runs on a disposable child, bounded by *timeout*.
    Missing CLI/login, unsupported builds, malformed payloads and telemetry-off all return an
    unavailable snapshot with a human-readable reason.
    """
    provider = str(provider or "claude-agent-session-experimental")
    if not telemetry_enabled():
        return _unavailable(provider, USAGE_TELEMETRY_OFF)
    base = dict(env if env is not None else os.environ)
    child = child_env(base)
    resolved = _resolve(command, child)
    if resolved is None:
        return _unavailable(provider, INSTALL_HINT)
    try:
        body, plan = _run_sync(asyncio.wait_for(_usage_response(resolved[0], child, timeout), timeout))
    except TimeoutError:
        return _unavailable(provider, "Claude Code did not answer in time")
    except Exception as error:
        logger.debug("claude-agent-session: usage probe failed", exc_info=True)
        if "timeout" in str(error).lower():
            return _unavailable(provider, "Claude Code did not answer in time")
        return _unavailable(provider, USAGE_UNSUPPORTED)
    limits = body.get("rate_limits")
    if not isinstance(limits, dict):
        if not body.get("rate_limits_available"):
            return _unavailable(provider, USAGE_NO_LOGIN)
        return _unavailable(provider, USAGE_NO_LIMITS)
    from agent.account_usage import AccountUsageWindow

    windows = _limits(limits, AccountUsageWindow) or _legacy_windows(limits, AccountUsageWindow)
    if not windows:
        return _unavailable(provider, USAGE_NO_LIMITS)
    return _snapshot(provider, windows=tuple(windows), details=_extra_details(limits), plan=plan)
