"""Errors Hermes can route: its retry/fallback ladder reads ``status_code``."""
from __future__ import annotations

from typing import Any

LOGGED_OUT_HINT = "Claude Code is not logged in. Run `claude auth login` on the machine running Hermes."
INSTALL_HINT = "Claude Code is not installed. Install it with `npm install -g @anthropic-ai/claude-code`."
CANCELLED = "Claude request cancelled"
NATIVE_ERROR_STATUS = {
    "rate_limit": 429,
    "billing_error": 402,
    "authentication_failed": 401,
    "overloaded": 529,
    "server_error": 503,
    "invalid_request": 400,
}


class ClaudeCodeMissing(RuntimeError):
    pass


class ClaudeCodeLoggedOut(RuntimeError):
    status_code = 401


class ClaudeAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def from_step(end: Any, stderr_tail: list[str] | None = None) -> Exception:
    """Turn a failed native turn into an exception carrying an HTTP-like status."""
    result = end.result
    code = end.native_error
    status = getattr(result, "api_error_status", None) if result is not None else None
    errors = list(getattr(result, "errors", None) or []) if result is not None else []
    detail = "; ".join(str(e) for e in errors)[:500] or (getattr(result, "subtype", "") if result is not None else "")
    if code == "authentication_failed":
        return ClaudeCodeLoggedOut(f"{LOGGED_OUT_HINT} (native: {detail})")
    if not isinstance(status, int) or status < 400:
        status = NATIVE_ERROR_STATUS.get(code or "")
    tail = ""
    if stderr_tail:
        tail = " | stderr: " + " / ".join(line[-200:] for line in stderr_tail[-3:])
    return ClaudeAPIError(f"Claude Code turn failed ({code or 'error'}): {detail}{tail}", status)


def classify_api_error(error: Exception, **_: Any) -> dict | None:
    # Hermes's stale-call watchdog cancels through Client.cancel(); classify that as the timeout it is.
    if isinstance(error, RuntimeError) and str(error) == CANCELLED:
        return {"reason": "timeout"}
    return None
