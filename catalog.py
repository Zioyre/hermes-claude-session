"""Locally owned Claude model routes: aliases, context windows and thinking rules.

The table is the single source of truth; ``MODEL_METADATA`` and ``MODEL_CAPABILITIES`` are derived
from it.  A route the table does not know is passed through untouched (and therefore has no pinned
metadata), so a model the CLI ships before this table does keeps working instead of being guessed
at.  No provider-plugin import, config override or external catalog file is involved.
"""
from __future__ import annotations

# Native route -> context window.  A 1M route is addressed with the ``[1m]`` suffix by Hermes.
CONTEXT_WINDOWS = {
    "claude-sonnet-5-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-haiku-5-5": 1_000_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-5-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-fable-5-1": 1_000_000,
}
ALIASES = {
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-1",
}
# Families that answer ``thinking: {"type": "disabled"}`` with a 400.  A caller's disable is omitted
# and the request asks for the lowest effort instead.
MANDATORY_THINKING = ("claude-fable", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
# Haiku 4.5 also rejects ``thinking: {"type": "adaptive"}``; its effort signal still applies.
NO_ADAPTIVE_THINKING = frozenset({"claude-haiku-4-5-20251001"})


def _route_base(base: str) -> str:
    """The catalogued id a short name points at, or the name unchanged when it points at none.

    ``sonnet-5-5`` and ``sonnet-5.5`` both name ``claude-sonnet-5-5``.  A name that matches nothing
    is returned as typed, so an unknown route still reaches the CLI and fails there rather than
    being hidden by a guess.
    """
    if base in ALIASES or base in CONTEXT_WINDOWS:
        return base
    dotted = base.replace(".", "-")
    if dotted != base:
        return _route_base(dotted)
    without_vendor = base.removeprefix("claude-")
    if "claude-" + without_vendor in CONTEXT_WINDOWS or "claude-" + without_vendor in ALIASES:
        return "claude-" + without_vendor
    return base


def native_model(model: str) -> str:
    """Resolve an accepted alias/route to the id sent to Claude Code, adding ``[1m]`` when pinned."""
    base = model.removesuffix("[1m]")
    base = _route_base(base)
    canonical = ALIASES.get(base, base)
    window = CONTEXT_WINDOWS.get(canonical)
    if window == 1_000_000:
        return canonical + "[1m]"
    if window == 200_000:
        if model.endswith("[1m]"):
            raise ValueError("Haiku 4.5 does not support a 1M context window")
        return canonical
    # Unknown route: pass the caller's id through untouched, including any [1m] selection.
    return model


def accepts_thinking_disable(model: str | None) -> bool:
    """True unless the route's family rejects ``thinking: {"type": "disabled"}``."""
    base = _route_base(model.removesuffix("[1m]")) if isinstance(model, str) else ""
    return not ALIASES.get(base, base).startswith(MANDATORY_THINKING)


def supports_adaptive_thinking(model: str | None) -> bool:
    """True unless the route is known to reject ``thinking: {"type": "adaptive"}``."""
    base = _route_base(model.removesuffix("[1m]")) if isinstance(model, str) else ""
    return ALIASES.get(base, base) not in NO_ADAPTIVE_THINKING


MODEL_METADATA = {
    native_model(model): {"canonical_model": model, "context_window": window}
    for model, window in CONTEXT_WINDOWS.items()
}


def _session_ids(canonical: str) -> set[str]:
    """Every id a session may carry for a pinned route (Hermes looks capabilities up by exact id)."""
    aliases = [alias for alias, target in ALIASES.items() if target == canonical]
    ids = {canonical, native_model(canonical), *aliases}
    if CONTEXT_WINDOWS[canonical] == 1_000_000:
        ids.update(alias + "[1m]" for alias in aliases)
    return ids


MODEL_CAPABILITIES = {
    model_id: {"supports_vision": True, "context_window": window}
    for canonical, window in CONTEXT_WINDOWS.items()
    for model_id in _session_ids(canonical)
}
