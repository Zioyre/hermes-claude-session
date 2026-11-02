"""Session host: one asyncio loop thread owning every live Claude Code process of this Hermes process.

Hermes calls ``chat.completions.create`` once per model step from its own threads; each call becomes
``SessionHost.run`` on the loop. Per Hermes session a lock serialises steps, a record (on disk) says
what Claude Code has already seen, and a LiveSession holds the process between steps.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    from . import errors
    from .bridge import PendingTools, make_hooks, make_mcp_server, openai_tool_names, tool_result_from_openai
    from .convert import user_content_to_blocks
    from .live import LiveSession, StepEnd, ToolServerUnavailable, load_sdk
    from .options import LaunchSpec, build_options, child_env, env_fingerprint, resolve_claude, write_private
    from .store import RecordStore
    from .sync import Group, SessionRecord, catch_up_text, decide, groups, rebuild_checkpoints
except ImportError:
    import errors
    from bridge import PendingTools, make_hooks, make_mcp_server, openai_tool_names, tool_result_from_openai
    from convert import user_content_to_blocks
    from live import LiveSession, StepEnd, ToolServerUnavailable, load_sdk
    from options import LaunchSpec, build_options, child_env, env_fingerprint, resolve_claude, write_private
    from store import RecordStore
    from sync import Group, SessionRecord, catch_up_text, decide, groups, rebuild_checkpoints

logger = logging.getLogger("claude_agent_session")
CONTINUE = "Continue from where the conversation above left off."


@dataclass(frozen=True)
class SessionRequest:
    home: str                      # HERMES_HOME of the calling profile
    session_id: str                # Hermes session id
    model: str                     # native model id
    effort: str | None
    thinking_disabled: bool
    system: str
    messages: list[dict]
    tools: list[dict]
    max_output_tokens: int | None = None
    env: dict | None = None        # child env, built on the caller's thread (profile-scoped settings)
    settings: dict = field(default_factory=dict)


@dataclass
class TurnOutcome:
    end: StepEnd
    message: dict                  # the OpenAI assistant message Hermes will store


@dataclass
class _Entry:
    lock: asyncio.Lock
    live: LiveSession | None = None
    launch_key: str | None = None
    last_used: float = 0.0
    idle: asyncio.TimerHandle | None = None
    busy: bool = False
    home: str = ""
    session_id: str = ""
    cancelled: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    lease: int | None = None
    settings: dict = field(default_factory=dict)


@dataclass
class _Plan:
    blocks: list[dict]
    history_base: list = field(default_factory=list)
    trailing_index: int | None = None     # group index of the trailing user run's first group
    resume: str | None = None
    new_session: str | None = None
    fresh_process: bool = False


def reply_message(end: StepEnd) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": end.text or None}
    if end.tool_calls:
        message["tool_calls"] = end.tool_calls
    if end.thinking:
        message["reasoning_content"] = end.thinking
    return message


def _int(settings: dict, key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def _root(home: str) -> Path:
    return Path(home) / "claude-agent-session"


def _canonical_tools(tools: list[dict]) -> tuple[str, ...]:
    return tuple(json.dumps(t, sort_keys=True, separators=(",", ":")) for t in tools or [])


def _user_blocks(messages: list[dict], g: list[Group]) -> list[dict]:
    blocks: list[dict] = []
    for group in g:
        for message in messages[group.start:group.end]:
            if message.get("role") == "user":
                blocks.extend(user_content_to_blocks(message.get("content") or ""))
    return blocks


class SessionHost:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, cli_path: str | None = None,
                 env_override: dict[str, str] | None = None) -> None:
        self._clock = clock
        self._cli_path = cli_path
        self._env_override = env_override
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Lock()
        self._launch_lock = asyncio.Lock()
        self._sdk: Any = None

    # -- loop thread -------------------------------------------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._started:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                ready = threading.Event()

                def run() -> None:
                    asyncio.set_event_loop(loop)
                    ready.set()
                    loop.run_forever()

                self._thread = threading.Thread(target=run, name="claude-agent-session", daemon=True)
                self._thread.start()
                ready.wait()
                self._loop = loop
        return self._loop

    def submit(self, coro) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())

    def interrupt(self, home: str, session_id: str) -> None:
        entry = self._entries.get((home, session_id))
        if entry is None or self._loop is None:
            return

        def stop() -> None:
            entry.cancelled = True
            entry.cancel_event.set()
            if entry.live is not None:
                entry.live.interrupt_soon()

        self._loop.call_soon_threadsafe(stop)

    def shutdown(self, timeout: float = 15) -> None:
        if self._loop is None:
            return

        async def close_all() -> None:
            for entry in list(self._entries.values()):
                async with entry.lock:
                    await self._close(entry)
                    self._release(entry)

        try:
            self.submit(close_all()).result(timeout)
        except Exception:
            logger.debug("claude-agent-session: shutdown did not finish cleanly", exc_info=True)

    # -- turn ----------------------------------------------------------------------------------
    def _entry(self, req: SessionRequest) -> _Entry:
        key = (req.home, req.session_id)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(lock=asyncio.Lock(), home=req.home, session_id=req.session_id)
            self._entries[key] = entry
        return entry

    async def run(self, req: SessionRequest, emit: Callable[[str, str], None]) -> TurnOutcome:
        entry = self._entry(req)
        async with entry.lock:
            entry.busy = True
            entry.cancelled = False
            entry.cancel_event.clear()
            entry.settings = dict(req.settings)
            if entry.idle is not None:
                entry.idle.cancel()
                entry.idle = None
            try:
                return await self._run_cancellable(entry, req, emit)
            finally:
                entry.busy = False
                entry.last_used = self._clock()
                if entry.live is None:
                    self._release(entry)
                self._arm_idle(entry)

    async def _run_cancellable(self, entry: _Entry, req: SessionRequest, emit) -> TurnOutcome:
        task = asyncio.create_task(self._run_locked(entry, req, emit))
        cancelled = asyncio.create_task(entry.cancel_event.wait())
        try:
            done, _ = await asyncio.wait({task, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if cancelled not in done:
                return await task
            # Let a cooperative native interrupt flush its transcript before forcing disconnect.
            finished, _ = await asyncio.wait({task}, timeout=1)
            if not finished:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await self._close(entry, force=True)
                store = RecordStore(_root(req.home) / "records")
                record = store.load(req.session_id)
                if record is not None:
                    record.interrupted = True
                    store.save(req.session_id, record)
            else:
                # Retrieve the native cancellation/error so no task exception is left unobserved.
                await asyncio.gather(task, return_exceptions=True)
            raise RuntimeError(errors.CANCELLED)
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)

    async def _run_locked(self, entry: _Entry, req: SessionRequest, emit) -> TurnOutcome:
        store = RecordStore(_root(req.home) / "records")
        if entry.lease is None:
            try:
                entry.lease = store.acquire(req.session_id)
            except BlockingIOError as error:
                raise errors.ClaudeAPIError("This Claude session is owned by another Hermes process", 409) from error
        record = store.load(req.session_id)
        if self._env_override is not None:
            env = dict(self._env_override)
        else:
            env = dict(req.env) if req.env is not None else child_env(max_output_tokens=req.max_output_tokens)
        spec = LaunchSpec(req.system, _canonical_tools(req.tools), req.max_output_tokens, req.thinking_disabled,
                          env_fingerprint(env))
        g = groups(req.messages)

        live = entry.live
        if live is not None and (not live.alive or entry.launch_key != spec.key):
            record = await self._retire(entry, store, record)
            live = None

        if live is not None and live.waiting:
            outcome = await self._try_resume_parked(entry, req, store, record, g, emit)
            if outcome is not None:
                return outcome
            record = await self._retire(entry, store, record)
        elif live is not None and live.in_turn:
            record = await self._retire(entry, store, record)

        plan = self._plan(req, store, record, g, entry)
        if plan.fresh_process and entry.live is not None:
            await self._retire(entry, store, None)
        if entry.live is None:
            async with self._launch_lock:
                await self._evict_for_room(entry)
                entry.live = await self._open(req, env, spec, resume=plan.resume, session_id=plan.new_session)
                entry.launch_key = spec.key
        if entry.cancelled:
            raise RuntimeError(errors.CANCELLED)
        live = entry.live
        await live.set_model(req.model)
        await live.set_effort(None if req.thinking_disabled else req.effort)
        user_uuid = str(uuid.uuid4())
        await live.send_user(plan.blocks, uuid=user_uuid)
        claude_sid = live.session_id or plan.resume or plan.new_session or ""
        record = SessionRecord(claude_session_id=claude_sid, cwd=str(_root(req.home) / "cwd"), history=plan.history_base)
        new_ids = {} if plan.trailing_index is None else {plan.trailing_index: user_uuid}
        return await self._step_and_save(entry, req, store, record, g, emit, new_ids)

    async def _try_resume_parked(self, entry, req, store, record, g, emit) -> TurnOutcome | None:
        """Tool results for the suspended native turn: resolve the blocked MCP calls, no new user message."""
        live = entry.live
        if record is None or record.history is None or record.interrupted:
            return None
        decision = decide(req.messages, record)
        if decision.kind not in ("continue", "catch_up"):
            return None
        missed = [grp for grp in g[decision.matched:]]
        if not missed or missed[0].role != "tool":
            return None
        results = {}
        for message in req.messages[missed[0].start:missed[0].end]:
            if message.get("role") == "tool":
                results[message.get("tool_call_id")] = message.get("content")
        if not set(live.waiting) <= set(results) or any(grp.role != "user" for grp in missed[1:]):
            return None
        for grp in missed[1:]:
            blocks = _user_blocks(req.messages, [grp])
            if blocks:
                await live.send_user(blocks, uuid=str(uuid.uuid4()), priority="next")
        await live.set_model(req.model)
        await live.set_effort(None if req.thinking_disabled else req.effort)
        live.resolve({tid: tool_result_from_openai(results[tid]) for tid in live.waiting})
        return await self._step_and_save(entry, req, store, record, g, emit, {},
                                         tool_group_index=decision.matched)

    def _plan(self, req: SessionRequest, store: RecordStore, record: SessionRecord | None, g: list[Group],
              entry: _Entry) -> _Plan:
        interrupted = bool(record and record.interrupted)
        decision = decide(req.messages, replace(record, interrupted=False) if record and record.history is not None else None)
        if record is not None and record.history is None:
            decision = decide(req.messages, None)
        if decision.kind == "edited":
            for edit in decision.edits:
                store.log_edit({"session": req.session_id, "group": edit.group_index, "old": edit.old_hash,
                                "new": edit.new_hash, "preview": edit.preview})
            logger.info("claude-agent-session: %d earlier message(s) changed in %s; kept Claude's transcript",
                        len(decision.edits), req.session_id)

        plan = _Plan(blocks=[])
        new_session = record is None or decision.kind in ("fresh", "compacted")
        if new_session:
            if record is not None:
                logger.info("claude-agent-session: %s -> new Claude session (%s)", req.session_id, decision.kind)
            plan.fresh_process = True
            plan.new_session = str(uuid.uuid4())
            first_new = 0
        elif decision.kind == "rewind":
            assert record is not None and record.history is not None
            sdk = self._load_sdk()
            try:
                fork = sdk.fork_session(record.claude_session_id, directory=record.cwd or None,
                                        up_to_message_id=decision.fork_at)
            except Exception as error:
                store.delete(req.session_id)
                raise errors.ClaudeAPIError(f"Native rewind unavailable; rebuild on retry: {error}", 503) from error
            plan.resume = fork.session_id
            fork_index = next(i for i, c in enumerate(record.history) if c.claude_message_id == decision.fork_at)
            first_new = fork_index + 1
            plan.history_base = [replace(c, claude_message_id=None) for c in record.history[:first_new]]
            plan.fresh_process = True
        else:
            assert record is not None
            plan.resume = None if entry.live is not None else record.claude_session_id
            plan.history_base = list(record.history or [])
            first_new = next((i for i, grp in enumerate(g) if grp.start >= decision.start), len(g))

        trailing = len(g)
        while trailing > first_new and g[trailing - 1].role == "user":
            trailing -= 1
        catch_groups = g[first_new:trailing]
        if interrupted:
            while catch_groups and catch_groups[0].role == "assistant":
                catch_groups = catch_groups[1:]
        blocks: list[dict] = []
        if catch_groups:
            text = catch_up_text(req.messages, catch_groups[0].start, catch_groups[-1].end)
            if text:
                blocks.append({"type": "text", "text": text})
        user = _user_blocks(req.messages, g[trailing:])
        if user:
            blocks.extend(user)
            plan.trailing_index = trailing
        elif blocks:
            blocks.append({"type": "text", "text": CONTINUE})
        if not blocks:
            if record is None:
                raise errors.ClaudeAPIError("claude-agent-session: no message to send for this step", 400)
            blocks.append({"type": "text", "text": CONTINUE})
        plan.blocks = blocks
        if new_session or decision.kind == "rewind":
            plan.history_base = plan.history_base[:first_new] if decision.kind == "rewind" else []
        return plan

    async def _step_and_save(self, entry, req, store, record, g, emit, new_ids,
                             tool_group_index: int | None = None) -> TurnOutcome:
        live = entry.live
        try:
            end = await live.step(emit, timeout=_int(req.settings, "step_timeout_seconds", 900))
        except ToolServerUnavailable as error:
            await self._close(entry)
            raise errors.ClaudeAPIError(str(error), 500) from error
        except asyncio.CancelledError:
            record.history = rebuild_checkpoints(g, record.history, new_ids)
            record.interrupted = True
            store.save(req.session_id, record)
            raise
        except Exception as error:
            stderr = list(live.stderr_tail)
            await self._close(entry)
            if not live._checked_init and getattr(live.options, "resume", None):
                store.delete(req.session_id)
            else:
                record.interrupted = True
                store.save(req.session_id, record)
            raise errors.ClaudeAPIError(f"Claude Code stopped unexpectedly: {error}"
                                        f"{' | ' + stderr[-1][-200:] if stderr else ''}", 503) from error
        if entry.cancelled:
            end.kind = "interrupted"
        if end.upstream_requests > 1:
            logger.warning("claude-agent-session: one step made %d upstream requests (%s)",
                           end.upstream_requests, req.session_id)
        message = reply_message(end)
        record.claude_session_id = live.session_id or record.claude_session_id
        if tool_group_index is not None and end.tool_result_ids:
            new_ids[tool_group_index] = end.tool_result_ids[-1]
        if end.kind in ("tool_calls", "stop", "length"):
            g_all = groups(req.messages + [message])
            if end.reply_ids:
                new_ids[len(g_all) - 1] = end.reply_ids[-1]
            record.history = rebuild_checkpoints(g_all, record.history, new_ids)
            record.interrupted = False
            store.save(req.session_id, record)
            return TurnOutcome(end, message)
        if end.kind == "interrupted":
            record.history = rebuild_checkpoints(g, record.history, new_ids)
            record.interrupted = True
            store.save(req.session_id, record)
            raise RuntimeError(errors.CANCELLED)
        # Native result confirms the input was consumed. Retry with CONTINUE, not duplicate user/tool rows.
        record.history = rebuild_checkpoints(g, record.history, new_ids)
        store.save(req.session_id, record)
        raise errors.from_step(end, live.stderr_tail)

    # -- processes ---------------------------------------------------------------------------
    def _load_sdk(self):
        if self._sdk is None:
            self._sdk = load_sdk()
        return self._sdk

    async def _open(self, req: SessionRequest, env: dict, spec: LaunchSpec, *, resume: str | None,
                    session_id: str | None) -> LiveSession:
        sdk = self._load_sdk()
        cli = self._cli_path or resolve_claude(env)
        if cli is None:
            raise errors.ClaudeCodeMissing(errors.INSTALL_HINT)
        root = _root(req.home)
        cwd = root / "cwd"
        cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
        system_file = write_private(root / "prompts" / (hashlib.sha256(req.system.encode()).hexdigest()[:24] + ".md"),
                                    req.system)
        pending = PendingTools()
        holder: list[LiveSession] = []

        def stderr(line: str) -> None:
            if holder:
                holder[0].on_stderr(line)

        options = build_options(
            sdk=sdk, cli_path=cli, cwd=cwd, model=req.model,
            effort=None if req.thinking_disabled else req.effort, thinking_disabled=req.thinking_disabled,
            system_prompt_file=system_file, mcp_server=make_mcp_server(req.tools, pending, sdk),
            hooks=make_hooks(pending, sdk), env=env, resume=resume, session_id=session_id, stderr=stderr)
        live = LiveSession(sdk=sdk, options=options, offered=openai_tool_names(req.tools), pending=pending,
                           model=req.model, effort=None if req.thinking_disabled else req.effort)
        holder.append(live)
        try:
            await live.start()
        except BaseException as error:
            await live.close(force=isinstance(error, asyncio.CancelledError))
            if isinstance(error, Exception):
                if resume:
                    # Only the native binding is discarded. Hermes still owns the full history.
                    RecordStore(_root(req.home) / "records").delete(req.session_id)
                raise errors.ClaudeAPIError(f"Claude Code could not start: {error}", 503) from error
            raise
        return live

    @staticmethod
    def _release(entry: _Entry) -> None:
        if entry.lease is not None:
            RecordStore.release(entry.lease)
            entry.lease = None

    async def _close(self, entry: _Entry, *, force: bool = False) -> None:
        if entry.live is not None:
            live, entry.live = entry.live, None
            await live.close(force=force)
        if not entry.busy:
            self._release(entry)

    async def _retire(self, entry: _Entry, store: RecordStore, record: SessionRecord | None) -> SessionRecord | None:
        """Stop a running/parked native turn cleanly; the next step catches Claude up from the record."""
        live = entry.live
        if live is not None and (live.waiting or live.in_turn):
            await live.interrupt_and_drain()
            if record is not None:
                record.interrupted = True
                store.save(entry.session_id, record)
        await self._close(entry)
        return record

    async def _evict_for_room(self, current: _Entry) -> None:
        limit = max(1, _int(current.settings, "max_live_processes", 2))
        live = [e for e in self._entries.values() if e.live is not None and e is not current]
        while len(live) >= limit:
            idle = [e for e in live if not e.busy and not e.lock.locked()]
            if not idle:
                raise errors.ClaudeAPIError("Claude session process limit reached; all slots are busy", 503)
            victim = min(idle, key=lambda e: (bool(e.live and e.live.waiting), e.last_used))
            async with victim.lock:
                store = RecordStore(_root(victim.home) / "records")
                await self._retire(victim, store, store.load(victim.session_id))
            live.remove(victim)

    def _arm_idle(self, entry: _Entry) -> None:
        if entry.live is None or self._loop is None:
            return
        parked = bool(entry.live.waiting)
        delay = (_int(entry.settings, "parked_idle_seconds", 1800) if parked
                 else _int(entry.settings, "idle_seconds", 300))

        def fire() -> None:
            asyncio.ensure_future(self._idle_stop(entry))

        entry.idle = self._loop.call_later(max(1, delay), fire)

    async def _idle_stop(self, entry: _Entry) -> None:
        if entry.busy or entry.lock.locked() or entry.live is None:
            return
        async with entry.lock:
            store = RecordStore(_root(entry.home) / "records")
            await self._retire(entry, store, store.load(entry.session_id))
            entry.idle = None
