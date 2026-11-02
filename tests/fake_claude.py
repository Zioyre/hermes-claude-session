#!/usr/bin/env python3
"""Scripted stand-in for the `claude` CLI speaking the SDK's stream-json control protocol.

Shapes follow real Claude Code 2.1.314 recordings (tests/tools/record.py). A JSON script picked by
FAKE_CLAUDE_SCRIPT drives it: a list of turns, one per user message that starts a turn; each turn a
list of API calls; each call a list of blocks ({"text"}, {"thinking"}, {"tool_use": {"name", "input"}})
plus optional "stop_reason", "error" (api error status) and "slow" (seconds between text deltas).
FAKE_CLAUDE_LOG (jsonl) records every line received plus argv, for test assertions.
FAKE_CLAUDE_NO_MCP=1 reports the hermes MCP server as failed.
"""
import json
import os
import queue
import sys
import threading
import time
import uuid

ARGV = sys.argv[1:]
if ARGV[:1] in (["-v"], ["--version"]):
    print("2.1.314 (Claude Code)")
    sys.exit(0)

LOG = open(os.environ["FAKE_CLAUDE_LOG"], "a", buffering=1) if os.environ.get("FAKE_CLAUDE_LOG") else None
SCRIPT = json.load(open(os.environ["FAKE_CLAUDE_SCRIPT"])) if os.environ.get("FAKE_CLAUDE_SCRIPT") else [[{"blocks": [{"text": "ok"}]}]]
DEBUG = "--debug-to-stderr" in ARGV
out_lock = threading.Lock()
inbox: "queue.Queue[dict]" = queue.Queue()
responses: dict[str, "queue.Queue[dict]"] = {}
interrupted = threading.Event()


def arg(name):
    for i, item in enumerate(ARGV):
        if item == name and i + 1 < len(ARGV):
            return ARGV[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


SESSION = arg("--resume") or arg("--session-id") or str(uuid.uuid4())
if LOG:
    LOG.write(json.dumps({"argv": ARGV, "entrypoint": os.environ.get("CLAUDE_CODE_ENTRYPOINT"),
                          "pid": os.getpid(), "cwd": os.getcwd()}) + "\n")


def send(obj):
    with out_lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if LOG:
            LOG.write(json.dumps({"in": msg}) + "\n")
        if msg.get("type") == "control_response":
            rid = msg["response"].get("request_id")
            responses.setdefault(rid, queue.Queue()).put(msg["response"])
        elif msg.get("type") == "control_request" and msg["request"].get("subtype") != "initialize":
            if msg["request"].get("subtype") == "interrupt" and os.environ.get("FAKE_CLAUDE_IGNORE_INTERRUPT") != "1":
                interrupted.set()
            send({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"],
                                                           "response": {}}})
        else:
            inbox.put(msg)
    inbox.put({"type": "_eof"})


def request(payload, timeout=30, interruptible=False):
    rid = str(uuid.uuid4())
    box = responses.setdefault(rid, queue.Queue())
    send({"type": "control_request", "request_id": rid, "request": payload})
    deadline = time.monotonic() + timeout
    while True:
        try:
            resp = box.get(timeout=0.05)
            break
        except queue.Empty:
            if interruptible and interrupted.is_set():
                return None
            if time.monotonic() > deadline:
                raise
    if resp.get("subtype") != "success":
        raise RuntimeError(f"control request failed: {resp}")
    return resp.get("response") or {}


hooks: dict[str, list[str]] = {}
tools: list[str] = []


def mcp(method, params=None, rid=None, interruptible=False):
    message = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if rid is not None:
        message["id"] = rid
    resp = request({"subtype": "mcp_message", "server_name": "hermes", "message": message},
                   timeout=600 if interruptible else 30, interruptible=interruptible)
    return None if resp is None else resp.get("mcp_response", {})


def handle_initialize(msg):
    for event, matchers in (msg["request"].get("hooks") or {}).items():
        hooks[event] = [cid for m in matchers for cid in m.get("hookCallbackIds", [])]
    send({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"],
                                                   "response": {"commands": [], "models": []}}})


def connect_mcp():
    if os.environ.get("FAKE_CLAUDE_NO_MCP") == "1" or "--mcp-config" not in ARGV:
        return "failed"
    mcp("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                       "clientInfo": {"name": "claude-code", "version": "2.1.314"}}, rid=0)
    mcp("notifications/initialized")
    listed = mcp("tools/list", rid=1)
    tools.extend("mcp__hermes__" + t["name"] for t in listed.get("result", {}).get("tools", []))
    return "connected"


def hook(event, payload):
    for cid in hooks.get(event, []):
        request({"subtype": "hook_callback", "callback_id": cid,
                 "input": {"session_id": SESSION, "hook_event_name": event, **payload},
                 "tool_use_id": payload.get("tool_use_id")})


def usage(output=5):
    return {"input_tokens": 4, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10,
            "output_tokens": output}


def stream(event):
    send({"type": "stream_event", "uuid": str(uuid.uuid4()), "session_id": SESSION, "event": event,
          "parent_tool_use_id": None})


def assistant(mid, block):
    send({"type": "assistant", "message": {"model": "claude-haiku-5-5", "id": mid, "type": "message",
                                           "role": "assistant", "content": [block], "stop_reason": None,
                                           "usage": usage()},
          "parent_tool_use_id": None, "session_id": SESSION, "uuid": str(uuid.uuid4())})


def result(subtype="success", is_error=False, text="", num_turns=1, api_error_status=None, stop_reason="end_turn",
           structured_output=None):
    payload = {"type": "result", "subtype": subtype, "is_error": is_error, "duration_ms": 10,
               "duration_api_ms": 10, "num_turns": num_turns, "result": text, "stop_reason": stop_reason,
               "session_id": SESSION, "total_cost_usd": 0.0, "usage": usage(), "uuid": str(uuid.uuid4())}
    if structured_output is not None:
        payload["structured_output"] = structured_output
    if api_error_status:
        payload["api_error_status"] = api_error_status
        payload["errors"] = [f"API Error: {api_error_status}"]
    send(payload)


def interrupted_result(num_turns):
    send({"type": "user", "message": {"role": "user", "content": [{"type": "text",
                                                                  "text": "[Request interrupted by user]"}]},
          "parent_tool_use_id": None, "session_id": SESSION, "uuid": str(uuid.uuid4())})
    result("error_during_execution", True, num_turns=num_turns, stop_reason=None)


def service_inbox(steer_box):
    """Mid-turn: queued user messages steer the turn; control requests still get answered."""
    while not inbox.empty():
        pending = inbox.get()
        if pending.get("type") == "user":
            steer_box.append(_text(pending))
        elif pending.get("type") == "control_request":
            send({"type": "control_response", "response": {"subtype": "success",
                                                           "request_id": pending["request_id"], "response": {}}})
        elif pending.get("type") == "_eof":
            inbox.put(pending)
            return


def run_turn(turn, steer_box):
    interrupted.clear()
    texts = []
    for n, call in enumerate(turn, start=1):
        if call.get("exit"):
            sys.exit(call["exit"])
        service_inbox(steer_box)
        if DEBUG:
            sys.stderr.write("2026-10-31T00:00:00.000Z [DEBUG] [API REQUEST] /v1/messages source=sdk\n")
            sys.stderr.flush()
        if call.get("error"):
            result("success", True, num_turns=n, api_error_status=call["error"], stop_reason=None)
            return
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        stream({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant",
                                                       "model": "claude-haiku-5-5", "content": [], "usage": usage(1)}})
        calls = []
        for index, block in enumerate(call["blocks"]):
            if "tool_use" in block:
                tid = f"toolu_{uuid.uuid4().hex[:20]}"
                name = "mcp__hermes__" + block["tool_use"]["name"]
                stream({"type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}})
                stream({"type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["tool_use"]["input"])}})
                assistant(mid, {"type": "tool_use", "id": tid, "name": name, "input": block["tool_use"]["input"]})
                stream({"type": "content_block_stop", "index": index})
                calls.append((tid, name, block["tool_use"]["input"]))
                continue
            kind = "thinking" if "thinking" in block else "text"
            text = block[kind]
            if kind == "text" and steer_box:
                text = text + " " + " ".join(steer_box)
                steer_box.clear()
            stream({"type": "content_block_start", "index": index, "content_block": {"type": kind, kind: ""}})
            delta_type = "thinking_delta" if kind == "thinking" else "text_delta"
            for word in text.split(" "):
                if interrupted.is_set():
                    break
                stream({"type": "content_block_delta", "index": index, "delta": {"type": delta_type, kind: word + " "}})
                time.sleep(call.get("slow", 0))
            assistant(mid, {"type": kind, kind: text, **({"signature": "fake-signature"} if kind == "thinking" else {})})
            stream({"type": "content_block_stop", "index": index})
            if kind == "text":
                texts.append(text)
            if interrupted.is_set():
                interrupted.clear()
                stream({"type": "message_stop"})
                interrupted_result(n)
                return
        stop = call.get("stop_reason") or ("tool_use" if calls else "end_turn")
        stream({"type": "message_delta", "delta": {"stop_reason": stop}, "usage": usage()})
        stream({"type": "message_stop"})
        for tid, name, tool_input in calls:
            hook("PreToolUse", {"tool_name": name, "tool_input": tool_input, "tool_use_id": tid})
            resp = mcp("tools/call", {"name": name.removeprefix("mcp__hermes__"), "arguments": tool_input},
                       rid=uuid.uuid4().int % 10_000, interruptible=True)
            if resp is None or interrupted.is_set():
                interrupted.clear()
                interrupted_result(n)
                return
            res = resp.get("result", {})
            hook("PostToolUse", {"tool_name": name, "tool_input": tool_input, "tool_use_id": tid,
                                 "tool_response": res})
            send({"type": "user", "message": {"role": "user", "content": [
                {"tool_use_id": tid, "type": "tool_result", "content": res.get("content", []),
                 "is_error": res.get("isError", False)}]},
                  "parent_tool_use_id": None, "session_id": SESSION, "uuid": str(uuid.uuid4())})
            service_inbox(steer_box)
        if not calls:
            result("success", False, text=" ".join(texts), num_turns=n,
                   stop_reason=call.get("stop_reason") or "end_turn", structured_output=call.get("structured_output"))
            return
    result("success", False, text=" ".join(texts), num_turns=len(turn))


def _text(msg):
    content = msg["message"]["content"]
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def main():
    threading.Thread(target=reader, daemon=True).start()
    turns = list(SCRIPT)
    mcp_status = None
    while True:
        msg = inbox.get()
        kind = msg.get("type")
        if kind == "_eof":
            return
        if kind == "control_request":
            sub = msg["request"].get("subtype")
            if sub == "initialize":
                handle_initialize(msg)
                continue
            send({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"],
                                                           "response": {}}})
            continue
        if kind != "user":
            continue
        if mcp_status is None:
            mcp_status = connect_mcp()
            send({"type": "system", "subtype": "init", "cwd": os.getcwd(), "session_id": SESSION,
                  "tools": list(tools), "mcp_servers": [{"name": "hermes", "status": mcp_status, "source": "sdk"}],
                  "model": arg("--model") or "claude-haiku-5-5", "permissionMode": "dontAsk",
                  "uuid": str(uuid.uuid4())})
        turn = turns.pop(0) if turns else [{"blocks": [{"text": "ok"}]}]
        run_turn(turn, [])


if __name__ == "__main__":
    main()
