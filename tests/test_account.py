"""Offline probes: auth status, SDK picker discovery and private get_usage parsing.

Each test drives a throwaway scripted CLI through the real vendored SDK.  The CLI never sends an
assistant/user frame; only control handshakes appear in its log, so "metadata does not spend a
Messages request" is asserted from observed input rather than assumed.
"""
import json
import sys
from types import SimpleNamespace

import pytest

import account

FAKE_CLI = r'''
import json
import os
import sys

ARGV = sys.argv[1:]
LOG = os.environ.get("FAKE_ACCOUNT_LOG")


def log(obj):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(obj) + "\n")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


if "-v" in ARGV or "--version" in ARGV:
    print("2.1.314 (Claude Code)")
    sys.exit(0)

log({"argv": ARGV, "entrypoint": os.environ.get("CLAUDE_CODE_ENTRYPOINT"),
     "api_key": os.environ.get("ANTHROPIC_API_KEY"),
     "auth_token": os.environ.get("ANTHROPIC_AUTH_TOKEN")})

if "auth" in ARGV and "status" in ARGV:
    sys.stdout.write(os.environ.get("FAKE_AUTH_STDOUT", "{}"))
    sys.exit(int(os.environ.get("FAKE_AUTH_EXIT", "0")))

MODE = os.environ.get("FAKE_ACCOUNT_MODE", "ok")
if MODE == "crash_on_connect":
    sys.exit(3)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    log({"in": message})
    if message.get("type") != "control_request":
        continue
    request_id = message.get("request_id")
    subtype = message.get("request", {}).get("subtype")
    if subtype == "initialize":
        payload = {
            "commands": [],
            "models": json.loads(os.environ.get("FAKE_MODELS", "[]")),
            "account": json.loads(os.environ.get("FAKE_ACCOUNT", "{}")),
            "capabilities": {},
        }
        send({"type": "control_response", "response": {"subtype": "success", "request_id": request_id,
                                                       "response": payload}})
        continue
    if subtype == "get_usage":
        if MODE == "unsupported":
            send({"type": "control_response", "response": {"subtype": "error", "request_id": request_id,
                                                           "error": "unknown subtype"}})
        elif MODE == "hang":
            continue
        elif MODE == "malformed":
            send({"type": "control_response", "response": {
                "subtype": "success", "request_id": request_id,
                "response": {"rate_limits": None, "rate_limits_available": True}}})
        elif MODE == "no_login":
            send({"type": "control_response", "response": {
                "subtype": "success", "request_id": request_id,
                "response": {"rate_limits": None, "rate_limits_available": False}}})
        else:
            send({"type": "control_response", "response": {
                "subtype": "success", "request_id": request_id,
                "response": {"rate_limits": json.loads(os.environ.get("FAKE_USAGE", "{}")),
                             "rate_limits_available": True}}})
        continue
    send({"type": "control_response", "response": {"subtype": "error", "request_id": request_id,
                                                   "error": "unsupported subtype"}})

log({"event": "eof"})
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cli = tmp_path / "claude"
    cli.write_text(f"#!{sys.executable}\n{FAKE_CLI}", encoding="utf-8")
    cli.chmod(0o755)
    log = tmp_path / "fake.jsonl"
    monkeypatch.setenv("FAKE_ACCOUNT_LOG", str(log))
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setenv("FAKE_AUTH_STDOUT", json.dumps({"loggedIn": True, "subscriptionType": "pro"}))
    return SimpleNamespace(path=cli, log=log)


def entries(fake):
    if not fake.log.exists():
        return []
    return [json.loads(line) for line in fake.log.read_text(encoding="utf-8").splitlines() if line]


def inputs(fake):
    return [entry["in"] for entry in entries(fake) if "in" in entry]


def control_subtypes(fake):
    return [message["request"]["subtype"] for message in inputs(fake)
            if message.get("type") == "control_request"]


def assert_no_user_frames(fake):
    messages = inputs(fake)
    assert not [message for message in messages if message.get("type") == "user"]
    assert not [message for message in messages if message.get("type") == "assistant"]


# -- setup status ------------------------------------------------------------------------------
def test_setup_status_missing_cli_is_reported_not_raised(tmp_path):
    status = account.setup_status(command=str(tmp_path / "absent"))
    assert status["available"] is False
    assert status["logged_in"] is False
    assert status["plan"] == ""
    assert "not installed" in status["detail"]
    assert status["login_command"] is None


def test_setup_status_emits_only_login_and_plan(fake, monkeypatch):
    monkeypatch.setenv("FAKE_AUTH_STDOUT", json.dumps({
        "loggedIn": True, "subscriptionType": "pro", "email": "private@example.invalid",
        "organization": "Secret Org", "accountUuid": "private-uuid",
    }))
    status = account.setup_status(command=str(fake.path))
    assert status == {
        "available": True, "logged_in": True, "plan": "Claude Pro", "detail": "",
        "login_command": [str(fake.path), "auth", "login"],
    }
    dumped = json.dumps(status)
    for secret in ("private@example.invalid", "Secret Org", "private-uuid"):
        assert secret not in dumped


def test_setup_status_sanitizes_the_child_environment(fake, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-reach-the-cli")
    account.setup_status(command=str(fake.path))
    entry = next(entry for entry in entries(fake) if "argv" in entry)
    assert entry["api_key"] is None
    assert entry["auth_token"] is None
    assert entry["entrypoint"] == "glibglorper"


def test_setup_status_malformed_json_is_logged_out(fake, monkeypatch):
    monkeypatch.setenv("FAKE_AUTH_STDOUT", "not json at all")
    status = account.setup_status(command=str(fake.path))
    assert status["available"] is True
    assert status["logged_in"] is False
    assert status["plan"] == ""
    assert status["detail"] == account.LOGIN_HINT


def test_setup_status_logged_out_does_not_report_a_plan(fake, monkeypatch):
    monkeypatch.setenv("FAKE_AUTH_STDOUT", json.dumps({"loggedIn": False, "subscriptionType": "max"}))
    status = account.setup_status(command=str(fake.path))
    assert status["logged_in"] is False
    assert status["plan"] == ""


# -- picker ------------------------------------------------------------------------------------
def test_discover_models_is_none_when_logged_out(fake, monkeypatch):
    monkeypatch.setenv("FAKE_AUTH_STDOUT", json.dumps({"loggedIn": False}))
    assert account.discover_models(command=str(fake.path)) is None


def test_discover_models_reads_the_picker_without_a_user_turn(fake, monkeypatch):
    models = [
        {"value": "default", "displayName": "Default", "description": "Default (recommended)"},
        {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "description": "Sonnet 5.5 · balanced"},
        {"value": "fable", "resolvedModel": "claude-fable-5-1", "description": "Fable 5.1 · billed as usage credits"},
        {"value": "claude-latest-10", "displayName": "Latest 10", "description": "For everyday tasks"},
        {"value": "future", "resolvedModel": "claude-future-9", "description": "Future 9"},
        {"value": "future-long", "resolvedModel": "claude-future-9[1m]", "description": "Future 9 long"},
    ]
    monkeypatch.setenv("FAKE_MODELS", json.dumps(models))
    monkeypatch.setenv("FAKE_ACCOUNT", json.dumps({"subscriptionType": "pro"}))
    rows = account.discover_models(command=str(fake.path))
    assert rows
    by_id = {row["id"]: row for row in rows}
    assert all(set(row) == {"id", "label", "note"} for row in rows)
    assert len(by_id) == len(rows)
    assert "default" not in by_id and "best" not in by_id
    assert by_id["claude-latest-10"] == {"id": "claude-latest-10", "label": "Latest 10", "note": "unpinned"}
    assert by_id["claude-sonnet-5-5[1m]"]["note"] == ""
    assert by_id["claude-fable-5-1[1m]"]["note"] == "usage credits"
    # An unpinned id the CLI resolved is kept, collapsed onto its advertised long-context route.
    assert by_id["claude-future-9[1m]"]["note"] == "unpinned"
    assert by_id["claude-future-9[1m]"]["label"] == "Future 9"
    # Pinned routes the picker did not announce stay selectable.
    assert "claude-haiku-4-5-20251001" in by_id
    assert by_id["claude-haiku-4-5-20251001"]["label"] == "Haiku 4.5"
    assert control_subtypes(fake) == ["initialize"]
    assert_no_user_frames(fake)
    assert any(entry.get("event") == "eof" for entry in entries(fake))


def test_discover_models_falls_back_to_none_on_a_bad_handshake(fake, monkeypatch):
    monkeypatch.setenv("FAKE_ACCOUNT_MODE", "crash_on_connect")
    assert account.discover_models(command=str(fake.path), timeout=3) is None


# -- usage -------------------------------------------------------------------------------------
def usage(fake, payload, monkeypatch, mode="ok"):
    monkeypatch.setenv("FAKE_USAGE", json.dumps(payload))
    monkeypatch.setenv("FAKE_ACCOUNT_MODE", mode)
    monkeypatch.setenv("FAKE_ACCOUNT", json.dumps({"subscriptionType": "pro"}))
    return account.fetch_account_usage("claude-agent-session-experimental", command=str(fake.path))


def test_usage_limits_are_sanitized_and_resets_parsed(fake, monkeypatch):
    snapshot = usage(fake, {
        "limits": [
            {"kind": "session", "percent": 12.5, "resets_at": "2026-11-02T12:00:00Z"},
            {"kind": "weekly_all", "percent": 101, "resets_at": "2026-11-09T00:00:00+00:00"},
            {"kind": "weekly_scoped", "percent": 33.0, "resets_at": 1760000000,
             "scope": {"model": {"display_name": "Fable"}}},
            {"kind": "weekly_opus", "percent": float("inf"), "scope": {"model": {"display_name": "Opus"}}},
            {"kind": "weekly_sonnet", "percent": True},
            {"kind": "weekly_bad", "percent": "12"},
        ],
        "extra_usage": {"is_enabled": True, "used_credits": 150, "monthly_limit": 500, "currency": "USD"},
    }, monkeypatch)
    assert snapshot.provider == "claude-agent-session-experimental"
    assert snapshot.source == "claude_code_get_usage"
    assert snapshot.plan == "Claude Pro"
    assert snapshot.available is True
    assert snapshot.unavailable_reason is None
    assert [(window.label, window.used_percent, window.scope) for window in snapshot.windows] == [
        ("Current session", 12.5, "account"),
        ("Current week", 100.0, "account"),
        ("Fable week", 33.0, "model"),
    ]
    first, second, third = (window.reset_at for window in snapshot.windows)
    assert first is not None and first.isoformat() == "2026-11-02T12:00:00+00:00"
    assert second is not None and second.isoformat() == "2026-11-09T00:00:00+00:00"
    assert third is not None and int(third.timestamp()) == 1760000000
    assert snapshot.details == ("Extra usage: 150.00 / 500.00 USD",)
    assert control_subtypes(fake) == ["initialize", "get_usage"]
    assert_no_user_frames(fake)
    assert any(entry.get("event") == "eof" for entry in entries(fake))


def test_usage_legacy_windows_are_parsed(fake, monkeypatch):
    snapshot = usage(fake, {
        "five_hour": {"utilization": 0.42, "resets_at": "2026-11-02T12:00:00Z"},
        "seven_day": {"utilization": 55},
        "seven_day_opus": {"utilization": 0.1},
        "seven_day_sonnet": {"utilization": "nope"},
    }, monkeypatch)
    assert [(window.label, window.used_percent, window.scope) for window in snapshot.windows] == [
        ("Current session", 42.0, "account"),
        ("Current week", 55.0, "account"),
        ("Opus week", 10.0, "model"),
    ]
    assert snapshot.available is True


def test_usage_malformed_body_is_unavailable(fake, monkeypatch):
    snapshot = usage(fake, {}, monkeypatch, mode="malformed")
    assert snapshot.available is False
    assert snapshot.windows == ()
    assert snapshot.unavailable_reason == account.USAGE_NO_LIMITS


def test_usage_unsupported_cli_is_unavailable(fake, monkeypatch):
    snapshot = usage(fake, {}, monkeypatch, mode="unsupported")
    assert snapshot.available is False
    assert snapshot.unavailable_reason == account.USAGE_UNSUPPORTED


def test_usage_without_a_login_is_unavailable(fake, monkeypatch):
    snapshot = usage(fake, {}, monkeypatch, mode="no_login")
    assert snapshot.available is False
    assert snapshot.unavailable_reason == account.USAGE_NO_LOGIN


def test_usage_with_no_windows_is_unavailable(fake, monkeypatch):
    snapshot = usage(fake, {"limits": [{"kind": "session", "percent": float("nan")}],
                            "five_hour": {"utilization": float("inf")}}, monkeypatch)
    assert snapshot.available is False
    assert snapshot.unavailable_reason == account.USAGE_NO_LIMITS


def test_usage_telemetry_off_does_not_start_a_process(fake, monkeypatch):
    monkeypatch.setattr(account, "telemetry_enabled", lambda: False)
    snapshot = account.fetch_account_usage("claude-agent-session-experimental", command=str(fake.path))
    assert snapshot.available is False
    assert snapshot.unavailable_reason == account.USAGE_TELEMETRY_OFF
    assert entries(fake) == []


def test_usage_missing_cli_is_unavailable(tmp_path):
    snapshot = account.fetch_account_usage("claude-agent-session-experimental",
                                           command=str(tmp_path / "absent"))
    assert snapshot.available is False
    assert snapshot.unavailable_reason == account.INSTALL_HINT


def test_usage_timeout_is_bounded(fake, monkeypatch):
    monkeypatch.setenv("FAKE_ACCOUNT_MODE", "hang")
    snapshot = account.fetch_account_usage("claude-agent-session-experimental", command=str(fake.path),
                                           timeout=1.0)
    assert snapshot.available is False
    assert snapshot.unavailable_reason == "Claude Code did not answer in time"


async def test_sync_probe_bridges_out_of_a_running_loop(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODELS", json.dumps([
        {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "description": "Sonnet 5.5"},
    ]))
    rows = account.discover_models(command=str(fake.path))
    assert rows and rows[0]["id"] == "claude-sonnet-5-5[1m]"
    assert control_subtypes(fake) == ["initialize"]
    assert_no_user_frames(fake)
