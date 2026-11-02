# Claude Agent Session for Hermes

An experimental model-provider plugin that keeps one Claude Code process alive between Hermes model steps. Hermes runs the tools, checks approvals, stores the conversation, and compresses context. The official Python Claude Agent SDK handles the connection to Claude Code.

Provider ID: `claude-agent-session-experimental`.

## Requirements

- Hermes Agent 0.21.4 or newer with `ProviderProfile.create_client()` and `build_api_kwargs_extras()` support.
- The official Claude Code CLI, installed and logged in with `claude auth login`.
- [Claude Subscription DirectSDK](https://github.com/NousResearch/hermes-official-plugins/tree/main/claude-subscription-directsdk) installed in the same Hermes profile. This plugin uses its model catalog and sends auxiliary requests through it.
- Python packages supplied by Hermes: `anyio>=4.0.0`, `jsonschema>=4.20.0`, `mcp>=1.23.0,<3`, and `sniffio>=1.0.0`. Python 3.10 also needs `typing-extensions>=4.0.0`. These must be available to Hermes's runtime interpreter.
- Linux. Unix file locking is required; native Windows is unsupported. macOS is declared in the manifest but has not been qualified.

The pure-Python `claude-agent-sdk` 0.2.176 is vendored under `_vendor/`. No Claude Code executable or credentials are included.

Subscription access, usage limits, and billing follow Anthropic's current plan rules. This plugin does not enforce an account-level spending cap.

## Install

Run these commands for the Hermes profile that will use the provider:

```sh
hermes plugins install claude-subscription-directsdk
hermes plugins install Zioyre/hermes-claude-session
claude auth login
```

For a named profile, put `--profile <name>` before `plugins`. Start a new CLI or restart the gateway after installing or changing plugin code. Do not restart a gateway from an agent turn running inside it.

Select **Claude Agent Session (Experimental)** in the model picker, or run:

```sh
hermes chat --provider claude-agent-session-experimental --model sonnet
```

Select `claude-subscription-directsdk-experimental` to return to the stateless provider. Neither provider changes the Hermes conversation database format.

## Settings

Merge optional settings into the target profile's `config.yaml`:

```yaml
plugins:
  entries:
    claude-agent-session-experimental:
      settings:
        max_live_processes: 2
        idle_seconds: 300
        parked_idle_seconds: 1800
        step_timeout_seconds: 900
        claude_config_dir: ""
        claude_code_telemetry: true
```

- `max_live_processes` limits native processes per Hermes process. Idle sessions are stopped first, then sessions waiting for tool results. If every slot is generating, new sessions receive a retryable capacity error. Auxiliary DirectSDK processes are outside this cap.
- `idle_seconds` stops a process after a finished turn. The next request resumes its transcript.
- `parked_idle_seconds` stops a turn waiting for Hermes tool results. Missing conversation is restored as a text checkpoint when needed.
- `step_timeout_seconds` limits the wait between native protocol messages, not the total duration of a streaming response.
- `claude_config_dir` selects the Claude Code login directory. An empty value uses inherited `CLAUDE_CONFIG_DIR` or the CLI default.
- `claude_code_telemetry: false` disables nonessential native traffic. The default preserves Claude Code's normal feature-flag behavior.

`CLAUDE_AGENT_SESSION_COMMAND` selects an executable path. `CLAUDE_AGENT_SESSION_DIRECTSDK_DIR` selects the directory containing DirectSDK's `model_catalog.py`; it does not install or register that dependency.

Native session processes use `CLAUDE_CODE_ENTRYPOINT=glibglorper`, replacing inherited values. This is a client attribution label, not a guarantee about usage limits or billing. Auxiliary DirectSDK calls are unchanged.

## How it works

At a native tool-call boundary, the plugin returns tool calls to Hermes and keeps Claude Code waiting in an MCP handler. Hermes runs the tools and supplies their results on the next completion request, allowing the same native turn to continue.

- Text and thinking deltas stream as they arrive. Tool calls are returned after the native assistant message completes.
- Native tools, skills, project instructions, memory, and compaction are disabled. Only Hermes's bridged tools are offered.
- `/steer` text is delivered before pending tool results are released.
- Model controls update the running process. Explicit effort changes use a private, version-sensitive `apply_flag_settings` request.
- Changes to the system prompt, tools, output limit, login directory, or thinking-disable mode restart the process and resume the transcript.
- Hermes compaction markers start a fresh native session. Rewinds fork at a clean checkpoint; missing history becomes text in `<conversation-checkpoint>`.
- Earlier message edits with an unchanged later checkpoint are logged but not applied to the native transcript. The model retains the older version until a fresh native session starts.
- Auxiliary tasks, requests without a Hermes session ID, iteration-limit summaries, structured output, forced tool selection, and multiple-choice requests go through DirectSDK.
- Cancellation gives native interruption one second to complete, then kills an unresponsive child. Hermes owns retries; the native request counter observes extra requests but does not enforce DirectSDK's single-request admission boundary.

## State and privacy

Private state lives below `$HERMES_HOME/claude-agent-session/`:

- `records/` contains checkpoint hashes, native transcript IDs, and ownership locks.
- `records/edits.jsonl` includes short previews of blocked historical edits. Treat it as conversation data.
- `prompts/` contains full system prompts in files created with mode 0600.
- `cwd/` is the stable project directory used for native transcript resume.

Claude Code also stores native transcripts in its configuration directory. Deleting a Hermes checkpoint record does not delete the corresponding native transcript. Session metadata and prompt files have no automatic retention cleanup. Empty lock files may remain after the owning process exits; do not delete them to bypass an active lock.

## Status and limits

This is experimental software, not a production-readiness claim. Prior Linux testing exercised Hermes tools, steering, interruption, and native session resume with Claude Code 2.1.314 and Python 3.14. Other versions and platforms need qualification.

- Live manual compaction and switching between model families remain unqualified. Offline protocol tests cover compaction-to-fresh-session behavior.
- Catch-up history is text, not a lossless replay of reasoning or old images.
- Clearing an explicit effort setting does not reset a running process to its model default. Select another named effort or start a fresh session.
- One Hermes process owns a native transcript at a time. A second owner receives a conflict until the first retires the session or exits.
- A synchronous wrapper around the client can still block an async event loop.

## Development

The suite requires Hermes itself, including its `providers` and `agent` modules, not just the SDK dependencies. Use Hermes's Python interpreter and point `HERMES_SRC` at its source checkout if it is not installed at `~/.hermes/hermes-agent`:

```sh
export HERMES_PYTHON=/path/to/hermes/python
export HERMES_SRC=/path/to/hermes-agent
"$HERMES_PYTHON" -m pip install --target .testdeps -r requirements-dev.txt
./run-tests.sh tests/
PYTHONPATH=.testdeps "$HERMES_PYTHON" -m ruff check .
"$HERMES_PYTHON" -m compileall -q .
```

`TESTDEPS` overrides the test-only dependency directory. The suite drives the real vendored SDK against a scripted CLI peer, without Anthropic requests. No project type-checker is configured.

## License

MIT. The vendored SDK retains Anthropic's copyright and MIT license in `_vendor/claude_agent_sdk-0.2.176.dist-info/licenses/LICENSE`. This project is not an Anthropic product.
