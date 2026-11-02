# Changelog

## 0.2.0

- Removed the runtime dependency on another provider plugin.
- Added an owned model catalog, CLI login checks, SDK model discovery, and native account usage reporting.
- Added disposable SDK sessions for auxiliary work, no-session calls, iteration-limit summaries, and JSON-schema output.
- Preserved image payloads in one-off history and validated structured results against the requested schema.
- Added timeout, cancellation, process-cleanup, and main-session-isolation checks for disposable calls.
- Kept disposable calls outside the persistent process cap, accepted host-only auth routing metadata, and added locally validated JSON-object mode.

## 0.1.0

Initial experimental release.

- Persistent Claude Code processes through the official Python Agent SDK.
- Hermes-owned tool execution, approvals, conversation storage, and compaction.
- Streaming text and thinking, steering, native transcript resume, and rewind forks.
- Auxiliary request routing through Claude Subscription DirectSDK.
- Idle shutdown, process-cap enforcement, transcript ownership locks, and bounded cancellation.
- Recovery from invalid native bindings and repeated native API errors.
- `glibglorper` client entrypoint attribution.
- Offline SDK protocol and regression tests.

Live manual compaction, cross-family model switching, and macOS support remain unqualified.
