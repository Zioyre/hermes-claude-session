import pytest
from helpers import fake_log, open_live

import options


@pytest.mark.parametrize("inherited", [None, "sdk-py", "claude-desktop"])
async def test_entrypoint_reaches_sdk_child(tmp_path, monkeypatch, inherited):
    monkeypatch.setattr(options, "setting", lambda key, default=None: default)
    base = {"KEEP": "1"}
    if inherited is not None:
        base["CLAUDE_CODE_ENTRYPOINT"] = inherited
    env = options.child_env(base)
    assert env["CLAUDE_CODE_ENTRYPOINT"] == "glibglorper"
    assert env["KEEP"] == "1"
    assert base.get("CLAUDE_CODE_ENTRYPOINT") == inherited

    live = await open_live(tmp_path, [[{"blocks": [{"text": "ok"}]}]], **env)
    try:
        starts = [row for row in fake_log(tmp_path) if "argv" in row]
        assert len(starts) == 1
        assert starts[0]["entrypoint"] == "glibglorper"
    finally:
        await live.close()
