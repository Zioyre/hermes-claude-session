"""Run the public plugin in an empty home with imports of the former provider forbidden."""
import os
import subprocess
import sys
from pathlib import Path

from helpers import FAKE, fake_env

ROOT = Path(__file__).resolve().parents[1]


def test_empty_profile_needs_no_other_provider(tmp_path):
    env = fake_env(tmp_path, [[{"blocks": [{"text": "standalone"}]}]])
    env.update(HOME=str(tmp_path / "home"), HERMES_HOME=str(tmp_path / "profile"),
               CLAUDE_AGENT_SESSION_COMMAND=str(FAKE),
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    script = r'''
import importlib.abc
import importlib.util
import sys
from pathlib import Path

blocked = []
class NoFormerProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if "directsdk" in fullname.lower():
            blocked.append(fullname)
            raise AssertionError("External provider import attempted")

sys.meta_path.insert(0, NoFormerProvider())
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("standalone_plugin", root / "__init__.py",
                                            submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
profile = module.profile
assert profile.name == "claude-agent-session-experimental"
assert profile.model_aliases["sonnet"].startswith("claude-sonnet")
assert profile.fetch_models()
assert profile.setup_status()["available"]
assert profile.fetch_account_usage().provider == profile.name
c = profile.create_client()
reply = c.chat.completions.create(model="haiku", messages=[{"role":"user", "content":"hello"}])
assert reply.choices[0].message.content.strip() == "standalone"
c.close()
assert not blocked, blocked
assert not any("directsdk" in name.lower() for name in sys.modules)
assert not (Path(__import__("os").environ["HERMES_HOME"]) / "claude-agent-session").exists()
print("Standalone import, catalog, account hooks and SDK completion passed")
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT)], check=False, env=env, capture_output=True, text=True,
                            timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SDK completion passed" in result.stdout
