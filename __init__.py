"""Claude Agent Session (Experimental): one long-lived Claude Code process per Hermes session.

Registration only; nothing starts until the first model call (the multiplexed gateway imports
provider plugins for every profile).
"""
from providers import register_provider
from providers.base import ProviderProfile

try:
    from .client import CATALOG, DIRECTSDK
    from .errors import classify_api_error
except ImportError:
    from client import CATALOG, DIRECTSDK
    from errors import classify_api_error

NAME = "claude-agent-session-experimental"


def _directsdk_profile():
    try:
        from providers import get_provider_profile
        return get_provider_profile(DIRECTSDK)
    except Exception:  # noqa: BLE001
        return None


class ClaudeAgentSessionProfile(ProviderProfile):
    model_metadata = getattr(CATALOG, "MODEL_METADATA", {})

    def get_model_context_length(self, model):
        route = CATALOG.native_model(model)
        pinned = self.model_metadata.get(route, {}).get("context_window")
        return pinned or (None if route.endswith("[1m]") else 200_000)

    def create_client(self, **client_kwargs):
        try:
            from .client import Client
        except ImportError:
            from client import Client
        return Client(**client_kwargs)

    def build_api_kwargs_extras(self, *, reasoning_config=None, **context):
        extra = {"reasoning": dict(reasoning_config)} if reasoning_config else {}
        session_id = context.get("session_id")
        if context.get("task") is not None or not session_id:
            return extra, {"hermes_auxiliary": True}
        return extra, {"hermes_session_id": str(session_id)}

    # Discovery, setup checks and plan limits are the same Claude Code login: ask DirectSDK.
    def fetch_models(self, **kwargs):
        other = _directsdk_profile()
        return other.fetch_models(**kwargs) if other else list(getattr(CATALOG, "MODEL_METADATA", {}))

    def discover_models(self, **kwargs):
        other = _directsdk_profile()
        return other.discover_models(**kwargs) if other and hasattr(other, "discover_models") else None

    def setup_status(self, **kwargs):
        other = _directsdk_profile()
        return other.setup_status(**kwargs) if other and hasattr(other, "setup_status") else None

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        other = _directsdk_profile()
        if other and hasattr(other, "fetch_account_usage"):
            return other.fetch_account_usage(base_url=base_url, api_key=api_key)
        return None


profile = ClaudeAgentSessionProfile(
    name=NAME,
    display_name="Claude Agent Session (Experimental)",
    description="Claude Pro/Max via one long-lived Claude Code process per session (Hermes owns tools)",
    api_mode="chat_completions",
    auth_type="external_process",
    supports_health_check=False,
    model_capabilities=getattr(CATALOG, "MODEL_CAPABILITIES", {}),
    env_vars=(),
    base_url=f"process://{NAME}",
    process_command="claude",
    process_args=(),
    process_command_env_vars=("CLAUDE_AGENT_SESSION_COMMAND",),
    default_aux_model="claude-sonnet-5[1m]",
    fallback_models=tuple(getattr(CATALOG, "MODEL_METADATA", {})),
    model_aliases={alias: CATALOG.native_model(alias) for alias in getattr(CATALOG, "ALIASES", {})},
    classify_api_error=classify_api_error,
)
register_provider(profile)
