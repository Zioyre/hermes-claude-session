"""Claude Agent Session (Experimental): one long-lived Claude Code process per Hermes session.

Registration only; nothing starts until the first model call (the multiplexed gateway imports
provider plugins for every profile).
"""
from providers import register_provider
from providers.base import ProviderProfile

try:
    from . import account
    from . import catalog as CATALOG
    from .errors import classify_api_error
except ImportError:
    import account
    import catalog as CATALOG
    from errors import classify_api_error

NAME = "claude-agent-session-experimental"


class ClaudeAgentSessionProfile(ProviderProfile):
    model_metadata = CATALOG.MODEL_METADATA

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

    def fetch_models(self, **kwargs):
        rows = account.discover_models(**kwargs)
        return [row["id"] for row in rows] if rows else list(CATALOG.MODEL_METADATA)

    def discover_models(self, **kwargs):
        return account.discover_models(**kwargs)

    def setup_status(self, **kwargs):
        return account.setup_status(**kwargs)

    def fetch_account_usage(self, *, base_url=None, api_key=None):
        return account.fetch_account_usage(self.name)


profile = ClaudeAgentSessionProfile(
    name=NAME,
    display_name="Claude Agent Session (Experimental)",
    description="Claude Pro/Max via one long-lived Claude Code process per session (Hermes owns tools)",
    api_mode="chat_completions",
    auth_type="external_process",
    supports_health_check=False,
    model_capabilities=CATALOG.MODEL_CAPABILITIES,
    env_vars=(),
    base_url=f"process://{NAME}",
    process_command="claude",
    process_args=(),
    process_command_env_vars=("CLAUDE_AGENT_SESSION_COMMAND",),
    default_aux_model="claude-sonnet-5[1m]",
    fallback_models=tuple(CATALOG.MODEL_METADATA),
    model_aliases={alias: CATALOG.native_model(alias) for alias in CATALOG.ALIASES},
    classify_api_error=classify_api_error,
)
register_provider(profile)
