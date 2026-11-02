"""Catalog behavior: accepted aliases, known routes/windows, and unknown-id pass-through."""
import pytest

import catalog


def test_aliases_resolve_to_pinned_routes():
    assert catalog.native_model("sonnet") == "claude-sonnet-5-5[1m]"
    assert catalog.native_model("opus") == "claude-opus-5-5[1m]"
    assert catalog.native_model("haiku") == "claude-haiku-5-5[1m]"
    assert catalog.native_model("fable") == "claude-fable-5-1[1m]"
    assert catalog.native_model("claude-haiku-4-5") == "claude-haiku-4-5-20251001"


def test_short_and_dotted_names_land_on_the_pinned_route():
    for name in ("sonnet-5-5", "sonnet-5.5", "claude-sonnet-5-5", "claude-sonnet-5.5"):
        assert catalog.native_model(name) == "claude-sonnet-5-5[1m]"
    for name in ("haiku-4-5", "claude-haiku-4-5", "haiku-4.5", "claude-haiku-4.5"):
        assert catalog.native_model(name) == "claude-haiku-4-5-20251001"


def test_unknown_ids_pass_through_untouched_without_metadata():
    assert catalog.native_model("claude-future-9") == "claude-future-9"
    assert catalog.native_model("claude-future-9[1m]") == "claude-future-9[1m]"
    assert catalog.native_model("sonnet-9") == "sonnet-9"
    assert "claude-future-9" not in catalog.MODEL_METADATA
    assert "claude-future-9[1m]" not in catalog.MODEL_METADATA
    assert "claude-future-9" not in catalog.MODEL_CAPABILITIES


def test_unknown_thinking_rules_are_conservative():
    assert catalog.accepts_thinking_disable("claude-future-9") is True
    assert catalog.supports_adaptive_thinking("claude-future-9") is True
    assert catalog.accepts_thinking_disable(None) is True
    assert catalog.supports_adaptive_thinking(None) is True


def test_haiku_45_rejects_a_one_megabyte_selection():
    with pytest.raises(ValueError):
        catalog.native_model("claude-haiku-4-5-20251001[1m]")
    with pytest.raises(ValueError):
        catalog.native_model("haiku-4-5[1m]")


def test_thinking_capability_table():
    for model in ("fable", "opus", "sonnet", "haiku"):
        assert catalog.accepts_thinking_disable(model) is False
    assert catalog.accepts_thinking_disable("opus-5") is True
    assert catalog.accepts_thinking_disable("haiku-4-5") is True
    assert catalog.supports_adaptive_thinking("haiku-4-5") is False
    assert catalog.supports_adaptive_thinking("haiku") is True
    assert catalog.supports_adaptive_thinking("opus-5") is True


def test_metadata_is_derived_from_the_single_window_table():
    assert set(catalog.MODEL_METADATA) == {
        catalog.native_model(model) for model in catalog.CONTEXT_WINDOWS
    }
    for model, window in catalog.CONTEXT_WINDOWS.items():
        route = catalog.native_model(model)
        assert catalog.MODEL_METADATA[route] == {"canonical_model": model, "context_window": window}
    assert catalog.MODEL_METADATA["claude-sonnet-5-5[1m]"]["context_window"] == 1_000_000
    assert catalog.MODEL_METADATA["claude-haiku-4-5-20251001"]["context_window"] == 200_000


def test_capabilities_cover_alias_session_ids():
    assert catalog.MODEL_CAPABILITIES["claude-sonnet-5-5[1m]"]["context_window"] == 1_000_000
    assert catalog.MODEL_CAPABILITIES["claude-sonnet-5-5"]["context_window"] == 1_000_000
    assert catalog.MODEL_CAPABILITIES["sonnet"]["context_window"] == 1_000_000
    assert catalog.MODEL_CAPABILITIES["sonnet[1m]"]["context_window"] == 1_000_000
    assert catalog.MODEL_CAPABILITIES["claude-haiku-4-5"]["context_window"] == 200_000
    assert all(entry["supports_vision"] is True for entry in catalog.MODEL_CAPABILITIES.values())


def test_context_fallback_matches_hermes_lookup():
    # The profile hook's rule: pinned window, then None for a [1m] route, then 200K.
    def context_length(model):
        route = catalog.native_model(model)
        pinned = catalog.MODEL_METADATA.get(route, {}).get("context_window")
        return pinned or (None if route.endswith("[1m]") else 200_000)

    assert context_length("sonnet") == 1_000_000
    assert context_length("haiku-4-5") == 200_000
    assert context_length("claude-future-9") == 200_000
    assert context_length("claude-future-9[1m]") is None
