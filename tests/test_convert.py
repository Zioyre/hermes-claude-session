import pytest

from convert import UsageTracker, user_content_to_blocks


def test_usage_keeps_max_per_message_id():
    tracker = UsageTracker()
    tracker.update(
        "m1",
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "output_tokens": 20,
        },
    )
    tracker.update(
        "m1",
        {
            "input_tokens": 90,
            "cache_read_input_tokens": 12,
            "cache_creation_input_tokens": 4,
            "output_tokens": 25,
        },
    )
    tracker.update("m2", {"input_tokens": 200, "cache_read_input_tokens": 20, "output_tokens": 30})
    usage = tracker.openai_usage()
    assert usage == {
        "prompt_tokens": 220,
        "completion_tokens": 55,
        "total_tokens": 275,
        "prompt_tokens_details": {"cached_tokens": 20},
        "cache_creation_input_tokens": 0,
        "native_usage": {
            "input_tokens": 200,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 0,
            "output_tokens": 30,
        },
    }


def test_usage_ignores_missing_counters():
    tracker = UsageTracker()
    tracker.update("m1", {"input_tokens": 7})
    tracker.update("m1", {"output_tokens": 3})
    usage = tracker.openai_usage()
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
    assert usage["native_usage"]["input_tokens"] == 7
    assert usage["native_usage"]["output_tokens"] == 3


def test_usage_without_updates_is_all_zero():
    usage = UsageTracker().openai_usage()
    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
        "cache_creation_input_tokens": 0,
        "native_usage": {},
    }


def test_user_content_from_string():
    assert user_content_to_blocks("hello") == [{"type": "text", "text": "hello"}]


def test_user_content_from_text_and_image_parts():
    content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZZZZ"}},
    ]
    assert user_content_to_blocks(content) == [
        {"type": "text", "text": "look"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "ZZZZ"},
        },
    ]


def test_user_content_rejects_remote_image_urls():
    with pytest.raises(ValueError, match="Only base64 data image_url inputs are supported"):
        user_content_to_blocks([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}])


def test_user_content_skips_unknown_parts():
    assert user_content_to_blocks([{"type": "file", "file": {"id": "1"}}]) == []
