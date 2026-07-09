import httpx

from app.fireworks_client import FireworksClient, FireworksConfig
from app.models import Tier


def test_client_uses_conversation_affinity_and_reports_cache_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert request.headers["x-session-affinity"] == "chat-42"
        assert '"prompt_cache_key":"chat-42"' in body
        assert '"reasoning_effort":"low"' in body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"total_tokens": 37, "prompt_tokens": 31},
                "perf_metrics": {"cached-prompt-tokens": 20},
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = FireworksClient(FireworksConfig(api_key="test-key"), http)
    answer = client.complete([{"role": "user", "content": "hello"}], Tier.SMALL, "chat-42")

    assert answer.token_count == 37
    assert answer.metadata["prompt_tokens"] == 31
    assert answer.metadata["prompt_cache_hit_tokens"] == 20


def test_cache_scope_changes_with_generation_settings() -> None:
    default = FireworksClient(FireworksConfig(api_key="test-key"))
    shorter = FireworksClient(FireworksConfig(api_key="test-key", max_completion_tokens=200))
    assert default.cache_scope() != shorter.cache_scope()
