import httpx
import json

from app.fireworks_client import FireworksClient, FireworksConfig
from app.models import Tier


def test_client_uses_conversation_affinity_and_reports_cache_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        assert request.headers["x-session-affinity"] == "chat-42"
        assert '"prompt_cache_key":"chat-42"' in body
        assert '"max_tokens":400' in body
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


def test_client_allows_a_tighter_task_output_cap() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}], "usage": {}})

    client = FireworksClient(FireworksConfig(api_key="test-key"), httpx.Client(transport=httpx.MockTransport(handler)))
    client.complete([{"role": "user", "content": "hello"}], Tier.SMALL, max_tokens=96)
    assert '"max_tokens":96' in captured["body"]


def test_client_retries_a_truncated_concise_answer_at_full_budget() -> None:
    token_limits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token_limits.append(json.loads(request.content)["max_tokens"])
        if len(token_limits) == 1:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}], "usage": {"total_tokens": 20}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}], "usage": {"total_tokens": 35}},
        )

    client = FireworksClient(FireworksConfig(api_key="test-key"), httpx.Client(transport=httpx.MockTransport(handler)))
    answer = client.complete([{"role": "user", "content": "hello"}], Tier.SMALL, max_tokens=56)

    assert token_limits == [56, 400]
    assert answer.answer == "complete"
    assert answer.token_count == 55
    assert answer.metadata["retried_after_truncation"] is True
