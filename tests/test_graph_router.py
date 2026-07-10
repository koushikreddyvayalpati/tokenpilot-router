from pathlib import Path

from app.cache import JsonAnswerCache
from app.fireworks_client import FireworksClient
from app.graph_router import LangGraphRouter
from app.models import ModelAnswer, Tier


class FakeFireworks(FireworksClient):
    def __init__(self) -> None:
        self.calls: list[Tier] = []
        self.requests: list[tuple[list[dict[str, str]], Tier, str]] = []
        self.token_limits: list[int | None] = []

    def cache_scope(self) -> str:
        return "fake-fireworks-v1"

    def complete(
        self,
        messages: list[dict[str, str]],
        tier: Tier,
        conversation_id: str = "manual",
        max_tokens: int | None = None,
    ) -> ModelAnswer:
        self.calls.append(tier)
        self.requests.append((messages, tier, conversation_id))
        self.token_limits.append(max_tokens)
        if tier == Tier.SMALL:
            return ModelAnswer(answer="maybe", confidence=0.2, tier=tier, token_count=8)
        return ModelAnswer(answer="therefore solved", confidence=0.9, tier=tier, token_count=20)


def test_local_arithmetic_zero_fireworks_tokens(tmp_path: Path) -> None:
    fake = FakeFireworks()
    router = LangGraphRouter(fake, JsonAnswerCache(tmp_path / "answers.json"))
    result = router.run("What is 17 * 23?")
    assert result["answer"].answer == "391"
    assert result["answer"].tier == Tier.LOCAL
    assert result["token_count"] == 0
    assert fake.calls == []


def test_escalates_until_confident(tmp_path: Path) -> None:
    fake = FakeFireworks()
    router = LangGraphRouter(fake, JsonAnswerCache(tmp_path / "answers.json"))
    result = router.run("Write and optimize a production parser with adversarial edge cases.")
    assert result["answer"].tier == Tier.LARGE
    assert result["answer"].confidence >= 0.72
    assert result["attempts"] == ["large"]
    assert fake.calls == [Tier.LARGE]
    assert fake.token_limits == [160]


def test_persistent_cache_skips_repeat_paid_call(tmp_path: Path) -> None:
    cache = JsonAnswerCache(tmp_path / "answers.json")
    fake = FakeFireworks()
    router = LangGraphRouter(fake, cache)
    prompt = "Write a Python function to parse and validate nested JSON."
    first = router.run(prompt)
    second = LangGraphRouter(fake, cache).run(prompt)
    assert first["answer"].tier == Tier.LARGE
    assert second["answer"].cache_hit is True
    assert fake.calls == [Tier.LARGE]


def test_follow_up_receives_prior_long_context(tmp_path: Path) -> None:
    fake = FakeFireworks()
    router = LangGraphRouter(fake, JsonAnswerCache(tmp_path / "answers.json"))
    long_prompt = ("Design an event processing platform with durable replay and ordered customer updates. " * 80)
    router.run(long_prompt, conversation_id="chat-1")
    router.run("Explain the recovery plan.", conversation_id="chat-1")

    follow_up_messages, _, session_id = fake.requests[-1]
    assert session_id == "chat-1"
    assert follow_up_messages[0]["content"] == long_prompt.strip()
    assert follow_up_messages[1]["role"] == "assistant"
    assert follow_up_messages[-1]["content"] == "Explain the recovery plan."
