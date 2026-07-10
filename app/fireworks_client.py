from __future__ import annotations

import os
import re
import time
import json
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from app.confidence import score_answer
from app.models import ModelAnswer, Tier
from app.conversation import ChatMessage, SYSTEM_PROMPT

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass(frozen=True)
class FireworksConfig:
    api_key: str | None = None
    base_url: str = FIREWORKS_BASE_URL
    small_model: str = ""
    large_model: str = ""
    max_completion_tokens: int = 400
    prompt_truncate_len: int = 8_000
    max_retries: int = 2
    reasoning_effort: str = "low"

    @classmethod
    def from_environment(cls) -> "FireworksConfig":
        allowed = _parse_allowed_models(os.getenv("ALLOWED_MODELS", ""))
        if not allowed:
            raise RuntimeError("ALLOWED_MODELS must be provided by the judging environment")
        base_url = os.getenv("FIREWORKS_BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("FIREWORKS_BASE_URL must be provided by the judging environment")
        small_model, large_model = _select_model_tiers(allowed)
        return cls(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url=base_url,
            small_model=small_model,
            large_model=large_model,
        )


def _parse_allowed_models(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        import json

        parsed = json.loads(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError("ALLOWED_MODELS JSON must be a list of model IDs")
        return [item.strip() for item in parsed if item.strip()]
    return [item.strip() for item in re.split(r"[,\n\s]+", value) if item.strip()]


def _model_size_hint(model: str) -> int:
    match = re.search(r"(?:^|[-_/])(\d+)(?:\.\d+)?b(?:$|[-_/])", model.lower())
    if match:
        return int(match.group(1))
    if "mini" in model.lower() or "small" in model.lower():
        return 1
    return 10_000


def _select_model_tiers(allowed_models: list[str]) -> tuple[str, str]:
    ordered = sorted(allowed_models, key=lambda model: (_model_size_hint(model), model))
    return ordered[0], ordered[-1]


class FireworksClient:
    def __init__(self, config: FireworksConfig | None = None, http_client: httpx.Client | None = None) -> None:
        self.config = config or FireworksConfig(api_key=os.getenv("FIREWORKS_API_KEY"))
        self.http = http_client or httpx.Client(timeout=30)

    def cache_scope(self) -> str:
        return json.dumps(
            {
                "small_model": self.config.small_model,
                "large_model": self.config.large_model,
                "max_completion_tokens": self.config.max_completion_tokens,
                "reasoning_effort": self.config.reasoning_effort,
                "system_prompt": SYSTEM_PROMPT,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tier: Tier,
        conversation_id: str = "manual",
        max_tokens: int | None = None,
    ) -> ModelAnswer:
        if tier not in {Tier.SMALL, Tier.LARGE}:
            raise ValueError("Fireworks tier must be small or large")
        if not self.config.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for paid Fireworks tiers")

        if not messages or messages[-1]["role"] != "user":
            raise ValueError("messages must end with a user message")

        model = self.config.small_model if tier == Tier.SMALL else self.config.large_model
        token_limit = max_tokens if max_tokens is not None else self.config.max_completion_tokens
        if token_limit < 16 or token_limit > self.config.max_completion_tokens:
            raise ValueError("max_tokens must be between 16 and the configured completion cap")
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            "temperature": 0,
            "max_tokens": token_limit,
            "prompt_cache_key": conversation_id,
            "perf_metrics_in_response": True,
        }
        response = self._post_with_retry(payload, conversation_id)
        payload: dict[str, Any] = response.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Fireworks response did not include a chat completion") from exc
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Fireworks returned an empty completion")
        usage = payload.get("usage", {})
        perf_metrics = payload.get("perf_metrics", {})
        metadata = {
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens") or perf_metrics.get("prompt-tokens"),
            "prompt_cache_hit_tokens": (
                usage.get("prompt_cache_hit_tokens")
                or usage.get("cached_tokens")
                or perf_metrics.get("cached-prompt-tokens")
                or response.headers.get("fireworks-cached-prompt-tokens")
            ),
        }
        return ModelAnswer(
            answer=text,
            confidence=score_answer(text),
            tier=tier,
            token_count=int(usage.get("total_tokens") or 0),
            metadata=metadata,
        )

    def _post_with_retry(self, payload: dict[str, Any], conversation_id: str) -> httpx.Response:
        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.http.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                        "x-session-affinity": conversation_id,
                    },
                    json=payload,
                )
                if response.status_code not in retryable_statuses:
                    response.raise_for_status()
                    return response
                last_error = httpx.HTTPStatusError(
                    f"retryable Fireworks status {response.status_code}", request=response.request, response=response
                )
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                time.sleep(min(0.25 * (2**attempt), 1.0))
        assert last_error is not None
        raise RuntimeError("Fireworks request failed after retries") from last_error
