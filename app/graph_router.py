from __future__ import annotations

from pathlib import Path
import json
from typing import NotRequired, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, StateGraph
from langgraph.types import CachePolicy

from app.cache import JsonAnswerCache
from app.conversation import ChatMessage, ConversationMemory
from app.classifier import classify_task
from app.confidence import is_good_enough
from app.fireworks_client import FireworksClient
from app.local_model import answer_locally
from app.models import ModelAnswer, RouteDecision, Tier
from app.output_limits import requested_output_limit


class RouterState(TypedDict):
    task_id: str
    prompt: str
    conversation_id: str
    messages: list[ChatMessage]
    cache_material: str
    route: NotRequired[RouteDecision]
    tier: NotRequired[Tier]
    answer: NotRequired[ModelAnswer]
    attempts: NotRequired[list[str]]
    token_count: NotRequired[int]


TIER_ORDER = [Tier.LOCAL, Tier.SMALL, Tier.LARGE]


def _next_tier(tier: Tier) -> Tier | None:
    try:
        return TIER_ORDER[TIER_ORDER.index(tier) + 1]
    except IndexError:
        return None


def _cache_key(state: RouterState) -> str:
    return f"{state['cache_material']}::{state.get('tier', Tier.LOCAL)}"


class LangGraphRouter:
    def __init__(
        self,
        fireworks: FireworksClient | None = None,
        answer_cache: JsonAnswerCache | None = None,
        conversations: ConversationMemory | None = None,
        confidence_threshold: float = 0.72,
    ) -> None:
        self.fireworks = fireworks or FireworksClient()
        self.answer_cache = answer_cache or JsonAnswerCache(Path(".router-cache/answers.json"))
        self.conversations = conversations or ConversationMemory()
        self.confidence_threshold = confidence_threshold
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RouterState)
        graph.add_node("classify", self._classify)
        graph.add_node("answer", self._answer, cache_policy=CachePolicy(key_func=_cache_key, ttl=3600))
        graph.add_node("judge", self._judge)
        graph.set_entry_point("classify")
        graph.add_edge("classify", "answer")
        graph.add_edge("answer", "judge")
        graph.add_conditional_edges("judge", self._should_continue, {"retry": "answer", "done": END})
        return graph.compile(cache=InMemoryCache())

    def _classify(self, state: RouterState) -> RouterState:
        route = classify_task(state["prompt"])
        return {"route": route, "tier": route.tier, "attempts": [], "token_count": 0}

    def _answer(self, state: RouterState) -> RouterState:
        tier = state.get("tier", Tier.LOCAL)
        cache_key = state["cache_material"]
        cached = self.answer_cache.get(cache_key, tier)
        if cached:
            answer = ModelAnswer.model_validate({**cached, "cache_hit": True})
        elif tier == Tier.LOCAL:
            answer = answer_locally(state["prompt"])
            self.answer_cache.set(cache_key, tier, answer.model_dump(mode="json"))
        else:
            limit = requested_output_limit(state["prompt"])
            # Kimi is reserved for logic and other hard work. A single adequate
            # completion is cheaper than a short attempt followed by a retry.
            if tier == Tier.LARGE:
                limit = max(limit or 0, 320)
            if limit is None:
                answer = self.fireworks.complete(state["messages"], tier, state["conversation_id"])
            else:
                answer = self.fireworks.complete(state["messages"], tier, state["conversation_id"], max_tokens=limit)
            self.answer_cache.set(cache_key, tier, answer.model_dump(mode="json"))
        return {
            "answer": answer,
            "attempts": [*state.get("attempts", []), tier.value],
            "token_count": state.get("token_count", 0) + answer.token_count,
        }

    def _judge(self, state: RouterState) -> RouterState:
        answer = state["answer"]
        if is_good_enough(answer, self.confidence_threshold):
            return state
        next_tier = _next_tier(answer.tier)
        return {"tier": next_tier or answer.tier}

    def _should_continue(self, state: RouterState) -> str:
        answer = state["answer"]
        if is_good_enough(answer, self.confidence_threshold):
            return "done"
        return "retry" if _next_tier(answer.tier) is not None else "done"

    def run(self, prompt: str, task_id: str = "manual", conversation_id: str = "manual") -> RouterState:
        messages = self.conversations.messages_for(conversation_id, prompt)
        cache_scope = getattr(self.fireworks, "cache_scope", lambda: "default")()
        cache_material = json.dumps({"messages": messages, "scope": cache_scope}, sort_keys=True, separators=(",", ":"))
        state = self.graph.invoke(
            {
                "task_id": task_id,
                "prompt": prompt,
                "conversation_id": conversation_id,
                "messages": messages,
                "cache_material": cache_material,
            }
        )
        self.conversations.append_turn(conversation_id, prompt, state["answer"].answer)
        return state
