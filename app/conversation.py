from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


SYSTEM_PROMPT = (
    "Return the final answer first and keep the entire response under 180 words. Do not use headings, "
    "an introduction, a restatement, a self-test, optional examples, or closing remarks. For derivations, "
    "show only the essential equations. For code, provide only the minimal working implementation."
)


def _validate_conversation_id(conversation_id: str) -> str:
    normalized = conversation_id.strip()
    if not normalized or len(normalized) > 128:
        raise ValueError("conversation_id must contain 1 to 128 characters")
    return normalized


@dataclass
class ConversationMemory:
    """In-process chat history with a bounded, whole-message context window."""

    max_context_chars: int = 24_000
    max_messages: int = 16
    _conversations: dict[str, list[ChatMessage]] = field(default_factory=lambda: defaultdict(list))
    _lock: RLock = field(default_factory=RLock)

    def messages_for(self, conversation_id: str, user_prompt: str) -> list[ChatMessage]:
        conversation_id = _validate_conversation_id(conversation_id)
        prompt = user_prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")

        with self._lock:
            history = [*self._conversations[conversation_id], {"role": "user", "content": prompt}]
        return self._trim(history)

    def append_turn(self, conversation_id: str, user_prompt: str, assistant_answer: str) -> None:
        conversation_id = _validate_conversation_id(conversation_id)
        with self._lock:
            self._conversations[conversation_id].extend(
                [
                    {"role": "user", "content": user_prompt.strip()},
                    {"role": "assistant", "content": assistant_answer.strip()},
                ]
            )
            self._conversations[conversation_id] = self._trim(self._conversations[conversation_id])

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._conversations.pop(_validate_conversation_id(conversation_id), None)

    def _trim(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        retained: list[ChatMessage] = []
        used = 0
        for message in reversed(messages[-self.max_messages :]):
            size = len(message["content"])
            if retained and used + size > self.max_context_chars:
                break
            retained.append(message)
            used += size
        retained.reverse()
        return retained
