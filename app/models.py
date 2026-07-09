from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Tier(StrEnum):
    LOCAL = "local"
    SMALL = "small"
    LARGE = "large"


class RouteDecision(BaseModel):
    tier: Tier
    difficulty: str
    reasons: list[str] = Field(default_factory=list)


class ModelAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    tier: Tier
    token_count: int = 0
    cache_hit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalTask(BaseModel):
    id: str
    prompt: str
    expected: str | None = None
    min_tier: Tier | None = None

