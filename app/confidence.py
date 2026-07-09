from __future__ import annotations

import re

from app.models import ModelAnswer

HEDGE_PATTERN = re.compile(r"\b(maybe|possibly|I think|not sure|cannot verify|might be|probably)\b", re.IGNORECASE)
REFUSAL_PATTERN = re.compile(r"\b(I can'?t|I cannot|unable to|sorry)\b", re.IGNORECASE)


def score_answer(answer: str) -> float:
    stripped = answer.strip()
    if not stripped:
        return 0.0

    score = 0.76
    if len(stripped) < 4:
        score -= 0.25
    if HEDGE_PATTERN.search(stripped):
        score -= 0.25
    if REFUSAL_PATTERN.search(stripped):
        score -= 0.35
    if "```" in stripped or re.search(r"\btherefore\b|\bbecause\b", stripped, re.IGNORECASE):
        score += 0.06
    if re.search(r"confidence\s*[:=]\s*(9|10|0\.9)", stripped, re.IGNORECASE) and HEDGE_PATTERN.search(stripped):
        score -= 0.2

    return max(0.0, min(1.0, round(score, 3)))


def is_good_enough(answer: ModelAnswer, threshold: float) -> bool:
    return answer.confidence >= threshold and bool(answer.answer.strip())

