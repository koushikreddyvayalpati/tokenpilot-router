from __future__ import annotations

import re


def requested_output_limit(prompt: str) -> int | None:
    """Use concise initial budgets, with the client retrying only truncated answers."""
    lowered = prompt.lower()
    sentence_limit = re.search(r"\bexactly\s+(\d+)\s+sentences?\b.*?\bunder\s+(\d+)\s+words?", lowered)
    if sentence_limit:
        count, words = (int(value) for value in sentence_limit.groups())
        return max(56, min(120, count * (words + 8) + 8))
    bullet_with_words = re.search(r"\bexactly\s+(\d+)\s+bullet.*?\bunder\s+(\d+)\s+words?", lowered)
    if bullet_with_words:
        count, words = (int(value) for value in bullet_with_words.groups())
        return max(48, min(112, count * (words + 7) + 8))
    word_limit = re.search(r"\bunder\s+(\d+)\s+words?\b", lowered)
    if word_limit:
        return max(72, min(160, int(word_limit.group(1)) + 64))
    bullet_limit = re.search(r"\bexactly\s+(\d+)\s+bullet", lowered)
    if bullet_limit:
        return max(48, min(112, int(bullet_limit.group(1)) * 32 + 8))
    if "in one sentence" in lowered:
        return 56
    if "one or two sentences" in lowered:
        return 72
    if "named entities" in lowered:
        return 96
    if "sentiment" in lowered:
        return 64
    if "summarize" in lowered:
        return 96
    if re.search(r"\b(find and explain the bug|what do(?:es)? .* evaluate|what happens when you call)\b", lowered, re.DOTALL):
        return 192
    if re.search(r"\b(write|implement)\b.*\b(function|algorithm|code)\b", lowered, re.DOTALL):
        return 240
    if re.search(r"\b(logic|puzzle|each (?:person|friend)|who owns|boxes?|houses?|runners?|coworkers?)\b", lowered):
        return 112
    if re.search(r"\b(what is|explain|define|when did|where is|who is)\b", lowered):
        return 88
    return 160
