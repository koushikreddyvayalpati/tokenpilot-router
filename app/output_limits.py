from __future__ import annotations

import re


def requested_output_limit(prompt: str) -> int | None:
    """Cap only outputs whose user prompt already imposes a short format."""
    lowered = prompt.lower()
    word_limit = re.search(r"\bunder\s+(\d+)\s+words?\b", lowered)
    if word_limit:
        return max(64, min(120, int(word_limit.group(1)) * 2 + 24))
    bullet_limit = re.search(r"\bexactly\s+(\d+)\s+bullet", lowered)
    if bullet_limit:
        return max(80, min(140, int(bullet_limit.group(1)) * 40))
    if "in one sentence" in lowered:
        return 96
    return None
