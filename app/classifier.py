from __future__ import annotations

import re

from app.models import RouteDecision, Tier
from app.local_model import can_answer_locally

ARITHMETIC_PATTERN = re.compile(r"^\s*(?:what\s+is|calculate|compute|solve)?\s*[-+*/().\d\s]+\??\s*$", re.IGNORECASE)
CODE_PATTERN = re.compile(r"\b(write|implement|debug|optimi[sz]e|refactor|python|javascript|sql|function|class|algorithm)\b", re.IGNORECASE)
PROOF_PATTERN = re.compile(r"\b(prove|derive|theorem|invariant|formal|correctness)\b", re.IGNORECASE)
COMPLEX_PATTERN = re.compile(r"\b(multi[- ]?step|compare|trade[- ]?off|architecture|production|security|adversarial|edge cases?|distributed|failure|crash|replay|deduplicat(?:e|ion)|ordering|latency)\b", re.IGNORECASE)
HARD_SYSTEM_PATTERN = re.compile(r"\b(exactly[- ]?once|threat model|vulnerabilit(?:y|ies)|safe implementation|500\s*gb|p99|worker crashes?)\b", re.IGNORECASE)
QUANTITATIVE_REASONING_PATTERN = re.compile(r"\b(probabilit(?:y|ies)|independent events?|at least \w+ (?:are )?healthy|expected value|combinatorics?)\b", re.IGNORECASE)
LOGIC_PATTERN = re.compile(r"\b(logic|deduc(?:e|tive)|constraint|puzzle|each (?:person|friend)|who owns|which (?:one|person)|(?:box(?:es)?|houses?|runners?|coworkers?|colleagues?|students?)\b.*?(?:heavier|lighter|different|each|row|finished))\b", re.IGNORECASE)


def classify_task(task_text: str) -> RouteDecision:
    text = task_text.strip()
    lowered = text.lower()
    reasons: list[str] = []
    if ARITHMETIC_PATTERN.match(text) and len(text) <= 80:
        return RouteDecision(tier=Tier.LOCAL, difficulty="easy", reasons=["deterministic arithmetic"])
    if can_answer_locally(text):
        return RouteDecision(tier=Tier.LOCAL, difficulty="easy", reasons=["structured deterministic math"])

    if PROOF_PATTERN.search(text):
        reasons.append("formal reasoning keyword")
    if CODE_PATTERN.search(text):
        reasons.append("coding task")
    if COMPLEX_PATTERN.search(text):
        reasons.append("complexity keyword")
    if HARD_SYSTEM_PATTERN.search(text):
        reasons.append("high-risk systems keyword")
    if QUANTITATIVE_REASONING_PATTERN.search(text):
        reasons.append("multi-step quantitative reasoning")
    if LOGIC_PATTERN.search(text):
        reasons.append("constraint reasoning")
    if len(text.split()) > 45:
        reasons.append("long prompt")
    if any(mark in lowered for mark in ["do not use", "ignore previous", "system prompt"]):
        reasons.append("instruction-conflict risk")

    hard_signals = sum([
        bool(PROOF_PATTERN.search(text)),
        bool(CODE_PATTERN.search(text) and COMPLEX_PATTERN.search(text)),
        len(text.split()) > 75,
        "adversarial" in lowered,
        bool(HARD_SYSTEM_PATTERN.search(text)),
        bool(QUANTITATIVE_REASONING_PATTERN.search(text)),
        bool(LOGIC_PATTERN.search(text)),
    ])
    if hard_signals:
        return RouteDecision(tier=Tier.LARGE, difficulty="hard", reasons=reasons or ["hard signal"])

    medium_signals = sum([
        bool(CODE_PATTERN.search(text)),
        bool(COMPLEX_PATTERN.search(text)),
        len(text.split()) > 25,
        "explain" in lowered,
    ])
    if medium_signals:
        return RouteDecision(tier=Tier.SMALL, difficulty="medium", reasons=reasons or ["medium signal"])

    return RouteDecision(tier=Tier.LOCAL, difficulty="easy", reasons=["short factual/simple task"])
