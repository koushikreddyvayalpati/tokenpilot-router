from app.classifier import classify_task, completion_token_budget
from app.models import Tier


def test_arithmetic_stays_local() -> None:
    assert classify_task("What is 17 * 23?").tier == Tier.LOCAL


def test_proof_goes_large() -> None:
    assert classify_task("Prove that this algorithm is correct.").tier == Tier.LARGE


def test_uppercase_coding_detected() -> None:
    assert classify_task("Write a Python function for topological sort.").tier in {Tier.SMALL, Tier.LARGE}


def test_security_design_goes_large() -> None:
    prompt = "Review this design for security vulnerabilities and propose a safe implementation."
    assert classify_task(prompt).tier == Tier.LARGE


def test_exactly_once_architecture_goes_large() -> None:
    prompt = "Design a distributed system that processes a 500 GB stream exactly once with replay and p99 latency below two seconds."
    assert classify_task(prompt).tier == Tier.LARGE


def test_multi_step_probability_goes_large() -> None:
    prompt = "Three services have independent failure probabilities. Compute the probability at least two are healthy."
    assert classify_task(prompt).tier == Tier.LARGE


def test_accuracy_profile_sends_ambiguous_categories_to_large() -> None:
    assert classify_task('Classify the sentiment: "Great, another outage."').tier == Tier.LARGE
    assert classify_task('Extract all named entities from: "Noor joined Solace Health."').tier == Tier.LARGE
    assert classify_task("Explain inflation in one sentence.").tier == Tier.LARGE


def test_constraint_puzzles_route_large() -> None:
    assert classify_task("Box A is heavier than box B. Box C is lighter than box B. Which is lightest?").tier == Tier.LARGE
    assert classify_task("Five houses are painted different colors and one is immediately left of another.").tier == Tier.LARGE


def test_completion_budget_preserves_space_for_code_and_summaries() -> None:
    assert completion_token_budget("Write a Python function that merges intervals.", Tier.SMALL) == 260
    assert completion_token_budget("Summarize this report in two bullets.", Tier.LARGE) == 110
    assert completion_token_budget("What is inflation?", Tier.SMALL) == 96
