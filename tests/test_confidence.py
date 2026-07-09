from app.confidence import score_answer


def test_empty_answer_low_confidence() -> None:
    assert score_answer("") == 0.0


def test_hedged_high_confidence_claim_is_penalized() -> None:
    answer = "Confidence: 10/10, but I might be wrong and cannot verify this."
    assert score_answer(answer) < 0.5


def test_reasoned_answer_gets_usable_confidence() -> None:
    assert score_answer("42 because 40 + 2 = 42") >= 0.72

