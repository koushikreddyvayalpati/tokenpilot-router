from app.local_model import answer_locally


def test_exact_arithmetic_stays_local() -> None:
    answer = answer_locally("What is 17 * 23?")
    assert answer.answer == "391"
    assert answer.confidence == 0.99


def test_open_ended_request_abstains() -> None:
    answer = answer_locally("Explain how a database index improves a query plan for a large table.")
    assert answer.confidence < 0.72
