from app.local_model import answer_locally


def test_open_ended_local_abstention_escalates() -> None:
    answer = answer_locally("Explain how a database index improves a query plan for a large table.")
    assert answer.confidence == 0.0
