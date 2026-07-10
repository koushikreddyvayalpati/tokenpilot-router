from app.local_model import answer_locally


def test_open_ended_local_abstention_escalates() -> None:
    answer = answer_locally("Explain how a database index improves a query plan for a large table.")
    assert answer.confidence == 0.0


def test_local_word_math_handles_compound_changes() -> None:
    answer = answer_locally(
        "You invest $1000. In year 1 it grows by 5%, in year 2 by 8%, "
        "in year 3 it shrinks by 3%. What is the final amount, rounded to 2 decimal places?"
    )
    assert answer.answer == "1099.98"
    assert answer.confidence == 0.99


def test_local_sentiment_detects_sarcasm() -> None:
    answer = answer_locally('Classify the sentiment: "Great, another quick update that took the app down."')
    assert answer.answer.startswith("Negative sentiment")


def test_local_entity_template_extracts_typed_entities() -> None:
    answer = answer_locally(
        'Extract all named entities from: "Noor met with representatives from Solace Health in Seoul on March 18, 2025."'
    )
    assert "Person: Noor" in answer.answer
    assert "Organization: Solace Health" in answer.answer
    assert "Location: Seoul" in answer.answer


def test_local_code_debug_detects_mutable_default() -> None:
    answer = answer_locally("Find and explain the bug: def add_item(item, bucket=[]): return bucket")
    assert "mutable default" in answer.answer
