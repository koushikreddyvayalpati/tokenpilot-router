from app.local_model import answer_locally
import pytest


def test_exact_arithmetic_stays_local() -> None:
    answer = answer_locally("What is 17 * 23?")
    assert answer.answer == "391"
    assert answer.confidence == 0.99


def test_open_ended_request_abstains() -> None:
    answer = answer_locally("Explain how a database index improves a query plan for a large table.")
    assert answer.confidence < 0.72


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("An item costs $71. It has a 30% discount applied, then 8% sales tax is added to the discounted price. What is the final price, rounded to 2 decimal places?", "53.68"),
        ("A tank starts with 480 liters. It drains at 8 liters per minute for 15 minutes, then is refilled at 12 liters per minute for 20 minutes, then drains again at 5 liters per minute for 10 minutes. How many liters are in the tank now?", "550"),
        ("Pipe A can fill a pool in 6 hours. Pipe B can fill the same pool in 4 hours. If both pipes are opened together, how many hours will it take to fill the pool? Answer as a decimal rounded to 1 decimal place.", "2.4"),
        ("A price is increased by 20%, then decreased by 20%, then increased by 10%. If the original price was $200, what is the final price?", "211.20"),
        ("You invest $1000. In year 1 it grows by 5%, in year 2 by 8%, in year 3 it shrinks by 3%. What is the final amount, rounded to 2 decimal places?", "1099.98"),
    ],
)
def test_structured_word_math_stays_local(prompt: str, expected: str) -> None:
    answer = answer_locally(prompt)
    assert answer.answer == expected
    assert answer.confidence == 0.99


def test_exact_entity_template_stays_local_but_ambiguous_entity_prompt_abstains() -> None:
    answer = answer_locally(
        'Extract all named entities (person, organization, location, date) from this sentence: "Noor met with representatives from Solace Health in Seoul on March 18, 2025 to finalize the merger."'
    )
    assert answer.answer == "Person: Noor; Organization: Solace Health; Location: Seoul; Date: March 18, 2025."
    assert answer.confidence == 0.99

    ambiguous = answer_locally(
        'Extract all named entities from this sentence, labeling each by type: "Amazon announced that Jordan will lead the Phoenix office in April."'
    )
    assert ambiguous.confidence < 0.72


def test_code_debugging_abstains_for_an_allowed_model() -> None:
    answer = answer_locally("Find and explain the bug in this Python function: def divide(a, b): return a / b")
    assert answer.confidence < 0.72
