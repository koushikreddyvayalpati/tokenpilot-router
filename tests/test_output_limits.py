from app.output_limits import requested_output_limit


def test_uses_a_cap_only_for_explicit_short_formats() -> None:
    assert requested_output_limit("Explain DNS in under 40 words.") == 104
    assert requested_output_limit("Summarize this in exactly 2 bullet points.") == 80
    assert requested_output_limit("Summarize this in one sentence.") == 96
    assert requested_output_limit("Write a Python function that merges intervals.") is None
