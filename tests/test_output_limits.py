from app.output_limits import requested_output_limit


def test_uses_concise_task_aware_initial_caps() -> None:
    assert requested_output_limit("Explain DNS in under 40 words.") == 80
    assert requested_output_limit("Summarize this in exactly 2 bullet points, each under 15 words.") == 52
    assert requested_output_limit("Summarize this in exactly 3 sentences, each under 12 words.") == 68
    assert requested_output_limit("Summarize this in one sentence.") == 56
    assert requested_output_limit("Write a Python function that merges intervals.") == 240
    assert requested_output_limit("Extract all named entities from this sentence.") == 96
