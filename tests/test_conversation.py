from app.conversation import ConversationMemory


def test_history_keeps_whole_recent_messages_within_budget() -> None:
    memory = ConversationMemory(max_context_chars=40, max_messages=8)
    memory.append_turn("chat", "first user message", "first assistant answer")
    messages = memory.messages_for("chat", "latest question")

    assert messages == [
        {"role": "assistant", "content": "first assistant answer"},
        {"role": "user", "content": "latest question"},
    ]


def test_empty_prompt_and_invalid_conversation_id_are_rejected() -> None:
    memory = ConversationMemory()
    try:
        memory.messages_for("", "hello")
    except ValueError as exc:
        assert "conversation_id" in str(exc)
    else:
        raise AssertionError("expected invalid conversation id to fail")

    try:
        memory.messages_for("chat", "   ")
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("expected empty prompt to fail")
