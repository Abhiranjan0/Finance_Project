from finance_advisor.backend.groq_client import chat_completion_text


def test_chat_gpt():
    reply = chat_completion_text([{"role": "user", "content": "Hello"}])
    assert isinstance(reply, str)
