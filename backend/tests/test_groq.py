import asyncio
import json
from types import SimpleNamespace

import pytest

from app import services
from app.services import ConversationStore, ProviderError, Settings, generate_coaching_response


def configured_settings(key: str = "test-groq-key") -> Settings:
    return Settings("", key, "openai/gpt-oss-120b", "timbre-v2.5", "Nalini")


def valid_response(reply: str = "That sounds like a good start.") -> str:
    return json.dumps({
        "reply": reply,
        "corrections": [{"original": "I go yesterday", "suggestion": "I went yesterday", "reason": "Use past tense."}],
        "vocabulary": [{"word": "confident", "meaning": "sure of yourself", "example": "I felt confident speaking."}],
    })


class FakeCompletions:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        raw = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])


class FakeAsyncGroq:
    def __init__(self, *, api_key, timeout, max_retries, completions):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self):
        self.closed = True


class FakeProviderError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def test_groq_first_and_chained_turns_include_ordered_context(monkeypatch):
    async def run():
        await services.conversation_store.clear()
        completions = FakeCompletions([valid_response(), valid_response("Yes, you could say it that way."), valid_response("For example, you might say...")])
        clients = []

        def make_client(**kwargs):
            client = FakeAsyncGroq(**kwargs, completions=completions)
            clients.append(client)
            return client

        monkeypatch.setattr(services, "AsyncGroq", make_client)
        settings = configured_settings()
        first = await generate_coaching_response("I go yesterday", proficiency="beginner", settings=settings)
        second = await generate_coaching_response("Can you say that more simply?", first["conversation_id"], proficiency="beginner", settings=settings)
        third = await generate_coaching_response("Give me one more example.", second["conversation_id"], proficiency="beginner", settings=settings)

        first_call, second_call, third_call = completions.calls
        assert first["conversation_id"]
        assert second["conversation_id"] == third["conversation_id"] == first["conversation_id"]
        assert first["reply"] == "That sounds like a good start."
        assert first["corrections"][0]["suggestion"] == "I went yesterday"
        assert first["vocabulary"][0]["word"] == "confident"
        assert first_call["model"] == second_call["model"] == third_call["model"] == settings.groq_model
        assert first_call["response_format"] == {"type": "json_object"}
        assert first_call["timeout"] == 60.0
        assert "Learner's self-selected English proficiency: beginner." in first_call["messages"][0]["content"]
        assert first_call["messages"] == [
            {"role": "system", "content": first_call["messages"][0]["content"]},
            {"role": "user", "content": "I go yesterday"},
        ]
        assert [(item["role"], item["content"]) for item in second_call["messages"][1:]] == [
            ("user", "I go yesterday"),
            ("assistant", "That sounds like a good start."),
            ("user", "Can you say that more simply?"),
        ]
        assert [(item["role"], item["content"]) for item in third_call["messages"][1:]] == [
            ("user", "I go yesterday"),
            ("assistant", "That sounds like a good start."),
            ("user", "Can you say that more simply?"),
            ("assistant", "Yes, you could say it that way."),
            ("user", "Give me one more example."),
        ]
        assert len(clients) == 3 and all(client.closed for client in clients)
        await services.conversation_store.clear()

    asyncio.run(run())


def test_json_response_parsing_and_validation():
    parsed = services._parse_coach_output(valid_response())
    assert set(parsed) == {"reply", "corrections", "vocabulary"}
    assert parsed["reply"] == "That sounds like a good start."
    assert parsed["corrections"][0]["original"] == "I go yesterday"
    assert parsed["vocabulary"][0]["word"] == "confident"

    for malformed in (
        "not json",
        '{"reply":"  ","corrections":[],"vocabulary":[]}',
        '{"reply":"Hello","corrections":null,"vocabulary":[]}',
    ):
        with pytest.raises(ProviderError):
            services._parse_coach_output(malformed)


def test_missing_groq_key_is_actionable_and_does_not_call_provider(monkeypatch):
    def should_not_construct(**kwargs):
        raise AssertionError("Groq must not be called without a key")

    monkeypatch.setattr(services, "AsyncGroq", should_not_construct)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        asyncio.run(generate_coaching_response("Hello", settings=configured_settings("")))


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_text"),
    [
        (FakeProviderError(429, "rate limit"), 429, "rate limiting"),
        (FakeProviderError(503, "busy"), 503, "temporarily unavailable"),
        (TimeoutError("request timed out"), 504, "did not respond in time"),
    ],
)
def test_groq_provider_errors_are_mapped_safely(monkeypatch, error, expected_status, expected_text):
    async def run():
        await services.conversation_store.clear()
        completions = FakeCompletions(error=error)
        monkeypatch.setattr(services, "AsyncGroq", lambda **kwargs: FakeAsyncGroq(**kwargs, completions=completions))
        with pytest.raises(ProviderError) as raised:
            await generate_coaching_response("Hello", settings=configured_settings())
        assert raised.value.status_code == expected_status
        assert expected_text in str(raised.value)
        await services.conversation_store.clear()

    asyncio.run(run())


def test_groq_error_logs_status_and_redacts_configured_key(monkeypatch, caplog):
    secret = "groq-secret-that-must-not-appear"
    error = FakeProviderError(400, f"Invalid request; api_key={secret}")

    async def run():
        await services.conversation_store.clear()
        completions = FakeCompletions(error=error)
        monkeypatch.setattr(services, "AsyncGroq", lambda **kwargs: FakeAsyncGroq(**kwargs, completions=completions))
        with caplog.at_level("ERROR", logger="bollo.groq"):
            with pytest.raises(ProviderError):
                await generate_coaching_response("Hello", settings=configured_settings(secret))
        await services.conversation_store.clear()

    asyncio.run(run())
    assert "http_status=400" in caplog.text
    assert "Invalid request" in caplog.text
    assert secret not in caplog.text


def test_malformed_provider_response_does_not_commit_history(monkeypatch):
    async def run():
        await services.conversation_store.clear()
        completions = FakeCompletions(["this is not json"])
        monkeypatch.setattr(services, "AsyncGroq", lambda **kwargs: FakeAsyncGroq(**kwargs, completions=completions))
        with pytest.raises(ProviderError, match="unreadable coaching response"):
            await generate_coaching_response("Hello", settings=configured_settings())
        assert await services.conversation_store.snapshot() == {}

    asyncio.run(run())


def test_history_is_bounded_by_message_and_conversation_limits():
    async def run():
        store = ConversationStore(max_conversations=2, max_messages=4, ttl_seconds=60)
        async with store.session(None) as (first_id, _, commit):
            commit("one", "reply one")
            commit("two", "reply two")
            commit("three", "reply three")
        messages = (await store.snapshot())[first_id]
        assert len(messages) == 4
        assert [(item["role"], item["content"]) for item in messages] == [
            ("user", "two"), ("assistant", "reply two"),
            ("user", "three"), ("assistant", "reply three"),
        ]
        async with store.session(None) as (second_id, _, commit):
            commit("new", "answer")
        async with store.session(None) as (third_id, _, commit):
            commit("another", "answer")
        snapshot = await store.snapshot()
        assert len(snapshot) == 2
        assert first_id not in snapshot
        assert second_id in snapshot and third_id in snapshot

    asyncio.run(run())


def test_expired_and_unknown_conversation_ids_are_rejected():
    now = [100.0]
    store = ConversationStore(max_conversations=2, ttl_seconds=5, clock=lambda: now[0])

    async def run():
        async with store.session(None) as (session_id, _, commit):
            commit("hello", "hi")
        now[0] += 6
        with pytest.raises(ProviderError) as raised:
            async with store.session(session_id):
                pass
        assert raised.value.status_code == 409
        assert "expired" in str(raised.value)
        with pytest.raises(ProviderError) as unknown:
            async with store.session("unknown-id"):
                pass
        assert unknown.value.status_code == 409

    asyncio.run(run())
