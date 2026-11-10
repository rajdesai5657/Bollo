from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import main, services
from app.main import app


def test_health_reports_groq_and_gnani_configuration_without_secrets(monkeypatch):
    monkeypatch.setenv("GNANI_API_KEY", "test-gnani-secret")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-secret")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "providers": {"gnani": "configured", "groq": "configured"}}
    assert "test-gnani-secret" not in response.text
    assert "test-groq-secret" not in response.text


def test_health_reports_missing_groq_key(monkeypatch):
    monkeypatch.setenv("GNANI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    response = TestClient(app).get("/health")
    assert response.json()["providers"] == {"gnani": "missing_key", "groq": "missing_key"}


def test_missing_groq_key_returns_actionable_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    response = TestClient(app).post("/api/coach/respond", json={"text": "Hello there"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_respond_route_preserves_frontend_response_contract(monkeypatch):
    async def response(text, conversation_id=None, proficiency="intermediate"):
        return {
            "reply": "That sounds like a lovely goal.",
            "corrections": [{"original": "I wants", "suggestion": "I want", "reason": "Use the base form."}],
            "vocabulary": [{"word": "goal", "meaning": "something you aim for", "example": "My goal is to speak clearly."}],
            "conversation_id": "local-session-id",
        }

    monkeypatch.setattr(main, "generate_coaching_response", response)
    result = TestClient(app).post("/api/coach/respond", json={"text": "I wants to improve"}).json()
    assert set(result) == {"reply", "corrections", "vocabulary", "conversation_id"}
    assert result["conversation_id"] == "local-session-id"
    assert result["corrections"][0]["suggestion"] == "I want"
    assert result["vocabulary"][0]["word"] == "goal"


def test_turn_route_preserves_audio_and_coaching_response_fields(monkeypatch):
    async def complete(**kwargs):
        return {
            "transcript": "Hello there",
            "reply": "Hi!",
            "corrections": [],
            "vocabulary": [],
            "conversation_id": "local-session-id",
            "audio_base64": "UklGRg==",
            "audio_mime_type": "audio/wav",
            "audio_error": None,
            "speech_to_text": None,
        }

    monkeypatch.setattr(main, "complete_turn", complete)
    response = TestClient(app).post("/api/coach/turn", data={"text": "Hello there"})
    assert response.status_code == 200
    assert set(response.json()) == {
        "transcript", "reply", "corrections", "vocabulary", "conversation_id",
        "audio_base64", "audio_mime_type", "audio_error", "speech_to_text",
    }
    assert response.json()["conversation_id"] == "local-session-id"


def test_empty_prisma_transcript_is_rejected_without_calling_groq(monkeypatch):
    async def no_speech(*args, **kwargs):
        return {"transcript": "[silence]"}

    async def should_not_call_groq(*args, **kwargs):
        raise AssertionError("Groq must not receive an empty or noise transcript")

    monkeypatch.setattr(services, "transcribe_audio", no_speech)
    monkeypatch.setattr(services, "generate_coaching_response", should_not_call_groq)
    response = TestClient(app).post(
        "/api/coach/turn",
        files={"audio_file": ("speech.wav", b"recorded audio", "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "I couldn’t hear any speech. Please try recording again or type your message."}


def test_groq_rate_limit_returns_http_429_with_retry_guidance(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class RateLimitError(Exception):
        status_code = 429
        message = "rate limit reached"

    error = RateLimitError()

    class Completions:
        async def create(self, **kwargs):
            raise error

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

        async def close(self):
            pass

    monkeypatch.setattr(services, "AsyncGroq", FakeGroq)
    response = TestClient(app).post("/api/coach/turn", data={"text": "Hello"})
    assert response.status_code == 429
    assert response.json() == {"detail": "Groq is rate limiting requests. Please wait a moment before trying again."}
