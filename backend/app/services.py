from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger("bollo.groq")

PRISMA_URL = "https://api.vachana.ai/stt/v3"
TIMBRE_URL = "https://api.vachana.ai/api/v1/tts/inference"
MAX_AUDIO_BYTES = 15 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
MAX_CONVERSATIONS = 256
MAX_HISTORY_MESSAGES = 20
CONVERSATION_TTL_SECONDS = 30 * 60


class ProviderError(RuntimeError):
    """A provider failed; messages are safe to return to the app client."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Settings:
    gnani_api_key: str
    groq_api_key: str
    groq_model: str
    timbre_model: str
    timbre_voice: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gnani_api_key=os.getenv("GNANI_API_KEY", "").strip(),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
            timbre_model=os.getenv("GNANI_TTS_MODEL", "timbre-v2.5").strip(),
            timbre_voice=os.getenv("GNANI_TTS_VOICE", "Nalini").strip(),
        )


def _require(value: str, name: str) -> str:
    if not value:
        raise ProviderError(f"{name} is not configured. Add it to backend/.env and restart the backend.")
    return value


def _provider_message(response: httpx.Response, provider: str) -> str:
    # Never echo request headers or provider payloads, which could contain secrets.
    if response.status_code in (401, 403):
        return f"{provider} rejected the configured API key or account access. Check the key and account permissions."
    if response.status_code == 429:
        return f"{provider} rate limit reached. Wait briefly and try again."
    return f"{provider} returned an error (HTTP {response.status_code}). Please try again."


def _provider_status(response: httpx.Response) -> int | None:
    # Preserve rate limits so clients can show a clear wait-and-retry message.
    return 429 if response.status_code == 429 else None


async def transcribe_audio(
    audio: bytes,
    filename: str = "speech.wav",
    content_type: str = "audio/wav",
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    key = _require(settings.gnani_api_key, "GNANI_API_KEY")
    if not audio:
        raise ProviderError("The audio recording is empty. Please record again.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ProviderError("The recording is too large. Keep recordings under 30 seconds and 15 MB.")
    if Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ProviderError("Unsupported audio format. Use WAV, MP3, OGG, FLAC, AAC, or M4A.")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            response = await client.post(
                PRISMA_URL,
                headers={"X-API-Key-ID": key},
                files={"audio_file": (filename, audio, content_type or "audio/wav")},
                data={"model": "prisma-v3.0", "language_code": "en-IN", "format": "transcribe"},
            )
    except httpx.TimeoutException as exc:
        raise ProviderError("Gnani Prisma timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("Could not reach Gnani Prisma. Check your connection and try again.") from exc

    if not response.is_success:
        raise ProviderError(_provider_message(response, "Gnani Prisma"), status_code=_provider_status(response))
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProviderError("Gnani Prisma returned an unreadable response.") from exc
    transcript = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(transcript, str):
        raise ProviderError("Gnani Prisma did not return a transcript.")
    transcript = transcript.strip()
    if not _has_speech_content(transcript):
        return {"transcript": "", "detected_language_code": payload.get("detected_language_code"), "request_id": payload.get("request_id")}
    return {"transcript": transcript, "detected_language_code": payload.get("detected_language_code"), "request_id": payload.get("request_id")}


def _has_speech_content(transcript: str) -> bool:
    cleaned = transcript.strip()
    if not cleaned or not re.search(r"[\w\d]", cleaned, flags=re.UNICODE):
        return False
    return not re.fullmatch(
        r"\[?(?:silence|noise|background noise|music|inaudible|unintelligible|no speech|empty)\]?|<unk>|\[.*\]",
        cleaned,
        flags=re.IGNORECASE,
    )


SYSTEM_INSTRUCTION = """You are Bollo, a warm, patient English-speaking coach for Indian students and young professionals. Continue the conversation naturally and respond to what the learner actually said. Adapt your vocabulary and sentence complexity to their stated English proficiency. Correct only important or repeated grammar mistakes, gently and without interrupting the conversation. Offer a more natural sentence structure or useful vocabulary when it would help. Never shame the learner or over-correct. Keep spoken replies concise (usually 1–3 sentences). Return exactly one JSON object, no markdown, with this shape: {\"reply\": string, \"corrections\": [{\"original\": string, \"suggestion\": string, \"reason\": string}], \"vocabulary\": [{\"word\": string, \"meaning\": string, \"example\": string}]}. Use empty arrays when there is nothing useful to teach. Do not invent a learner mistake; if the sentence is natural, leave corrections empty."""
@dataclass
class _Conversation:
    messages: list[dict[str, str]]
    last_used: float
    lock: asyncio.Lock


class ConversationStore:
    """Bounded process-local history; sessions expire after inactivity."""

    def __init__(
        self,
        *,
        max_conversations: int = MAX_CONVERSATIONS,
        max_messages: int = MAX_HISTORY_MESSAGES,
        ttl_seconds: float = CONVERSATION_TTL_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.max_conversations = max_conversations
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, _Conversation] = OrderedDict()
        self._guard = asyncio.Lock()

    def _prune_expired(self, now: float) -> None:
        expired = [
            key for key, value in self._entries.items()
            if now - value.last_used >= self.ttl_seconds and not value.lock.locked()
        ]
        for key in expired:
            self._entries.pop(key, None)

    @asynccontextmanager
    async def session(self, conversation_id: str | None) -> AsyncIterator[tuple[str, list[dict[str, str]], Any]]:
        now = self._clock()
        created = False
        async with self._guard:
            self._prune_expired(now)
            if conversation_id:
                state = self._entries.get(conversation_id)
                if state is None:
                    raise ProviderError("This conversation has expired or is unavailable. Start a new practice session.", status_code=409)
                session_id = conversation_id
                self._entries.move_to_end(session_id)
            else:
                if len(self._entries) >= self.max_conversations:
                    evict_key = next((key for key, value in self._entries.items() if not value.lock.locked()), None)
                    if evict_key is None:
                        raise ProviderError("Bollo is handling many conversations right now. Please try again shortly.", status_code=503)
                    self._entries.pop(evict_key, None)
                session_id = uuid.uuid4().hex
                state = _Conversation(messages=[], last_used=now, lock=asyncio.Lock())
                self._entries[session_id] = state
                created = True

        await state.lock.acquire()
        async with self._guard:
            if self._entries.get(session_id) is not state:
                state.lock.release()
                raise ProviderError("This conversation has expired or is unavailable. Start a new practice session.", status_code=409)
            state.last_used = self._clock()
            self._entries.move_to_end(session_id)

        committed = False

        def commit(user_message: str, assistant_message: str) -> None:
            nonlocal committed
            state.messages.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ])
            overflow = len(state.messages) - self.max_messages
            if overflow > 0:
                del state.messages[:overflow]
            committed = True

        try:
            yield session_id, list(state.messages), commit
        finally:
            async with self._guard:
                if created and not committed and not state.messages and self._entries.get(session_id) is state:
                    self._entries.pop(session_id, None)
                elif self._entries.get(session_id) is state:
                    state.last_used = self._clock()
                    self._entries.move_to_end(session_id)
                state.lock.release()

    async def clear(self) -> None:
        async with self._guard:
            self._entries.clear()

    async def snapshot(self) -> dict[str, list[dict[str, str]]]:
        async with self._guard:
            return {key: list(value.messages) for key, value in self._entries.items()}


conversation_store = ConversationStore()


def _safe_groq_diagnostic(error: Exception, api_key: str) -> tuple[int | None, str]:
    """Return bounded diagnostics with common credential formats removed."""
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    detail = getattr(error, "message", None)
    if not isinstance(detail, str) or not detail.strip():
        detail = str(error)
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", detail)
    detail = re.sub(r"(?i)(api[_-]?key|authorization)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", detail)
    detail = " ".join(detail.split())[:1000]
    return status if isinstance(status, int) else None, detail or "No provider error message was returned."


def _groq_client_error(error: Exception) -> tuple[str, int]:
    """Map Groq failures to safe messages and appropriate HTTP status codes."""
    detail = str(error).lower()
    response = getattr(error, "response", None)
    status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
    if isinstance(error, TimeoutError) or "timeout" in detail or "timed out" in detail:
        return "Groq did not respond in time. Please try again.", 504
    if status == 429:
        return "Groq is rate limiting requests. Please wait a moment before trying again.", 429
    if status in (401, 403):
        return "Groq rejected the configured API key or account access. Check the key and account permissions.", 502
    if status == 404:
        return "The configured Groq model is unavailable to this API key. Check GROQ_MODEL and model access.", 502
    if isinstance(status, int) and status >= 500:
        return "Groq is temporarily unavailable. Please try again shortly.", 503
    if "apiconnectionerror" in type(error).__name__.lower() or "connecterror" in detail:
        return "Bollo could not connect to Groq. Check the backend's internet access and try again.", 503
    return "Groq could not generate a response. Check your API key, model access, and connection.", 502


def _parse_coach_output(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProviderError("Bollo returned an unreadable coaching response. Please try again.") from exc
    if not isinstance(result, dict) or not isinstance(result.get("reply"), str) or not result["reply"].strip():
        raise ProviderError("Bollo did not return a coaching reply. Please try again.")
    corrections = result.get("corrections", [])
    vocabulary = result.get("vocabulary", [])
    if not isinstance(corrections, list) or not isinstance(vocabulary, list):
        raise ProviderError("Bollo returned an incomplete coaching response. Please try again.")
    return {
        "reply": result["reply"].strip(),
        "corrections": [item for item in corrections if isinstance(item, dict) and all(isinstance(item.get(k), str) for k in ("original", "suggestion", "reason"))],
        "vocabulary": [item for item in vocabulary if isinstance(item, dict) and all(isinstance(item.get(k), str) for k in ("word", "meaning", "example"))],
    }


async def generate_coaching_response(
    message: str,
    conversation_id: str | None = None,
    proficiency: str = "intermediate",
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    key = _require(settings.groq_api_key, "GROQ_API_KEY")
    message = message.strip()
    if not message:
        raise ProviderError("There was no speech to respond to. Please try again.")
    proficiency = proficiency if proficiency in {"beginner", "intermediate", "advanced"} else "intermediate"
    instruction = f"{SYSTEM_INSTRUCTION}\nLearner's self-selected English proficiency: {proficiency}."
    async with conversation_store.session(conversation_id) as (session_id, history, commit):
        messages: list[dict[str, str]] = [{"role": "system", "content": instruction}, *history, {"role": "user", "content": message}]
        client: AsyncGroq | None = None
        try:
            client = AsyncGroq(api_key=key, timeout=60.0, max_retries=0)
            completion = await client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                max_completion_tokens=1200,
                timeout=60.0,
            )
        except Exception as exc:
            status, diagnostic = _safe_groq_diagnostic(exc, key)
            logger.error(
                "Groq Chat Completions request failed (exception=%s, http_status=%s): %s",
                type(exc).__name__,
                status if status is not None else "unavailable",
                diagnostic,
            )
            safe_message, http_status = _groq_client_error(exc)
            raise ProviderError(safe_message, status_code=http_status) from exc
        finally:
            if client is not None:
                await client.close()

        raw = None
        try:
            raw = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            pass
        if not isinstance(raw, str) or not raw.strip():
            raise ProviderError("Groq returned an incomplete coaching response. Please try again.")
        result = _parse_coach_output(raw)
        commit(message, result["reply"])
        result["conversation_id"] = session_id
        return result


async def synthesize_speech(text: str, *, settings: Settings | None = None) -> tuple[bytes, str]:
    settings = settings or Settings.from_env()
    key = _require(settings.gnani_api_key, "GNANI_API_KEY")
    if not text.strip():
        raise ProviderError("There is no response text to speak.")
    body = {
        "text": text,
        "model": settings.timbre_model,
        "voice": settings.timbre_voice,
        "language": "en-IN",
        "speed": 1.0,
        "audio_config": {
            "encoding": "linear_pcm", "container": "wav", "num_channels": 1,
            "sample_rate": 24000, "sample_width": 2,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            response = await client.post(TIMBRE_URL, headers={"X-API-Key-ID": key}, json=body)
    except httpx.TimeoutException as exc:
        raise ProviderError("Gnani Timbre timed out. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("Could not reach Gnani Timbre. Check your connection and try again.") from exc
    if not response.is_success:
        raise ProviderError(_provider_message(response, "Gnani Timbre"), status_code=_provider_status(response))
    mime_type = response.headers.get("content-type", "audio/wav").split(";", 1)[0]
    if not response.content or not mime_type.startswith("audio/"):
        raise ProviderError("Gnani Timbre did not return playable audio.")
    return response.content, mime_type


async def complete_turn(
    *,
    text: str | None,
    audio: bytes | None,
    filename: str,
    content_type: str,
    conversation_id: str | None,
    proficiency: str,
) -> dict[str, Any]:
    transcript = text.strip() if text else ""
    stt_metadata = None
    if audio:
        transcription = await transcribe_audio(audio, filename, content_type)
        transcript = transcription["transcript"]
        stt_metadata = {key: transcription.get(key) for key in ("detected_language_code", "request_id")}
    if not _has_speech_content(transcript):
        raise ProviderError("I couldn’t hear any speech. Please try recording again or type your message.", status_code=422)
    if len(transcript) > 5000:
        raise ProviderError("That message is too long. Keep each turn under 5,000 characters.")
    if conversation_id and len(conversation_id) > 256:
        raise ProviderError("This conversation reference is invalid. Start a new practice session.")
    coach = await generate_coaching_response(transcript, conversation_id, proficiency)
    audio_bytes = b""
    mime_type = "audio/wav"
    audio_error = None
    try:
        audio_bytes, mime_type = await synthesize_speech(coach["reply"])
    except ProviderError as error:
        # Keep the transcript and coaching available if voice synthesis is down.
        audio_error = str(error)
    return {
        "transcript": transcript,
        "reply": coach["reply"],
        "corrections": coach["corrections"],
        "vocabulary": coach["vocabulary"],
        "conversation_id": coach["conversation_id"],
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_mime_type": mime_type,
        "audio_error": audio_error,
        "speech_to_text": stt_metadata,
    }
