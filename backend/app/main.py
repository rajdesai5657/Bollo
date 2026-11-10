from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services import MAX_AUDIO_BYTES, ProviderError, Settings, complete_turn, generate_coaching_response, synthesize_speech, transcribe_audio

app = FastAPI(title="Bollo Coach API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TextTurn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    conversation_id: str | None = None
    proficiency: str = "intermediate"


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


def _http_error(error: ProviderError) -> HTTPException:
    message = str(error)
    status = error.status_code or (503 if "not configured" in message else 502)
    return HTTPException(status_code=status, detail=message)


@app.get("/health")
async def health() -> dict[str, object]:
    settings = Settings.from_env()
    return {
        "status": "ok",
        "providers": {
            "gnani": "configured" if settings.gnani_api_key else "missing_key",
            "groq": "configured" if settings.groq_api_key else "missing_key",
        },
    }


@app.post("/api/coach/turn")
async def coach_turn(
    audio_file: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
    conversation_id: Annotated[str | None, Form()] = None,
    proficiency: Annotated[str, Form()] = "intermediate",
) -> dict[str, object]:
    try:
        audio = await audio_file.read(MAX_AUDIO_BYTES + 1) if audio_file else None
        return await complete_turn(
            text=text,
            audio=audio,
            filename=audio_file.filename or "speech.wav" if audio_file else "speech.wav",
            content_type=audio_file.content_type or "audio/wav" if audio_file else "audio/wav",
            conversation_id=conversation_id,
            proficiency=proficiency,
        )
    except ProviderError as error:
        raise _http_error(error) from error


@app.post("/api/coach/transcribe")
async def transcribe_endpoint(audio_file: Annotated[UploadFile, File()]) -> dict[str, object]:
    try:
        audio = await audio_file.read(MAX_AUDIO_BYTES + 1)
        return await transcribe_audio(audio, audio_file.filename or "speech.wav", audio_file.content_type or "audio/wav")
    except ProviderError as error:
        raise _http_error(error) from error


@app.post("/api/coach/respond")
async def respond_endpoint(request: TextTurn) -> dict[str, object]:
    try:
        return await generate_coaching_response(request.text, request.conversation_id, request.proficiency)
    except ProviderError as error:
        raise _http_error(error) from error


@app.post("/api/coach/speak")
async def speak_endpoint(request: SpeechRequest) -> dict[str, str]:
    import base64

    try:
        audio, mime_type = await synthesize_speech(request.text)
    except ProviderError as error:
        raise _http_error(error) from error
    return {"audio_base64": base64.b64encode(audio).decode("ascii"), "audio_mime_type": mime_type}
