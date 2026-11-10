"""Real-provider smoke tests; requires keys and a supported recording of real speech."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app
from app.services import Settings, complete_turn, generate_coaching_response, synthesize_speech, transcribe_audio


async def run(audio_path: Path) -> None:
    settings = Settings.from_env()
    missing = [name for name, value in (("GNANI_API_KEY", settings.gnani_api_key), ("GROQ_API_KEY", settings.groq_api_key)) if not value]
    if missing:
        raise SystemExit("Missing configuration: " + ", ".join(missing) + ". Copy backend/.env.example to backend/.env and add local keys.")
    if not audio_path.is_file():
        raise SystemExit(f"Speech sample does not exist: {audio_path}")
    extension = audio_path.suffix.lower()
    content_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac", ".m4a": "audio/mp4"}
    content_type = content_types.get(extension)
    if not content_type:
        raise SystemExit("Use WAV, MP3, OGG, FLAC, AAC, or M4A speech audio.")
    audio = audio_path.read_bytes()
    print("1/6 Backend startup and health route…")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["providers"] == {"gnani": "configured", "groq": "configured"}
    print("   PASS — FastAPI starts and reports configured providers.")

    print("2/6 Gnani Prisma transcription (live)…")
    transcript_result = await transcribe_audio(audio, audio_path.name, content_type)
    transcript = transcript_result["transcript"]
    if not transcript:
        raise RuntimeError("Prisma returned no speech. Choose a clear English voice recording under 30 seconds.")
    print(f"   PASS — transcript received ({len(transcript)} characters).")

    print("3/6 Groq response generation (live)…")
    first = await generate_coaching_response(transcript, proficiency="intermediate")
    print(f"   PASS — Groq returned a response ({len(first['reply'])} characters).")

    print("4/6 Gnani Timbre synthesis (live)…")
    audio_bytes, mime_type = await synthesize_speech(first["reply"])
    assert audio_bytes and mime_type.startswith("audio/")
    print(f"   PASS — received {mime_type}, {len(audio_bytes)} bytes.")

    print("5/6 Complete voice conversation (live)…")
    turn_one = await complete_turn(text=None, audio=audio, filename=audio_path.name, content_type=content_type, conversation_id=None, proficiency="intermediate")
    assert turn_one["transcript"] and turn_one["reply"] and turn_one["audio_base64"]
    print("   PASS — recorded speech → Prisma → Groq → Timbre.")

    print("6/6 Three consecutive context-linked turns (live)…")
    turn_two = await complete_turn(text="Could you explain that in a simpler way?", audio=None, filename="speech.wav", content_type="audio/wav", conversation_id=turn_one["conversation_id"], proficiency="intermediate")
    turn_three = await complete_turn(text="Can you give me one more example?", audio=None, filename="speech.wav", content_type="audio/wav", conversation_id=turn_two["conversation_id"], proficiency="intermediate")
    assert turn_two["conversation_id"] == turn_one["conversation_id"]
    assert turn_three["conversation_id"] == turn_two["conversation_id"]
    assert turn_two["reply"] and turn_three["reply"] and turn_three["audio_base64"]
    print("   PASS — three consecutive Groq turns using the same conversation history, with Timbre audio on every turn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Path to a supported real speech recording no longer than 30 seconds")
    asyncio.run(run(parser.parse_args().audio))
