"""Make two real Groq Chat Completions calls and verify Bollo history chaining."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import Settings, generate_coaching_response


async def run() -> None:
    settings = Settings.from_env()
    if not settings.groq_api_key:
        raise SystemExit("Missing GROQ_API_KEY. Set it locally in backend/.env.")

    print(f"Sending a real first-turn request to Groq model {settings.groq_model}…")
    first = await generate_coaching_response(
        "I want to practise introducing myself in a job interview.",
        proficiency="intermediate",
        settings=settings,
    )
    assert first["reply"] and first["conversation_id"]
    print(f"PASS — received a parsed reply ({len(first['reply'])} characters) and a conversation ID.")

    print("Sending a second real request using the first turn's conversation ID…")
    second = await generate_coaching_response(
        "Can you make that introduction shorter?",
        conversation_id=first["conversation_id"],
        proficiency="intermediate",
        settings=settings,
    )
    assert second["reply"] and second["conversation_id"] == first["conversation_id"]
    print(f"PASS — received a chained reply ({len(second['reply'])} characters); the server reused the same bounded history session.")


if __name__ == "__main__":
    asyncio.run(run())
