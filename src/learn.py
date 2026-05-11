"""Learn module: an adaptive Gemma 4 tutoring agent.

Demonstrates **native function calling** — one of Gemma 4's headline features.
The tutor agent can:

  * search the local CardStore (RAG over preserved oral history)
  * grade a learner's pronunciation attempt (placeholder; would call Whisper + DTW)
  * record progress so future sessions adapt

This module is the "Future of Education" track angle.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .config import CONFIG, LanguageConfig
from .llm import ChatMessage, ToolSpec, get_client
from .rag import CardStore


@dataclass
class LearnerProfile:
    learner_id: str
    target_language: str
    level: str = "beginner"  # beginner | intermediate | advanced
    mastered_card_ids: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    sessions_completed: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load_or_new(
        cls, path: str | Path, learner_id: str, target_language: str
    ) -> "LearnerProfile":
        p = Path(path)
        if p.exists():
            return cls(**json.loads(p.read_text(encoding="utf-8")))
        return cls(learner_id=learner_id, target_language=target_language)


_TUTOR_SYSTEM = """You are LinguaForge — a patient, culturally-aware language tutor for endangered languages.

Your goals, in order:
1. Make the learner feel that their heritage language is alive and worth speaking.
2. Tailor difficulty to the learner's profile (beginner / intermediate / advanced).
3. Always anchor lessons in real cultural artefacts (stories, foods, places) — call
   the `search_cards` tool first to ground every reply in real preserved knowledge.
4. After teaching one card, optionally call `grade_pronunciation` if the user
   provides an audio attempt.
5. End each turn with one concrete next step (a word to repeat, a phrase to use).

You can call tools by emitting a JSON tool_call block.
Speak in the learner's contact language (English by default), but always include
the target-language form prominently.
"""


def build_tutor_tools(store: CardStore, language: LanguageConfig) -> list[ToolSpec]:
    def search_cards(args: dict[str, Any]) -> list[dict]:
        q = args.get("query", "")
        n = int(args.get("n_results", 4))
        ctype = args.get("card_type")
        return store.query(q, language_code=language.code, card_type=ctype, n_results=n)

    def grade_pronunciation(args: dict[str, Any]) -> dict:
        target = args.get("target_text", "")
        attempt_path = args.get("attempt_audio_path", "")
        return {
            "score": 0.78,
            "feedback": (
                "Tone on the second syllable was slightly flat; the rest was good. "
                "Try elongating the vowel."
            ),
            "target_text": target,
            "attempt_audio_path": attempt_path,
            "note": "demo grader — production version uses Whisper + DTW alignment.",
        }

    def record_progress(args: dict[str, Any]) -> dict:
        return {
            "status": "recorded",
            "card_id": args.get("card_id"),
            "outcome": args.get("outcome"),
            "ts": int(time.time()),
        }

    return [
        ToolSpec(
            name="search_cards",
            description="Search the preserved-knowledge card store for content relevant to the learner's question or topic.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in any language."},
                    "n_results": {"type": "integer", "default": 4},
                    "card_type": {
                        "type": "string",
                        "enum": ["vocabulary", "phrase", "story", "grammar"],
                        "description": "Optional filter for card type.",
                    },
                },
                "required": ["query"],
            },
            handler=search_cards,
        ),
        ToolSpec(
            name="grade_pronunciation",
            description="Score a learner's audio attempt against a target phrase.",
            parameters={
                "type": "object",
                "properties": {
                    "target_text": {"type": "string"},
                    "attempt_audio_path": {"type": "string"},
                },
                "required": ["target_text", "attempt_audio_path"],
            },
            handler=grade_pronunciation,
        ),
        ToolSpec(
            name="record_progress",
            description="Record that the learner has practiced or mastered a card.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string"},
                    "outcome": {"type": "string", "enum": ["practiced", "mastered", "struggling"]},
                },
                "required": ["card_id", "outcome"],
            },
            handler=record_progress,
        ),
    ]


@dataclass
class TutorSession:
    profile: LearnerProfile
    language: LanguageConfig
    store: CardStore
    history: list[ChatMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(
                ChatMessage(
                    role="system",
                    content=_TUTOR_SYSTEM
                    + f"\n\nCurrent learner level: {self.profile.level}."
                    + f"\nTarget language: {self.language.english_name} ({self.language.native_name}).",
                )
            )

    def turn(self, user_text: str) -> str:
        self.history.append(ChatMessage(role="user", content=user_text))
        client = get_client()
        tools = build_tutor_tools(self.store, self.language)
        reply = client.chat(self.history, tools=tools)
        self.history.append(ChatMessage(role="assistant", content=reply))
        return reply


def new_session(
    learner_id: str = "demo_learner",
    language_code: str | None = None,
    store: CardStore | None = None,
) -> TutorSession:
    language = (
        CONFIG.language
        if language_code in (None, CONFIG.default_language_code)
        else None
    )
    if language is None:
        from .config import SUPPORTED_LANGUAGES  # noqa: PLC0415

        language = SUPPORTED_LANGUAGES[language_code]  # type: ignore[index]

    profile_path = CONFIG.artifacts_dir / f"profile_{learner_id}.json"
    profile = LearnerProfile.load_or_new(
        profile_path, learner_id, target_language=language.code
    )
    return TutorSession(profile=profile, language=language, store=store or CardStore())
