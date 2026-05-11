"""Listen module: oral history → multimodal learning cards.

The "Listen" pillar of LinguaForge. Workflow:

    elder_audio.wav  →  ASR transcription  →  Gemma 4 (text + image)
                                                  ↓
                                          LearningCard objects
                                          (vocabulary, grammar, story)

Why this matters: many endangered-language speakers are non-literate elders.
Their knowledge dies with them unless we capture it. This module turns a
single oral story into hundreds of structured learning artefacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger

from .config import CONFIG, LanguageConfig
from .llm import ChatMessage, get_client


@dataclass
class LearningCard:
    """One bite-sized teaching unit derived from oral input."""

    card_id: str
    language_code: str
    card_type: str  # "vocabulary" | "phrase" | "story" | "grammar"
    native_text: str
    english_gloss: str
    pinyin_or_ipa: str = ""
    cultural_note: str = ""
    image_prompt: str = ""  # to be fed to image generator later
    audio_clip_path: str | None = None
    source_audio_path: str | None = None
    source_timestamp_ms: tuple[int, int] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str


def transcribe_audio(audio_path: str | Path, language_code: str = "auto") -> list[TranscriptSegment]:
    """Whisper-based ASR. We use Whisper because Gemma 4 audio is text-out-only;
    we need word-level timestamps to slice cards back to specific audio clips.
    """
    try:
        import whisper  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper not installed. pip install openai-whisper"
        ) from exc

    model = whisper.load_model("large-v3-turbo")
    kwargs = {} if language_code == "auto" else {"language": language_code}
    result = model.transcribe(str(audio_path), word_timestamps=False, **kwargs)
    segments = result.get("segments", [])
    return [
        TranscriptSegment(
            start_ms=int(seg["start"] * 1000),
            end_ms=int(seg["end"] * 1000),
            text=seg["text"].strip(),
        )
        for seg in segments
    ]


_CARD_GENERATION_SYSTEM = """You are a linguistic field-worker AI helping to preserve endangered languages.
Given a transcribed segment in a low-resource language, produce a JSON array of LearningCards
that maximally extract teaching value. For each segment produce 1-5 cards mixing types:
"vocabulary", "phrase", "story", "grammar".

Rules:
- Always include English gloss accessible to a beginner.
- For vocabulary cards: pick semantically rich or culturally specific words.
- For grammar cards: highlight one grammatical pattern visible in the segment.
- For story cards: summarize the cultural meaning in 2-3 sentences.
- The image_prompt field should be a vivid, culturally accurate description suitable
  for an image generator (e.g. "a Cherokee elder weaving a river-cane basket at dusk").
- Output ONLY a JSON array. No prose."""


def cards_from_segments(
    segments: list[TranscriptSegment],
    language: LanguageConfig,
    *,
    source_audio_path: str | None = None,
    max_segments: int | None = None,
) -> list[LearningCard]:
    """Convert transcript segments into LearningCards via Gemma 4."""
    client = get_client()
    cards: list[LearningCard] = []
    seg_iter = segments[:max_segments] if max_segments else segments

    for idx, seg in enumerate(seg_iter):
        user_msg = (
            f"Language: {language.english_name} ({language.native_name}, ISO {language.code})\n"
            f"Segment text: {seg.text}\n"
            f"Segment time: {seg.start_ms}ms-{seg.end_ms}ms"
        )
        try:
            raw = client.chat(
                [
                    ChatMessage(role="system", content=_CARD_GENERATION_SYSTEM),
                    ChatMessage(role="user", content=user_msg),
                ]
            )
            parsed = _safe_json_array(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Card generation failed for segment {}: {}", idx, exc)
            continue

        for j, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            cards.append(
                LearningCard(
                    card_id=f"{language.code}-{idx:04d}-{j:02d}",
                    language_code=language.code,
                    card_type=str(item.get("card_type", "vocabulary")),
                    native_text=str(item.get("native_text", "")),
                    english_gloss=str(item.get("english_gloss", "")),
                    pinyin_or_ipa=str(item.get("pinyin_or_ipa", "")),
                    cultural_note=str(item.get("cultural_note", "")),
                    image_prompt=str(item.get("image_prompt", "")),
                    source_audio_path=source_audio_path,
                    source_timestamp_ms=(seg.start_ms, seg.end_ms),
                    tags=list(item.get("tags", [])),
                )
            )

    logger.info("Generated {} cards from {} segments.", len(cards), len(seg_iter))
    return cards


def _safe_json_array(raw: str) -> list[Any]:
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []


def save_cards(cards: list[LearningCard], path: str | Path) -> None:
    payload = [asdict(c) for c in cards]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("Saved {} cards → {}", len(cards), path)


def load_cards(path: str | Path) -> list[LearningCard]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LearningCard(**c) for c in raw]


def run_pipeline(
    audio_path: str | Path,
    *,
    language_code: str | None = None,
    output_path: str | Path | None = None,
    max_segments: int | None = None,
) -> list[LearningCard]:
    """End-to-end: audio file → saved learning-card JSON."""
    lang_code = language_code or CONFIG.default_language_code
    language = CONFIG.language if lang_code == CONFIG.default_language_code else None
    if language is None:
        from .config import SUPPORTED_LANGUAGES  # noqa: PLC0415

        if lang_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {lang_code}")
        language = SUPPORTED_LANGUAGES[lang_code]

    segments = transcribe_audio(audio_path, language_code=lang_code)
    cards = cards_from_segments(
        segments,
        language,
        source_audio_path=str(audio_path),
        max_segments=max_segments,
    )
    if output_path:
        save_cards(cards, output_path)
    return cards
