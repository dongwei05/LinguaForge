"""Top-level orchestration: ties Listen + Learn + Revive into a single agent.

This is what the writeup screenshot will show: one entrypoint, one mental
model — the LinguaForge Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger

from .config import CONFIG, LanguageConfig, SUPPORTED_LANGUAGES
from .learn import TutorSession, new_session
from .listen import LearningCard, run_pipeline as run_listen
from .rag import CardStore


Pillar = Literal["listen", "learn", "revive"]


@dataclass
class LinguaForgeAgent:
    """Convenience facade. Useful for headless scripting & the Kaggle notebook."""

    language_code: str = CONFIG.default_language_code
    store_dir: str = "./artifacts/chroma"

    def __post_init__(self) -> None:
        if self.language_code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language_code}")
        self.language: LanguageConfig = SUPPORTED_LANGUAGES[self.language_code]
        self.store: CardStore = CardStore(persist_dir=self.store_dir)

    def listen(
        self,
        audio_path: str | Path,
        *,
        index: bool = True,
        max_segments: int | None = None,
    ) -> list[LearningCard]:
        cards = run_listen(
            audio_path,
            language_code=self.language_code,
            max_segments=max_segments,
        )
        if index:
            self.store.add(cards)
        return cards

    def learn(self, learner_id: str = "demo_learner") -> TutorSession:
        return new_session(
            learner_id=learner_id,
            language_code=self.language_code,
            store=self.store,
        )

    def stats(self) -> dict:
        return {
            "language": self.language.english_name,
            "native_name": self.language.native_name,
            "speakers": self.language.speakers,
            "status": self.language.status,
            "indexed_cards": self.store.count(),
        }


def banner() -> str:
    return r"""
    __    _                       ____                       
   / /   (_)___  ____ ___  ______/ __/___  _________ ____   
  / /   / / __ \/ __ `/ / / / __ \  / __ \/ ___/ __ `/ _ \  
 / /___/ / / / / /_/ / /_/ / / / / / /_/ / /  / /_/ /  __/  
/_____/_/_/ /_/\__, /\__,_/_/ /_/_/\____/_/   \__, /\___/   
              /____/                         /____/         
        Offline AI for Endangered Languages · Gemma 4
"""


def info() -> None:
    logger.info(banner())
    for code, lang in SUPPORTED_LANGUAGES.items():
        logger.info(
            "  [{}] {} ({}): ~{:,} speakers, status={}",
            code, lang.english_name, lang.native_name,
            lang.speakers, lang.status,
        )


if __name__ == "__main__":
    info()
