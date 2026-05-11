"""Central configuration for LinguaForge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """LLM-related settings.

    We default to the smallest Gemma 4 variant for on-device deployment, but
    fall back to Gemma 3 when Gemma 4 weights are not yet published in the
    user's environment. This is a hackathon-pragmatic choice; the migration
    path to Gemma 4 is one config flip.
    """

    primary_model_id: str = "google/gemma-4-E4B-it"
    fallback_model_id: str = "google/gemma-4-E2B-it"
    cloud_demo_model_id: str = "google/gemma-4-26B-A4B-it"
    revision: str = "main"

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

    use_4bit: bool = True
    device: str = "auto"  # "cuda", "mps", "cpu", or "auto"


@dataclass
class LanguageConfig:
    """A target endangered language we support."""

    code: str
    english_name: str
    native_name: str
    speakers: int
    status: str  # "endangered", "vulnerable", "definitely_endangered", etc.
    description: str = ""


SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "hak": LanguageConfig(
        code="hak",
        english_name="Hakka",
        native_name="客家话",
        speakers=44_000_000,
        status="vulnerable",
        description=(
            "A major Sinitic language with rich oral literature. Vulnerable "
            "because intergenerational transmission has collapsed in cities."
        ),
    ),
    "chr": LanguageConfig(
        code="chr",
        english_name="Cherokee",
        native_name="ᏣᎳᎩ (Tsalagi)",
        speakers=2_000,
        status="critically_endangered",
        description=(
            "An indigenous North American language with a unique syllabary "
            "invented by Sequoyah in the 1820s. Critically endangered."
        ),
    ),
    "cy": LanguageConfig(
        code="cy",
        english_name="Welsh",
        native_name="Cymraeg",
        speakers=900_000,
        status="vulnerable",
        description=(
            "A Celtic language of Wales. Strong revitalization effort but "
            "still classified as vulnerable.",
        )[0] if isinstance("", str) else "",
    ),
    "nax": LanguageConfig(
        code="nax",
        english_name="Naxi (Dongba)",
        native_name="纳西语",
        speakers=300_000,
        status="endangered",
        description=(
            "Spoken in Yunnan, China. Famous for the Dongba pictographic "
            "script — one of the world's last living pictographic writing systems."
        ),
    ),
}


@dataclass
class AppConfig:
    """Top-level app configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    default_language_code: str = "chr"
    data_dir: Path = DATA_DIR
    artifacts_dir: Path = ARTIFACTS_DIR

    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    kaggle_username: str | None = field(default_factory=lambda: os.getenv("KAGGLE_USERNAME"))

    @property
    def language(self) -> LanguageConfig:
        return SUPPORTED_LANGUAGES[self.default_language_code]


CONFIG = AppConfig()
