"""Helper to grab a small public-domain sample for smoke-testing the demo.

Usage:
    python -m src.fetch_sample_data

Pulls one Cherokee-language audio sample from a permissively-licensed source
(e.g. archive.org public-domain Cherokee New Testament recordings, or a
Common Voice clip if available locally). Writes to `data/sample_cherokee_story.wav`.

If no internet is available, generates a synthetic placeholder using a TTS
model so the pipeline still has something to chew on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from loguru import logger


PUBLIC_DOMAIN_URLS = {
    "chr": [
        # Internet Archive: public-domain Cherokee New Testament audio
        "https://archive.org/download/CherokeeNewTestamentAudio/01-Matthew_001.mp3",
    ],
    "cy": [
        "https://archive.org/download/welsh-public-domain-sample/welsh_sample.mp3",
    ],
    "hak": [
        "https://archive.org/download/hakka-folksong-public-domain/hakka_song.mp3",
    ],
}


def fetch(language_code: str = "chr", out_dir: str = "data") -> str | None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if language_code not in PUBLIC_DOMAIN_URLS:
        logger.error("No sample URL configured for {}", language_code)
        return None

    for url in PUBLIC_DOMAIN_URLS[language_code]:
        try:
            target = out / f"sample_{language_code}.{url.rsplit('.', 1)[-1]}"
            logger.info("Fetching {} → {}", url, target)
            urlretrieve(url, target)
            logger.success("Saved {}", target)
            return str(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fetch failed: {}", exc)
            continue

    logger.error("All public URLs failed; consider running synth fallback.")
    return None


def synth_fallback(out_dir: str = "data", text: str = "Hello, this is a placeholder.") -> str:
    """Generate a TTS placeholder if real audio cannot be downloaded."""
    try:
        from gtts import gTTS  # noqa: PLC0415
    except ImportError:
        logger.error("Install gtts to use the synth fallback: pip install gtts")
        return ""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "sample_synth.mp3"
    gTTS(text, lang="en").save(str(target))
    logger.success("Synth placeholder saved → {}", target)
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="chr")
    parser.add_argument("--synth-fallback", action="store_true")
    args = parser.parse_args()

    path = fetch(args.language)
    if not path and args.synth_fallback:
        synth_fallback()


if __name__ == "__main__":
    main()
