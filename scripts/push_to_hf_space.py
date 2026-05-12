"""Create + populate a public Hugging Face Space for LinguaForge.

Reads HF_TOKEN from env (must have write scope), uploads `space/` folder to
`zcgf111/LinguaForge`. Free ZeroGPU tier; uses `spaces` decorator in `app.py`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login


HF_USER = "zcgf111"
SPACE_REPO = f"{HF_USER}/LinguaForge"
SPACE_DIR = Path(__file__).resolve().parents[1] / "space"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var missing.")
    assert SPACE_DIR.is_dir(), f"Space dir not found: {SPACE_DIR}"

    print("Space content:")
    for f in sorted(SPACE_DIR.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1024:.1f} KB")

    login(token=token)
    api = HfApi(token=token)
    api.create_repo(
        SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print(f"\nSpace ready: https://huggingface.co/spaces/{SPACE_REPO}")

    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="Deploy LinguaForge demo: base vs +LoRA Gemma 4 E4B side-by-side",
    )
    print(f"\nUpload done. View at https://huggingface.co/spaces/{SPACE_REPO}")
    print("Build progress: https://huggingface.co/spaces/" + SPACE_REPO + "?logs=container")


if __name__ == "__main__":
    main()
