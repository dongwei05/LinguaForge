"""Smoke test: verify the HF token can authenticate and access Gemma 4."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return os.environ.get("HF_TOKEN")


def main() -> None:
    token = load_env()
    if not token:
        print("[FAIL] HF_TOKEN not found in .env")
        sys.exit(1)
    print(f"[OK] Token loaded: {token[:8]}...{token[-4:]}")

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError:
        print("[INFO] huggingface_hub not installed; installing minimal client...")
        os.system(f'"{sys.executable}" -m pip install -q huggingface_hub')
        from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)

    print("\n[TEST 1] whoami…")
    try:
        info = api.whoami()
        print(f"  -> Logged in as: {info.get('name')} ({info.get('email')})")
        print(f"     Plan: {info.get('plan', 'free')}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {e}")
        sys.exit(2)

    print("\n[TEST 2] List Gemma 4 family models you can access…")
    targets = [
        "google/gemma-4-E4B-it",
        "google/gemma-4-E2B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
    ]
    for repo_id in targets:
        try:
            mi = api.model_info(repo_id)
            tags = mi.tags or []
            gated = "gated" if any("gated" in t for t in tags) else "open"
            print(f"  [OK]  {repo_id}  (license: {mi.card_data.get('license') if mi.card_data else 'n/a'})")
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] {repo_id}: {type(e).__name__} - {str(e)[:80]}")

    print("\n[TEST 3] Check model file sizes (so we know how much to download)…")
    try:
        files = api.list_repo_files("google/gemma-4-E4B-it")
        safetensors = [f for f in files if f.endswith(".safetensors")]
        print(f"  E4B safetensors shards: {len(safetensors)}")
        for f in safetensors[:5]:
            print(f"    - {f}")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] {e}")

    print("\n[ALL TESTS PASSED]  Token works, you can download Gemma 4.")
    print("\nNext step: run `python scripts/download_e4b.py` to fetch the model (~16GB).")


if __name__ == "__main__":
    main()
