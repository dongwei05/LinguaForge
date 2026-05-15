"""Upload LoRA adapter to HF Hub using hf_transfer + per-file uploads with
progress prints, so we know whether it's actually doing work."""
import os
import sys
import time
from pathlib import Path

# Enable hf_transfer for faster multipart uploads.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, login

HF_USER = "zcgf111"
MODEL_REPO = f"{HF_USER}/linguaforge-gemma4-204lang-lora"
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "auto_run" / "out_v8_adapter" / "lora_out"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var missing.")

    print(f"Adapter dir: {ADAPTER_DIR}")
    files = [f for f in sorted(ADAPTER_DIR.iterdir()) if f.is_file()]
    for f in files:
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.2f} MB")

    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    print(f"\nRepo ready: https://huggingface.co/{MODEL_REPO}\n", flush=True)

    # Upload smaller files first via per-file API so we see progress.
    for f in files:
        if f.name == "dataset-metadata.json":
            continue  # this one is for kaggle, not HF
        size_mb = f.stat().st_size / 1e6
        print(f"[{time.strftime('%H:%M:%S')}] uploading {f.name} ({size_mb:.1f} MB) ...", flush=True)
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Add {f.name}",
        )
        dt = time.time() - t0
        rate = size_mb / dt if dt > 0 else 0
        print(f"  done in {dt:.1f}s ({rate:.2f} MB/s)", flush=True)

    print(f"\nView at https://huggingface.co/{MODEL_REPO}", flush=True)


if __name__ == "__main__":
    main()
