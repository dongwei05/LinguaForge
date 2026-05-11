"""Push a Kaggle kernel with HF_TOKEN injected into the notebook (in a temp dir).

The on-disk source notebook stays sanitized for GitHub. Use this helper for any
kernel where Kaggle Secrets aren't available.

Usage:
    set HF_TOKEN=hf_...
    python scripts/push_kaggle_with_token.py notebooks/auto_run_gguf
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TOKEN_PLACEHOLDER_LINE = "assert os.environ.get('HF_TOKEN'), 'HF_TOKEN missing"


def inject(nb_path: Path, token: str) -> dict:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    injected = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "HF_TOKEN" not in src:
            continue
        if "os.environ['HF_TOKEN'] = '" in src:  # already has hardcoded token
            continue
        # Find the assert line and prepend a hardcoded assignment above it.
        new_src = src.replace(
            "if not os.environ.get('HF_TOKEN'):",
            f"# (Token injected by push helper for Kaggle run; never commit this.)\nos.environ.setdefault('HF_TOKEN', {token!r})\nif not os.environ.get('HF_TOKEN'):",
            1,
        )
        cell["source"] = new_src.splitlines(keepends=True)
        injected = True
        break
    if not injected:
        raise SystemExit(f"Could not find an HF_TOKEN cell in {nb_path}")
    return nb


def main(kernel_dir: str):
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var missing.")
    kdir = Path(kernel_dir).resolve()
    assert kdir.is_dir(), f"Not a directory: {kdir}"
    meta = kdir / "kernel-metadata.json"
    nbs = list(kdir.glob("*.ipynb"))
    assert nbs, f"No notebook found in {kdir}"
    nb_path = nbs[0]
    print(f"Kernel dir: {kdir}")
    print(f"Notebook  : {nb_path.name}")
    print(f"Token tail: ...{token[-6:]}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy2(meta, td / "kernel-metadata.json")
        injected = inject(nb_path, token)
        out_path = td / nb_path.name
        out_path.write_text(json.dumps(injected, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Temp dir: {td}")
        print("Pushing to Kaggle ...")
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(td)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        print("--- stdout ---")
        print(res.stdout)
        print("--- stderr ---")
        print(res.stderr)
        sys.exit(res.returncode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: push_kaggle_with_token.py <kernel_dir>")
    main(sys.argv[1])
