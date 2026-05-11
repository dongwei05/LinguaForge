"""CPU-only smoke test. Slow (~30s/token) but PROVES the inference logic.

Use this to verify Gemma 4 + transformers integration works end-to-end on
your machine. For real development we'll run on Kaggle T4 (16GB VRAM, fits
the full multimodal model comfortably).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env_file()

    print("[1/3] Importing torch + transformers (CPU only)…")
    t0 = time.time()
    import torch  # noqa: F401
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    print(f"    Imported in {time.time() - t0:.1f}s")

    LOCAL_PATH = r"G:\models\gemma-4-E4B-bnb-4bit"
    print(f"\n[2/3] Loading {LOCAL_PATH} on CPU (slow but proves correctness)…")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(LOCAL_PATH)
    # Force CPU and bf16 for the LLM only — skip the 4-bit quant entirely.
    # We pass quantization_config=None via load_in_4bit=False elsewhere; here
    # the simplest correct path is to delete the embedded quant_config.
    import json
    cfg_path = Path(LOCAL_PATH) / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    saved_quant = cfg.pop("quantization_config", None)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(
            LOCAL_PATH,
            torch_dtype="bfloat16",
            device_map="cpu",
        )
    finally:
        # restore for future use
        if saved_quant is not None:
            cfg["quantization_config"] = saved_quant
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"    Loaded in {time.time() - t0:.1f}s")

    print("\n[3/3] One inference (will take a minute or two on CPU)…")
    messages = [
        {"role": "system", "content": "You are LinguaForge, a tutor for endangered languages."},
        {"role": "user",
         "content": "In two sentences: name one endangered language and one phrase from it."},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt")

    t0 = time.time()
    out = model.generate(
        **inputs, max_new_tokens=80, do_sample=False, temperature=1.0, top_p=0.95, top_k=64,
    )
    elapsed = time.time() - t0
    response = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("\n" + "=" * 72)
    print("Q: Name one endangered language and one phrase from it.")
    print(f"A: {response.strip()}")
    print("=" * 72)
    n_new = out.shape[1] - inputs["input_ids"].shape[-1]
    print(f"\n{n_new} tokens in {elapsed:.1f}s ({n_new/elapsed:.2f} tok/s — CPU is slow but logic verified)")


if __name__ == "__main__":
    main()
