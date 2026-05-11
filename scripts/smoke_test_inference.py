"""Smoke test: load Gemma 4 E4B in 4-bit and run one inference.

Loads the unsloth pre-quantized 4-bit weights from the local path that
`scripts/download_e4b_curl.ps1` populates. We use the local path (not the HF
hub identifier) because Hugging Face's Xet protocol is unreliable in mainland
China; we download via direct HTTPS to hf-mirror.com instead.

Expected runtime on RTX 4060 8GB:
  * first time loading (cold disk cache): ~90s
  * subsequent runs: ~30s load + 3-5s inference
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
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"

    print(f"HF cache       : {os.environ.get('HF_HOME', '<default>')}")
    print(f"Token          : {os.environ['HF_TOKEN'][:8]}...")

    print("\n[1/4] Importing torch + transformers…")
    t0 = time.time()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    print(f"    Imported in {time.time() - t0:.1f}s")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU            : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    LOCAL_MODEL_PATH = r"G:\models\gemma-4-E4B-bnb-4bit"
    HF_FALLBACK_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

    if Path(LOCAL_MODEL_PATH).exists() and any(Path(LOCAL_MODEL_PATH).glob("*.safetensors")):
        model_source = LOCAL_MODEL_PATH
        print(f"\n[2/4] Using local pre-quantized weights at {model_source}")
    else:
        model_source = HF_FALLBACK_ID
        print(f"\n[2/4] Local weights missing; falling back to HF hub: {model_source}")
        print("    Tip: run `powershell -File scripts\\download_e4b_curl.ps1` first.")

    print(f"\n[3/4] Loading {model_source}…")
    print("    Using CPU offload for layers that don't fit in 8GB VRAM.")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_source, token=os.environ.get("HF_TOKEN"))

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        device_map="auto",
        max_memory={0: "6.5GiB", "cpu": "24GiB"},
        quantization_config=bnb_cfg,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"    Model loaded in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        print(f"    VRAM used      : {used_gb:.2f} GB")

    print("\n[4/4] Running one tiny inference …")
    messages = [
        {"role": "system", "content": "You are LinguaForge, a tutor for endangered languages."},
        {"role": "user",
         "content": "In one sentence: why is preserving an endangered language an act of love?"},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    elapsed = time.time() - t0
    response = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("\n" + "=" * 72)
    print("Q: Why is preserving an endangered language an act of love?")
    print(f"A: {response.strip()}")
    print("=" * 72)
    print(f"\nGenerated {out.shape[1] - inputs['input_ids'].shape[-1]} tokens in {elapsed:.1f}s")
    print(f"Throughput: {(out.shape[1] - inputs['input_ids'].shape[-1]) / elapsed:.1f} tok/s")
    print("\n[OK] Smoke test passed. Gemma 4 E4B is alive on your machine.")


if __name__ == "__main__":
    main()
