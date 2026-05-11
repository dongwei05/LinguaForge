"""Push the trained LoRA adapter to Hugging Face Hub as a public model repo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

HF_USER = "zcgf111"
MODEL_REPO = f"{HF_USER}/linguaforge-gemma4-204lang-lora"
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "auto_run" / "out_v8_adapter" / "lora_out"

MODEL_CARD = """\
---
license: cc-by-sa-4.0
base_model: google/gemma-4-E4B-it
library_name: peft
tags:
  - lora
  - gemma-4
  - multilingual
  - low-resource-languages
  - endangered-languages
language:
  - multilingual
datasets:
  - facebook/flores
  - shiyue/chr_en
---

# LinguaForge — Gemma 4 E4B LoRA across 204 languages

A single ~170 MB LoRA adapter that shifts Google DeepMind's
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
toward every language in **FLORES-200** (Meta NLLB Team, *No Language Left
Behind*, Nature 2024) plus Cherokee depth from the **ChrEn** corpus
(Zhang, Frey & Bansal, EMNLP 2020). Cherokee is *not* in FLORES-200.

Trained as part of the **LinguaForge / 古韵 GuYun** submission to the
Gemma 4 Hackathon (`AI for Good — endangered language preservation`).

## Training summary (Kaggle T4, ~5 h 9 min)

| | |
|---|---|
| Base model | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` (4-bit NF4) |
| Trainable params | 42,401,792 / 8,038,558,240 (0.53%) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Total chat samples | 33,480 (alt. `en → target` / `target → en`) |
| Languages covered | **203 FLORES-200 languages + Cherokee from ChrEn = 204** |
| Continents | 6 (Africa, Asia, Europe, Pacific, South America, Diaspora) + N. America |
| Optimizer steps | 8,370 (1 epoch, batch 2 × grad-accum 2) |
| Reproducer | Kaggle kernel [`dongwei666/linguaforge-auto`](https://www.kaggle.com/code/dongwei666/linguaforge-auto) |

## Held-out evaluation (FLORES-200 devtest + ChrEn seed=99)

Numbers from Kaggle kernel
[`dongwei666/linguaforge-eval`](https://www.kaggle.com/code/dongwei666/linguaforge-eval),
50 unseen sentences per language, greedy decoding, `sacrebleu` corpus-level
metrics.

| Language | base BLEU | +LoRA BLEU | Δ BLEU | base chrF | +LoRA chrF | Δ chrF |
|---|---:|---:|---:|---:|---:|---:|
| Cherokee (`chr_Cher`) | 0.04 | **0.45** | +0.41 | 2.30 | **7.87** | **+5.56** (3.4×) |
| Tibetan (`bod_Tibt`)  | 0.12 | **0.21** | +0.09 | 19.14 | **27.05** | **+7.91** |
| Welsh (`cym_Latn`)    | 3.90 | **6.13** | **+2.23** | 31.11 | 31.21 | +0.10 |
| Quechua (`quy_Latn`)  | 1.02 | **1.93** | +0.91 | 19.94 | **22.49** | +2.55 |
| Māori (`mri_Latn`)    | 3.64 | **4.16** | +0.52 | 28.48 | 27.58 | −0.90 |
| Yoruba (`yor_Latn`)   | 2.54 | 1.12 | −1.42 | 21.65 | 11.10 | −10.55 ⚠ |
| **Mean (6 langs)**    | **1.88** | **2.33** | **+0.45** | **20.44** | **21.22** | **+0.78** |

Honest read: the LoRA's biggest wins are on **languages whose scripts the
base model could barely write** (Cherokee chrF 3.4×, Tibetan chrF +7.91).
Welsh shows the largest BLEU jump (+2.23) — the adapter strips a
`**Welsh Translation:**` boilerplate prefix from the base model. Yoruba
regressed into a repetition loop; reported transparently. With more
samples per language or per-community LoRAs, that regression should
resolve.

## Usage

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

model, tok = FastLanguageModel.from_pretrained(
    model_name="zcgf111/linguaforge-gemma4-204lang-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
tok = get_chat_template(tok, chat_template="gemma")
FastLanguageModel.for_inference(model)

msgs = [
    {"role": "system", "content": "You are LinguaForge, a multilingual tutor for endangered and low-resource languages."},
    {"role": "user",   "content": "Translate this English sentence into Maori (Polynesian, Pacific):\\n\\nHello, my name is Sarah."},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok.tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                         pad_token_id=tok.tokenizer.eos_token_id)
print(tok.tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Citations

```bibtex
@article{nllb2024,
  title={Scaling neural machine translation to 200 languages},
  author={{NLLB Team} and Costa-juss{\\`a}, Marta R. and others},
  journal={Nature},
  year={2024},
  doi={10.1038/s41586-024-07335-x}
}
@inproceedings{zhang-etal-2020-chren,
  title={{ChrEn}: {Cherokee-English} Machine Translation for Endangered Language Revitalization},
  author={Zhang, Shiyue and Frey, Benjamin and Bansal, Mohit},
  booktitle={EMNLP},
  year={2020}
}
```

## License

CC-BY-SA 4.0, matching the FLORES-200 license. ChrEn is released
under CC-BY-SA 4.0 by its authors.
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN env var missing; export it first.")
    assert ADAPTER_DIR.is_dir(), f"Adapter dir not found: {ADAPTER_DIR}"
    print(f"Adapter dir: {ADAPTER_DIR}")
    for f in sorted(ADAPTER_DIR.iterdir()):
        print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.2f} MB")

    login(token=token)
    api = HfApi(token=token)
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    print(f"\nRepo ready: https://huggingface.co/{MODEL_REPO}")

    card_path = ADAPTER_DIR / "README.md"
    card_path.write_text(MODEL_CARD, encoding="utf-8")
    print(f"Wrote model card → {card_path}")

    print(f"\nUploading folder → {MODEL_REPO} ...")
    api.upload_folder(
        folder_path=str(ADAPTER_DIR),
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Initial release: 169.7 MB LoRA across 204 languages (FLORES-200 + ChrEn)",
    )
    print(f"\nDone. View at https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
