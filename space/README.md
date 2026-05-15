---
title: LinguaForge — Gemma 4 LoRA for 204 endangered languages
emoji: 🌍
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.15.0
python_version: "3.10.13"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Base Gemma 4 vs LinguaForge LoRA on 204 languages
models:
  - google/gemma-4-E4B-it
  - zcgf111/linguaforge-gemma4-204lang-lora
hf_oauth: false
---

# LinguaForge — multilingual Gemma 4 demo

Live side-by-side comparison of the base
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) and our
LoRA adapter trained on **all 203 non-English languages in FLORES-200** plus
**Cherokee** from the ChrEn corpus (Zhang et al., EMNLP 2020).

## What this Space shows

1. **Translate** — paste an English sentence, pick a target language; the same
   prompt goes to the base model and to the LoRA-adapted model, and you see
   both answers next to each other.
2. **Tutor chat** — multi-turn conversation as a language tutor; LoRA-on.
3. **Numbers** — held-out FLORES-200 / ChrEn BLEU & chrF table.

## Engineering

- **Adapter:** [`zcgf111/linguaforge-gemma4-204lang-lora`](https://huggingface.co/zcgf111/linguaforge-gemma4-204lang-lora) (169.7 MB, r=16, α=32, 7 targets).
- **Backend:** Unsloth (`FastLanguageModel`) loads the adapter directly; base
  weights are pulled in 4-bit NF4 from HF Hub.
- **Runtime:** ZeroGPU lease per call (`@spaces.GPU(duration=90)`). The first
  request is slow because the model is being placed on the H200; subsequent
  requests run at full GPU speed.

## Reproducibility

- Training Kaggle kernel: [`dongwei666/linguaforge-auto`](https://www.kaggle.com/code/dongwei666/linguaforge-auto)
- Evaluation Kaggle kernel: [`dongwei666/linguaforge-eval`](https://www.kaggle.com/code/dongwei666/linguaforge-eval)
- Local source: GitHub repo linked from the writeup.

## Citations

- NLLB Team. *Scaling neural machine translation to 200 languages.* Nature, 2024.
- Zhang, Frey, Bansal. *ChrEn: Cherokee-English Machine Translation for Endangered Language Revitalization.* EMNLP 2020.

Apache-2.0 for code, CC-BY-SA 4.0 for the adapter weights.
