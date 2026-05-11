# LinguaForge / 古韵 GuYun

> Offline AI for endangered language preservation, powered by Gemma 4.
> Submission for the **Gemma 4 Good Hackathon** (Kaggle, 2026).

## TL;DR

Every two weeks, the world loses a language. LinguaForge is an offline-first AI
companion that helps communities **Listen** to their elders, **Learn** from
preserved knowledge, and **Revive** their heritage by fine-tuning Gemma 4 on
their own corpus and shipping it through Ollama.

The whole submission rests on **one 169.7 MB LoRA adapter** that we trained on
**all 203 non-English languages in FLORES-200 plus Cherokee from the ChrEn
corpus** — 6 continents, 14 writing systems, 33,480 chat samples, 5 hours
on a free Kaggle T4.

## Highlights (real numbers from the eval kernel)

| Language | base chrF | +LoRA chrF | Δ |
|---|---:|---:|---:|
| Cherokee (`chr_Cher`) | 2.30 | **7.87** | **3.4× ↑** |
| Tibetan (`bod_Tibt`)  | 19.14 | **27.05** | **+7.91** |
| Welsh BLEU (`cym_Latn`) | 3.90 → **6.13** | | +2.23 |
| Yoruba ⚠ | 21.65 | 11.10 | −10.55 (reported transparently) |

See `writeup/writeup.md` for the full panel, methodology, and a frank
discussion of the Yoruba regression.

## Project layout

```
A_Gemma4_Hackathon/
├── README.md                       # this file
├── STRATEGY.md                     # competition strategy, theme selection, plan
├── LICENSE                         # MIT (code); adapter weights are CC-BY-SA 4.0
├── requirements.txt                # all Python deps
│
├── src/                            # local Python implementation
│   ├── config.py                   # languages, models, paths
│   ├── llm.py                      # Gemma 4 client + tool-calling loop
│   ├── listen.py                   # Audio → LearningCard pipeline
│   ├── rag.py                      # Chroma vector store
│   ├── learn.py                    # tutor agent w/ native function calls
│   ├── revive.py                   # Unsloth fine-tune + Ollama export
│   └── agent.py                    # top-level facade
│
├── notebooks/
│   ├── auto_run/                   # training Kaggle kernel (linguaforge-auto)
│   ├── auto_run_eval/              # eval kernel (linguaforge-eval) — BLEU+chrF
│   ├── auto_run_listen/            # Listen pillar kernel — Cherokee audio
│   ├── auto_run_gguf/              # GGUF Q4_K_M export kernel
│   └── linguaforge_demo.ipynb      # interactive demo notebook
│
├── space/                          # Hugging Face Space (Gradio + ZeroGPU)
│   ├── app.py                      # base-vs-LoRA side-by-side UI
│   ├── requirements.txt
│   └── README.md                   # Space metadata header
│
├── demo/
│   └── app.py                      # full local Gradio (Whisper + RAG + Ollama)
│
├── writeup/
│   ├── writeup.md                  # main Kaggle writeup
│   └── video_script.md             # 3-min pitch video script + storyboard
│
├── scripts/                        # CLI helpers
│   ├── push_to_hf.py               # push adapter to HF Hub
│   ├── push_kaggle_with_token.py   # token-injecting Kaggle pusher
│   ├── smoke_test_inference.py     # local Gemma 4 inference sanity check
│   ├── smoke_test_cpu.py
│   └── verify_token.py
│
└── data/                           # local data (gitignored except CC audio)
    └── audio/Morning-song-on-Cherokee.opus   # CC-BY-SA 4.0
```

## Quick start

### Run the demo locally

```bash
pip install -r requirements.txt
python demo/app.py
# then open http://127.0.0.1:7860
```

### Run end-to-end on a Kaggle notebook

Open `notebooks/linguaforge_demo.ipynb` on Kaggle and **Run all** with a free T4
accelerator. Total runtime ≈ 25 min.

### Fine-tune on community data

```bash
# 1. assemble corpus (jsonl with {"messages": [...]} entries)
python -m src.revive dummy-corpus --n 64

# 2. LoRA fine-tune Gemma 4-4B with Unsloth (≈ 35 min on T4)
python -m src.revive train

# 3. export to GGUF for offline use
python -m src.revive export-gguf

# 4. write Modelfile + ship through Ollama
python -m src.revive modelfile --gguf artifacts/gguf/unsloth.Q4_K_M.gguf --language Cherokee
ollama create linguaforge-cherokee -f artifacts/Modelfile
ollama run linguaforge-cherokee
```

## Languages

The released LoRA adapter covers **all 203 non-English languages in FLORES-200**
(Africa 21 · Asia 17 · Europe 4 · Pacific 4 · S. America 3 · Diaspora 1) **plus
Cherokee** from the ChrEn corpus, for a total of **204 languages across 6
continents and 14 writing systems**. We curate ~50 of them as a showcase with
rich human-readable metadata. Adding a new language is one entry in the
showcase dict; no code changes needed.

## What's submitted

- **Public code repo**: this directory (MIT licence)
- **Trained LoRA adapter (169.7 MB)**: HF Hub `zcgf111/linguaforge-gemma4-204lang-lora` + Kaggle dataset `dongwei666/linguaforge-gemma4-204lang-lora`
- **Live demo (Gradio + ZeroGPU)**: HF Space `zcgf111/LinguaForge` — base vs +LoRA side-by-side translation, multi-turn tutor chat, eval numbers tab
- **3-min pitch video**: YouTube unlisted (URL added at submission time)
- **Kaggle writeup**: `writeup/writeup.md` (pasted into Kaggle UI)
- **4 reproducer Kaggle kernels**: `linguaforge-auto` (training, 5 h 9 min), `linguaforge-eval` (BLEU+chrF, 3 h 47 min), `linguaforge-listen` (multimodal audio), `linguaforge-gguf` (Q4_K_M export + bench)

## Tracks targeted

- **Main Track** ($100K) — full submission
- **Impact / Digital Equity & Inclusivity** ($50K bucket) — language preservation thesis
- **Special Technology Track** ($50K bucket) — Unsloth + Ollama integration

## License

Code: MIT.
Cultural materials: please respect community ownership and Indigenous data
sovereignty conventions when using or extending this project.

## Acknowledgements

To the elders. To the linguists keeping field notes alive on shelves.
And to whoever, two thousand years from now, still says *osiyo*.
