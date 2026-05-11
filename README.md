# LinguaForge / 古韵 GuYun

> Offline AI for endangered language preservation, powered by Gemma 4.
> Submission for the Gemma 4 Good Hackathon (Kaggle, 2026).

## TL;DR

Every two weeks, the world loses a language. LinguaForge is an offline-first AI
companion that helps communities **Listen** to their elders, **Learn** from
preserved knowledge, and **Revive** their heritage by fine-tuning Gemma 4 on
their own corpus and shipping it through Ollama.

## Project layout

```
A_Gemma4_Hackathon/
├── README.md                 # this file
├── STRATEGY.md               # competition strategy, theme selection, plan
├── requirements.txt          # all Python deps
│
├── src/                      # Python source
│   ├── config.py             # Languages, models, paths
│   ├── llm.py                # Gemma 4 client + tool-calling loop
│   ├── listen.py             # Audio → LearningCard pipeline
│   ├── rag.py                # Chroma vector store
│   ├── learn.py              # Tutor agent with native function calls
│   ├── revive.py             # Unsloth fine-tune + Ollama export
│   └── agent.py              # Top-level facade
│
├── notebooks/
│   └── linguaforge_demo.ipynb   # End-to-end Kaggle notebook
│
├── demo/
│   └── app.py                   # Gradio live demo
│
├── writeup/
│   └── writeup.md               # The Kaggle writeup we submit
│
├── video_assets/
│   └── script_3min.md           # 3-minute pitch video script
│
└── data/                        # local data (gitignored)
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

## Languages currently supported

| Code | Language | Speakers | Status |
|---|---|---|---|
| `chr` | Cherokee (ᏣᎳᎩ) | ~2,000 | Critically endangered |
| `hak` | Hakka (客家话) | ~44 M | Vulnerable |
| `cy` | Welsh (Cymraeg) | ~900 K | Vulnerable |
| `nax` | Naxi/Dongba (纳西语) | ~300 K | Endangered |

Adding a new language is one entry in `src/config.py`.

## What's submitted

- **Public code repo**: this directory
- **Live demo**: Hugging Face Space (URL added at submission time)
- **3-min video**: YouTube unlisted (URL added at submission time)
- **Kaggle writeup**: `writeup/writeup.md` (pasted into Kaggle UI)

## Tracks targeted

- ✅ **Main Track** ($100K)
- ✅ **Impact / Digital Equity & Inclusivity** ($50K bucket)
- ✅ **Special Technology Track** — Unsloth + Ollama ($50K bucket)

## License

Code: MIT.
Cultural materials: please respect community ownership and Indigenous data
sovereignty conventions when using or extending this project.

## Acknowledgements

To the elders. To the linguists keeping field notes alive on shelves.
And to whoever, two thousand years from now, still says *osiyo*.
