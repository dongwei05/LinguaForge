# `auto_run/` — fully reproducible Kaggle kernel

This folder is what the Kaggle API ate when we pushed the LinguaForge submission
notebook end-to-end without any manual cell clicks. The metadata, source, and
output evidence are all checked into the repository so a judge can reproduce
the run with **two commands** on their own machine:

```bash
# 1. Authenticate (Kaggle API token in ~/.kaggle/access_token or env)
kaggle kernels push -p notebooks/auto_run

# 2. After completion, pull the executed artifacts
kaggle kernels output dongwei666/linguaforge-auto -p notebooks/auto_run/out
```

## What the run produced (Kernel v8 — planet-scale, Tesla T4)

| Section | Result |
|---|---|
| 1. Setup | HF token loaded, deps installed, `Tesla T4` with 14.6 GB usable VRAM |
| 2. Load Gemma 4 E4B | `Loaded google/gemma-4-E4B-it in 119.7s`, resident `2.49 GB VRAM` |
| 2. First inference | "Preserving an endangered language is an act of love because it honors the unique worldview, cultural heritage, and intricate knowledge system of a community, allowing their history and identity to continue flourishing for future generations." |
| 3. Listen | gracefully skipped (no Cherokee public-domain audio attached) |
| 4. Learn (function call) | `call:search_cards{query:Cherokee greeting}` ✅ native tool call |
| 5. Revive — fetch FLORES-200 | `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` extracted to `/kaggle/working/flores200_dataset` |
| 5. Revive — multilingual corpus | **203 / 203 FLORES-200 non-English languages swept** + 500 ChrEn Cherokee pairs |
| 5. Revive — corpus stats | **33,480 chat samples** across **204 languages**, 6 continents, 14+ writing systems, 30+ language families |
| 5. Revive (Unsloth LoRA) | `42,401,792 of 8,038,558,240 (0.53%) trainable`, **8,370 steps in ~5 h (2.16 s/step)** |
| Adapter on disk | `adapter_model.safetensors` — **169.7 MB** (covers all 204 languages in one file) |
| Total wall-clock | **~5 h 9 min** including pip installs, model download, FLORES tarball, and full-corpus training |

## Files

- `linguaforge_auto.ipynb` — source notebook (HF token baked in for kernel-only use; rotate after final submission)
- `kernel-metadata.json` — Kaggle kernel config (T4, GPU on, internet on, private)
- `extract_outputs.py` — local helper to flatten any executed `.ipynb` into `outputs.md`/`outputs.json`
- `out_v8_adapter/` — artifacts pulled back after the run:
  - `out_v8_adapter/lora_out/adapter_model.safetensors` — **trained LoRA adapter, 169.7 MB**
  - `out_v8_adapter/lora_out/adapter_config.json` — PEFT config
  - `out_v8_adapter/lora_out/tokenizer*.json` — tokenizer with restored Unsloth metadata
  - `out_v8_adapter/linguaforge-auto.log` — papermill stream log with all stdout/stderr

## Why "auto_run" exists

This was originally a regular Kaggle notebook the user ran cell by cell. We
rewrote it so that the **same notebook, same model, same prompts** can be run
without anybody touching a screen — pushed by `kaggle kernels push`, watched
by `kaggle kernels status`, harvested by `kaggle kernels output`. That makes
it a one-button reproducer for reviewers and removes "but it worked on my
machine" from the conversation.
