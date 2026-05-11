# LinguaForge — Listen pillar Kaggle kernel

**Kernel:** [`dongwei666/linguaforge-listen`](https://www.kaggle.com/code/dongwei666/linguaforge-listen)
**Source:** `linguaforge_listen.ipynb`
**Mounted dataset:** `dongwei666/linguaforge-gemma4-204lang-lora`

## What it does

1. Downloads a 43-second Cherokee morning song from Wikimedia Commons
   (`Morning-song-on-Cherokee (1).opus` by user Bono.Ruma, CC-BY-SA 4.0).
2. Resamples to 16 kHz mono WAV with `ffmpeg`.
3. Loads Google DeepMind's `google/gemma-4-E4B-it` in 4-bit NF4 via Unsloth,
   then applies the **LinguaForge LoRA** trained across 204 languages.
4. Feeds the audio + a structured prompt to the multimodal processor and
   asks for: (a) language/musical-style identification, (b) vocal
   characteristics, (c) three Cherokee vocabulary cards for a learner.
5. Runs the **exact same prompt twice** — once with the LoRA disabled
   (`model.disable_adapter()`), once with it enabled — so we can compare
   base Gemma 4 vs. LinguaForge head-to-head on the same recording.
6. Saves the structured output as `listen_results.json`.

## Why this kernel matters for the writeup

It is the only place in the whole submission where we exercise Gemma 4's
**audio modality**. The base model fails gracefully (says it cannot
transcribe, falls back to canonical Cherokee greetings). The LoRA gives
better audio analysis but a vocabulary-card repetition loop appears —
honest evidence for the writeup's "per-community LoRAs next" roadmap.

## Reproducer

```bash
# from notebooks/auto_run_listen/
export HF_TOKEN=hf_xxx     # needs read access to google/gemma-4-E4B-it
kaggle kernels push -p .
```

> The HF token must be available as Kaggle Secret (`HF_TOKEN`) on the
> account running the kernel; the source notebook reads from
> `kaggle_secrets.UserSecretsClient` first and falls back to the env var,
> so neither path leaks the token into the committed `.ipynb`.

## Output files (after a successful run)

| Path | Purpose |
|---|---|
| `out_v3/listen_results.json` | structured `{base_output, lora_output, audio_source}` |
| `out_v3/linguaforge-listen.log` | full papermill log |

Both are mirrored under `out_v3/` in this folder for archival reproducibility.
