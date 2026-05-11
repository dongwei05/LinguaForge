# LinguaForge — GGUF Q4_K_M export Kaggle kernel

**Kernel:** [`dongwei666/linguaforge-gguf`](https://www.kaggle.com/code/dongwei666/linguaforge-gguf)
**Source:** `linguaforge_gguf.ipynb`
**Mounted dataset:** `dongwei666/linguaforge-gemma4-204lang-lora`

## What it does (six steps)

1. Mounts the LinguaForge LoRA dataset and loads it on top of
   `google/gemma-4-E4B-it` in 4-bit via Unsloth.
2. **Merges the LoRA into FP16 weights** (`save_pretrained_merged`,
   `save_method='merged_16bit'`) into `/tmp/merged_fp16` — `/kaggle/working`
   is only 20 GB so we route the heavy intermediate through `/tmp` (~73 GB).
3. Clones and builds **llama.cpp** (`llama-quantize` + `llama-cli` targets).
4. Converts merged FP16 → GGUF F16 with `convert_hf_to_gguf.py`.
5. Quantises GGUF F16 → **Q4_K_M** (~4.7 GB).
6. Runs a tiny CPU benchmark with `llama-cli -t 4`, parses the
   "tokens/sec" line out of stderr, and saves it to `cpu_bench.json`.

Final published outputs in `/kaggle/working/gguf/`:

| File | Purpose |
|---|---|
| `linguaforge-gemma4-e4b.Q4_K_M.gguf` | quantised model, ready to ship |
| `Modelfile` | drop-in for `ollama create linguaforge-cherokee -f Modelfile` |
| `cpu_bench.json` | `{elapsed_s, tokens_per_sec_cpu, eval_tokens}` |

## Local Ollama install (one minute)

```bash
# 1. download the GGUF + Modelfile from the kernel output (or from this repo)
kaggle kernels output dongwei666/linguaforge-gguf -p ./gguf

# 2. register with Ollama
cd gguf
ollama create linguaforge -f Modelfile

# 3. run offline
ollama run linguaforge "Translate this English sentence into Cherokee (Iroquoian, North America):\\n\\nThe river remembers every footstep on its bank."
```

That command works with **no internet connection**. The whole model fits
under 5 GB on disk and ~6 GB of RAM on a CPU laptop, which is the
"$100 phone in a remote village" deployment target from the writeup.

## Pipeline disk budget (why this is fiddly)

| Stage | Disk used | Where |
|---|---|---|
| Base model download | ~5 GB | `~/.cache/huggingface/hub` |
| `merged_fp16` | ~16 GB | `/tmp/merged_fp16` |
| `f16.gguf` | ~16 GB | `/tmp/gguf/` (deleted immediately after quantise) |
| `Q4_K_M.gguf` | ~4.7 GB | `/kaggle/working/gguf/` (published) |

Without the `/tmp` routing the kernel hits *No space left on device* near
the end of the GGUF-f16 conversion. The v4 notebook fixes this.

## Reproducer

```bash
export HF_TOKEN=hf_xxx
kaggle kernels push -p notebooks/auto_run_gguf
```

The HF token is required to download `google/gemma-4-E4B-it`.
