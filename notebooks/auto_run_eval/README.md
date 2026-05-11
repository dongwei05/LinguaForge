# `auto_run_eval/` — held-out BLEU + chrF evaluation of the v8 LoRA

This kernel is the quantitative complement to `notebooks/auto_run/`. It
loads the trained 169.7 MB LoRA adapter, toggles it on and off on the same
Gemma 4 E4B base, and translates 50 unseen FLORES-200 devtest sentences
(plus 50 held-out ChrEn pairs for Cherokee, drawn with a different random
seed than training) for each of 6 endangered languages spanning 6 continents.

```bash
# 1. Push and run (Kaggle Python API token in ~/.kaggle/access_token or env)
kaggle kernels push -p notebooks/auto_run_eval

# 2. After completion, fetch the metrics JSON
kaggle kernels output dongwei666/linguaforge-eval \
  -p notebooks/auto_run_eval/out_eval \
  --file-pattern "eval_results.json"
```

## Headline results

Greedy decoding (`do_sample=False`), `max_new_tokens=128`, identical prompt
format to training, evaluated with `sacrebleu` corpus-level BLEU and chrF.

| Lang | base BLEU | +LoRA BLEU | Δ BLEU | base chrF | +LoRA chrF | Δ chrF |
|---|---:|---:|---:|---:|---:|---:|
| Cherokee (`chr_Cher`) | 0.04 | **0.45** | **+0.41** | 2.30 | **7.87** | **+5.56** (3.4×) |
| Tibetan (`bod_Tibt`)  | 0.12 | **0.21** | +0.09 | 19.14 | **27.05** | **+7.91** |
| Welsh (`cym_Latn`)    | 3.90 | **6.13** | **+2.23** | 31.11 | 31.21 | +0.10 |
| Quechua (`quy_Latn`)  | 1.02 | **1.93** | +0.91 | 19.94 | **22.49** | +2.55 |
| Māori (`mri_Latn`)    | 3.64 | **4.16** | +0.52 | 28.48 | 27.58 | −0.90 |
| Yoruba (`yor_Latn`)   | 2.54 | 1.12 | **−1.42** | 21.65 | 11.10 | **−10.55** ⚠ |
| **MEAN** (6 langs)    | **1.88** | **2.33** | **+0.45** | **20.44** | **21.22** | **+0.78** |

* The biggest wins are for non-Latin scripts the base model could barely
  produce: Cherokee chrF 2.30 → 7.87, Tibetan chrF 19.14 → 27.05.
* Welsh BLEU jumps +2.23, largely because the LoRA strips the base model's
  habit of prefixing replies with `**Welsh Translation:**`.
* Yoruba regressed: the adapter falls into a degenerate
  `àgbọ́n àgbọ́n àgbọ́n…` repetition loop on long inputs. We report this
  transparently rather than hiding it; the fix is more samples per language
  (we used 80 each across 203 langs in a single epoch). Two roadmap items
  address it: per-community LoRAs, and a coverage-floor sampler that
  oversamples low-resource Niger-Congo languages on the next sweep.

## Files

- `linguaforge_eval.ipynb` — source notebook (HF token baked in for
  kernel-only use; rotate after final submission)
- `kernel-metadata.json` — Kaggle kernel config; consumes the trained LoRA
  via `dataset_sources: ["dongwei666/linguaforge-gemma4-204lang-lora"]`
- `eval_results.json` — full machine-readable metrics + 3 qualitative
  samples per language (saved snapshot of kernel output)
- `linguaforge-eval.log` — papermill stream log with all stdout/stderr
  (saved snapshot of kernel output)
- `out_eval/` — fresh kernel pulls, ignored by version control after the
  snapshot above is checked in
