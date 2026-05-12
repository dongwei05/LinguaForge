# LinguaForge — Final submission checklist

A short, mechanical checklist for the **day of submission**. Run through it
top to bottom in one sitting; budget ~30 minutes.

## 0. Identity & contact (Kaggle Hackathon entry form)

- **Display name:** `dongwei` (matches HF + Kaggle profiles)
- **Email:** `zcgf111@163.com`
- **Phone (China):** `+86 18702157158`
- **Kaggle username:** `dongwei666`
- **HF username:** `zcgf111`
- **GitHub username:** `<fill in once gh repo create succeeds>`

> One legal name is enough; the Kaggle form lets one submitter represent the
> team. Tracks ticked: **Main**, **Impact / Digital Equity & Inclusivity**,
> **Special Technology (Unsloth + Ollama)**.

## 1. Flip the Kaggle artefacts to *public*

The reproducer kernels are currently private. Open each link and click
**Settings → Sharing → Public** in the right-hand panel:

- [`dongwei666/linguaforge-auto`](https://www.kaggle.com/code/dongwei666/linguaforge-auto)
- [`dongwei666/linguaforge-eval`](https://www.kaggle.com/code/dongwei666/linguaforge-eval)
- [`dongwei666/linguaforge-listen`](https://www.kaggle.com/code/dongwei666/linguaforge-listen)
- [`dongwei666/linguaforge-gguf`](https://www.kaggle.com/code/dongwei666/linguaforge-gguf)

…and the dataset that all four kernels mount:

- [`dongwei666/linguaforge-gemma4-204lang-lora`](https://www.kaggle.com/datasets/dongwei666/linguaforge-gemma4-204lang-lora) — **Settings → Public**

Sanity check after flipping: open each link in an *incognito* tab and confirm
the badge says **Public**.

## 2. Hugging Face

- Adapter model: `huggingface.co/zcgf111/linguaforge-gemma4-204lang-lora` — confirm visibility is **Public** and the model card renders.
- Live Space: `huggingface.co/spaces/zcgf111/LinguaForge` — confirm it loads, the first translation call returns Cherokee within ~60 s (ZeroGPU cold start), and the Numbers tab shows the BLEU/chrF table.

## 3. GitHub

- Repo: `https://github.com/<your-handle>/LinguaForge` — public, README renders cleanly, no token / `.env` committed.
- Smoke check the README: it should link to the HF Space, both Kaggle kernels, the HF Hub adapter, and the YouTube pitch video.

## 4. Pitch video

- Render `writeup/video_script.md` into a 2:55 video (see that file for AI-video-gen prompts vs. things to record yourself).
- Upload **unlisted** to YouTube.
- Paste link into the writeup, README, and HF Space README.

## 5. Kaggle writeup

- Open the [Gemma 4 Hackathon writeup page](https://kaggle.com/competitions/gemma-4-good-hackathon).
- Copy-paste the rendered version of `writeup/writeup.md` into the writeup editor.
- After paste: check the BLEU/chrF table renders, image link works, all Kaggle / HF / YouTube / GitHub URLs are clickable.
- Hit **Submit** and screenshot the confirmation.

## 6. Hard checks before clicking Submit

- [ ] `grep -r hf_AiDcM .` returns **zero hits** anywhere under the GitHub repo.
- [ ] `git log --all --oneline` shows only the four meaningful commits.
- [ ] No PII other than what's in section 0 above is anywhere in code.
- [ ] All licenses present: MIT on code, CC-BY-SA 4.0 on adapter weights, attribution to NLLB + ChrEn + Bono.Ruma.
- [ ] HF Space loads from an incognito browser tab.
- [ ] Kaggle writeup loads from an incognito browser tab.
- [ ] YouTube pitch video plays from an incognito browser tab.

When all six boxes are ticked, you are submission-ready.
