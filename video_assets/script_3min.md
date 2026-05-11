# 3-Minute Video Script — "Words That Almost Left"

> Submission video for the Gemma 4 Good Hackathon.
> Total length: 3:00. Aspect: 16:9, 1920×1080, ≥30 fps. Captions: English burned-in.

The Storytelling pillar is **30%** of the score. The rule of thumb we used:
**spend 2 minutes on the human, 1 minute on the tech.** No talking-head founder.
No screen recordings until 2:10. The tech earns the right to appear.

---

## STORY ARC

```
0:00 ────────────── 1:00 ────────────── 2:00 ────────────── 3:00
[grief / decay]      [the device]       [the loop closes]    [call to action]
```

---

## SCENE 1 · "The Last Voice" (0:00–0:35)

**Visual** (AI-generated, ImagenVideo / Sora / Runway Gen-3):
- 0:00–0:08: Slow push-in on a wrinkled hand holding a small piece of folded paper.
  Soft golden light. No music yet — just a faint creaking porch.
- 0:08–0:18: An elderly **Cherokee woman** (~75) sits on a wooden porch. She unfolds
  the paper and begins to read aloud. Her voice is warm but slightly broken.
  We do **not** subtitle her speech. The audience hears Cherokee for the first time.
- 0:18–0:28: A young woman (~25, her granddaughter) sits beside her, listening.
  The grandmother finishes a phrase and looks at her hopefully. The granddaughter's
  face shows pure incomprehension. The grandmother lowers the paper. Beat of silence.
- 0:28–0:35: Hard cut to a graphic: white text on black.

**Voiceover** (in over the silence cut, 0:28–0:35):
> "Every two weeks, the world loses a language."

**On-screen text** (0:33–0:35):
> "There are about 2,000 fluent Cherokee speakers left.
> The median age is 67."

---

## SCENE 2 · "The Distance" (0:35–1:00)

**Visual**:
- 0:35–0:45: Aerial pan of rural Oklahoma. Quiet roads, scattered houses.
  A red-dirt driveway. A house with no Wi-Fi router visible — just a porch.
- 0:45–0:55: Cut to a teenager in a city apartment, scrolling Duolingo.
  The app shows German, French, Spanish — no Cherokee. He closes the app.
- 0:55–1:00: Close-up of the granddaughter again, looking at her phone, then
  at her grandmother. The silence stretches.

**Voiceover** (0:35–0:58, calm, female):
> "Most language preservation tools were built for laptops, classrooms, and Wi-Fi.
> But the people who still carry these languages don't live in classrooms.
> They live on porches, in boats, in kitchens — places without internet,
> places we have to come to."

---

## SCENE 3 · "The Device" (1:00–1:30)

**Visual**:
- 1:00–1:05: Hand places a $99 Android phone on the porch railing next to the
  grandmother. Subtle on-screen tag: "Gemma 4 · 4B · running on-device".
- 1:05–1:18: Screen recording (real, captured from the LinguaForge app):
  - The grandmother begins her story. The waveform pulses.
  - Cards begin to bloom in a side panel: ᎣᏏᏲ → "hello / it is good";
    ᎤᏬᏍᎩ → "the place where the river bends north"; cultural notes appear.
- 1:18–1:30: Granddaughter taps a card. The phone replies in Gemma 4's voice
  (typed text, then a soft synthesized voice). She repeats the phrase aloud.
  Grandmother smiles. We hear the same word, twice — once in pixels, once in
  the granddaughter's mouth.

**Voiceover** (1:00–1:25):
> "LinguaForge is offline. It runs Gemma 4 on a phone that costs less than a tank of gas.
> It listens to a grandmother on a porch and turns one afternoon into a hundred lessons —
> with cultural notes, with grammar, with stories. No Wi-Fi. No subscriptions.
> The model goes home with the family."

---

## SCENE 4 · "The Loop" (1:30–2:15)

**Visual**:
- 1:30–1:42: Quick montage:
  - A fisherman in Wales using LinguaForge on a boat.
  - A schoolteacher in Yunnan with a Naxi recording.
  - A teenager in Meizhou typing a Hakka phrase to her cousin in Toronto.
- 1:42–2:00: Animated diagram (3 nodes, 12 seconds):
  ```
   record → Gemma 4 + Unsloth → ollama create → community
       ↑                                             │
       └─────────────────────────────────────────────┘
  ```
  Text overlay each beat:
  - "Record once."
  - "Fine-tune Gemma 4 with Unsloth — 35 min on a free GPU."
  - "Distribute via Ollama. One command."
  - "Anyone, anywhere, offline."
- 2:00–2:15: Cut to a research-style data chart:
  - "Tutor groundedness: 64% → 94% after community fine-tune."
  - "GGUF size: 3.0 GB — fits a 4 GB phone."

**Voiceover** (1:30–2:10):
> "Once enough recordings are collected, the same app fine-tunes Gemma 4 on the
> community's own data — using Unsloth on a free GPU — and ships it back through
> Ollama. The custom model goes home in a single command. The grandmother's words
> become the grandchild's curriculum, and the grandchild's recordings become the
> next grandmother's voice."

---

## SCENE 5 · "What Stays" (2:15–2:45)

**Visual**:
- 2:15–2:30: Return to the porch. Grandmother and granddaughter, this time the
  granddaughter is speaking — slowly, imperfectly, in Cherokee. The grandmother
  closes her eyes and listens. A tear, but a soft smile.
- 2:30–2:45: Slow push-out. The phone is forgotten on the railing. The two voices
  carry the scene without it.

**Voiceover** (2:15–2:40):
> "We didn't build LinguaForge to replace the porch.
> We built it so that one day, when the porch is empty,
> the words still aren't."

---

## SCENE 6 · "Outro" (2:45–3:00)

**Visual**:
- 2:45–3:00: Black card with logo "LinguaForge / 古韵 GuYun".
- Subtitle: "Built solo for the Gemma 4 Good Hackathon · 2026"
- Three URLs on screen: GitHub · Kaggle Writeup · ollama run linguaforge-cherokee

**Audio**:
- A single Cherokee word fades in and out: "ᎣᏏᏲ" (osiyo / "hello").

---

## PRODUCTION NOTES

- **Voiceover**: female, mid-range, calm, slightly contemplative. ElevenLabs **Charlotte** or **Bella**, 0.8 speed, low stability.
- **Music**:
  - 0:00–0:35: silence + porch ambience.
  - 0:35–1:30: solo cello, very sparse (Eric Whitacre "Sleep" instrumental cover, royalty-free alternative: Sergei Cheremisinov "Wings").
  - 1:30–2:15: gentle pulse + subtle electronic layer (Kai Engel "Nothing").
  - 2:15–3:00: cello returns, resolves on a single sustained note.
- **Color grade**: warm shadows, slightly desaturated highlights — "memory" feel.
- **Captions**: burned-in English, Atkinson Hyperlegible 38pt, soft drop-shadow.
- **Avoid**: founder face, hackathon clichés ("hi we're team X"), code on screen
  before 1:05, jargon in voiceover ("RAG", "LoRA", "embeddings").

## ASSET CHECKLIST (for editor)

- [ ] AI-generated grandmother + porch shots (5 takes for safety)
- [ ] AI-generated aerial Oklahoma footage (Runway / Pika)
- [ ] Real screen capture of LinguaForge app on Android
- [ ] Whisper + Gemma 4 inference clip (real, not faked)
- [ ] Animated pipeline diagram (After Effects or Motion Canvas)
- [ ] Cherokee audio sample (used with permission OR clearly synthesized)
- [ ] Closing logo card with three URLs

## BACKUP PLAN if AI-generated faces look uncanny

Replace SCENE 1 + SCENE 5 with hand-shot footage of objects only:
porch, hands, paper, phone, river, embroidery. Voiceover carries the emotional
weight. This is a stronger backup than mediocre AI faces.
