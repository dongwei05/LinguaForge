"""LinguaForge — Gradio live demo.

Run locally:
    python demo/app.py

Or push to Hugging Face Spaces (free GPU tier) for the live submission link.

The interface has three tabs that map 1:1 to the three pillars described in
the writeup: Listen, Learn, Revive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIG, SUPPORTED_LANGUAGES  # noqa: E402
from src.learn import new_session  # noqa: E402
from src.listen import LearningCard, run_pipeline  # noqa: E402
from src.rag import CardStore  # noqa: E402


HEADER_MD = """
# 🌍 LinguaForge / 古韵 GuYun

**An offline AI companion for endangered languages, powered by Gemma 4.**

> Every two weeks the world loses a language. We built LinguaForge so that
> when a grandmother passes, her words don't pass with her.

This is the live demo for the [Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon).
"""


def listen_tab(audio_file, language_code, max_segments):
    if audio_file is None:
        return "Please upload an audio file.", None

    cards = run_pipeline(
        audio_file,
        language_code=language_code,
        max_segments=int(max_segments),
    )
    md_parts = [f"### Generated {len(cards)} learning cards\n"]
    rows = []
    for c in cards[:30]:
        md_parts.append(
            f"**[{c.card_type}]** `{c.native_text}` — {c.english_gloss}"
            + (f"\n> 💡 {c.cultural_note}" if c.cultural_note else "")
        )
        rows.append(
            [
                c.card_id,
                c.card_type,
                c.native_text,
                c.english_gloss,
                c.cultural_note,
            ]
        )

    return "\n\n".join(md_parts), rows


def learn_tab_chat(message, history, language_code, learner_id):
    session = LEARN_SESSIONS.get((learner_id, language_code))
    if session is None:
        session = new_session(learner_id=learner_id, language_code=language_code)
        LEARN_SESSIONS[(learner_id, language_code)] = session
    reply = session.turn(message)
    history = history + [(message, reply)]
    return history, history


def revive_tab(corpus_jsonl, base_model_id):
    return (
        "**This is a long-running operation** (~20-60min on a free Kaggle GPU).\n\n"
        f"Equivalent CLI:\n```bash\n"
        f"python -m src.revive train --model-id {base_model_id} "
        f"--dataset {corpus_jsonl}\n"
        f"python -m src.revive export-gguf\n"
        f"python -m src.revive modelfile --gguf artifacts/gguf/unsloth.Q4_K_M.gguf "
        f"--language Cherokee\n"
        f"ollama create linguaforge-cherokee -f artifacts/Modelfile\n"
        f"```\n\n"
        "Once done, the community can install your custom model anywhere with a single command:\n"
        "```bash\nollama run linguaforge-cherokee\n```"
    )


LEARN_SESSIONS: dict[tuple[str, str], object] = {}


def build_demo() -> gr.Blocks:
    lang_choices = [
        (f"{l.english_name} ({l.native_name}) — {l.status}", code)
        for code, l in SUPPORTED_LANGUAGES.items()
    ]

    with gr.Blocks(title="LinguaForge", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER_MD)

        with gr.Tab("🎙️ Listen"):
            gr.Markdown(
                "Upload a recording of an elder telling a story. Gemma 4 will turn "
                "it into a deck of multimodal learning cards: vocabulary, grammar, "
                "and cultural notes."
            )
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(type="filepath", label="Elder's recording")
                    lang_in = gr.Dropdown(
                        choices=lang_choices,
                        value=CONFIG.default_language_code,
                        label="Language",
                    )
                    max_seg = gr.Slider(1, 30, value=8, step=1, label="Max segments")
                    listen_btn = gr.Button("Generate cards", variant="primary")
                with gr.Column():
                    listen_md = gr.Markdown()
                    listen_table = gr.Dataframe(
                        headers=["id", "type", "native", "english", "note"],
                        wrap=True,
                    )
            listen_btn.click(
                listen_tab,
                inputs=[audio_in, lang_in, max_seg],
                outputs=[listen_md, listen_table],
            )

        with gr.Tab("📚 Learn"):
            gr.Markdown(
                "Chat with your personal tutor. The tutor uses Gemma 4's native "
                "function calling to retrieve preserved knowledge from the local "
                "card store before each reply."
            )
            with gr.Row():
                lang_chat = gr.Dropdown(
                    choices=lang_choices,
                    value=CONFIG.default_language_code,
                    label="Target language",
                    scale=1,
                )
                learner_id = gr.Textbox(value="demo_learner", label="Learner ID", scale=1)
            chatbot = gr.Chatbot(height=420)
            msg = gr.Textbox(placeholder="e.g. Teach me a Cherokee greeting.", label="You")
            state = gr.State([])
            msg.submit(
                learn_tab_chat,
                inputs=[msg, state, lang_chat, learner_id],
                outputs=[chatbot, state],
            ).then(lambda: "", outputs=msg)

        with gr.Tab("🔁 Revive"):
            gr.Markdown(
                "Fine-tune a Gemma 4-4B with **Unsloth** on community-collected "
                "data, then ship it through **Ollama** for offline use anywhere."
            )
            corpus = gr.Textbox(value="data/corpus.jsonl", label="Corpus (.jsonl)")
            base_model = gr.Textbox(value="unsloth/gemma-4-E4B-it", label="Base model")
            revive_out = gr.Markdown()
            gr.Button("Show pipeline").click(
                revive_tab, inputs=[corpus, base_model], outputs=revive_out
            )

        gr.Markdown(
            "---\n"
            "Built with Gemma 4 · Unsloth · Ollama · Whisper · LlamaIndex · Gradio.\n\n"
            "[GitHub](https://github.com/) · [Kaggle Writeup](https://kaggle.com/) · "
            "[3-min video](https://youtube.com/)"
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", share=False)
