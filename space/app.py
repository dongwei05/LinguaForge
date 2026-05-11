"""LinguaForge — Hugging Face Space demo.

Side-by-side comparison of the base Gemma 4 E4B vs. the LinguaForge LoRA on
endangered / low-resource translation, plus an "ask the tutor" chat panel.

Designed for HF Spaces ZeroGPU (free tier). The model is loaded once at module
import and the inference function is decorated with `@spaces.GPU` so each call
acquires a short GPU lease.

License: code Apache-2.0, demo adapter CC-BY-SA 4.0.
"""

from __future__ import annotations

import os
import textwrap
import time
from typing import Tuple

import gradio as gr
import torch

try:
    import spaces  # type: ignore[import-not-found]

    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

    class _SpacesShim:  # local-dev fallback (no ZeroGPU on developer machine)
        def GPU(self, *_args, **_kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    spaces = _SpacesShim()  # type: ignore[assignment]


ADAPTER_REPO = "zcgf111/linguaforge-gemma4-204lang-lora"

TUTOR_HEAD = (
    "You are LinguaForge, a multilingual tutor for endangered and low-resource "
    "languages. Stay accurate, concise, and faithful to the target script."
)

# Hand-picked 16 languages to keep the dropdown short; the adapter actually
# saw all 203 FLORES-200 + Cherokee from ChrEn during training.
SHOWCASE = [
    ("Cherokee (Iroquoian, North America)", "Cherokee (Iroquoian, North America)"),
    ("Maori (Polynesian, Pacific)", "Maori (Polynesian, Pacific)"),
    ("Hawaiian (Polynesian, Pacific)", "Hawaiian (Polynesian, Pacific)"),
    ("Welsh (Celtic, Europe)", "Welsh (Celtic, Europe)"),
    ("Irish Gaelic (Celtic, Europe)", "Irish Gaelic (Celtic, Europe)"),
    ("Tibetan (Sino-Tibetan, Asia)", "Tibetan (Sino-Tibetan, Asia)"),
    ("Mongolian (Mongolic, Asia)", "Mongolian (Mongolic, Asia)"),
    ("Ayacucho Quechua (Quechuan, South America)", "Ayacucho Quechua (Quechuan, South America)"),
    ("Aymara (Aymaran, South America)", "Aymara (Aymaran, South America)"),
    ("Yoruba (Niger-Congo, West Africa)", "Yoruba (Niger-Congo, West Africa)"),
    ("Igbo (Niger-Congo, West Africa)", "Igbo (Niger-Congo, West Africa)"),
    ("Tigrinya (Semitic, East Africa)", "Tigrinya (Semitic, East Africa)"),
    ("Twi/Akan (Niger-Congo, West Africa)", "Twi/Akan (Niger-Congo, West Africa)"),
    ("Sanskrit (Indo-Aryan, South Asia)", "Sanskrit (Indo-Aryan, South Asia)"),
    ("Bambara (Mande, West Africa)", "Bambara (Mande, West Africa)"),
    ("Esperanto (Constructed, Diaspora)", "Esperanto (Constructed, Diaspora)"),
]

EXAMPLES = [
    ["Hello, my name is Sarah. What is your name?", SHOWCASE[0][0]],
    ["The old man told stories about the river under the moonlight.", SHOWCASE[1][0]],
    ["Knowledge is the fruit of patience and humility.", SHOWCASE[3][0]],
    ["The mountain knows the names of every wind that crosses it.", SHOWCASE[5][0]],
    ["My grandmother teaches us how to weave with red and white wool.", SHOWCASE[7][0]],
]


def _load_model():
    """Load Gemma 4 E4B + LoRA via Unsloth in 4-bit."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print(f"Loading {ADAPTER_REPO} (Unsloth, 4-bit)...")
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_REPO,
        max_seq_length=2048,
        load_in_4bit=True,
        token=os.environ.get("HF_TOKEN"),
    )
    tok = get_chat_template(tok, chat_template="gemma")
    FastLanguageModel.for_inference(model)
    print(f"Model + adapter loaded in {time.time() - t0:.1f}s")
    return model, tok


# Lazy module-level load: HF Spaces import the module once, on a CPU node for
# ZeroGPU; the heavy state stays warm for subsequent GPU calls.
MODEL = None
TOK = None
TEXT_TOK = None


def _ensure_model():
    global MODEL, TOK, TEXT_TOK
    if MODEL is None:
        MODEL, TOK = _load_model()
        TEXT_TOK = getattr(TOK, "tokenizer", TOK)
    return MODEL, TOK, TEXT_TOK


@spaces.GPU(duration=90)
def _generate(prompt_msgs, use_lora: bool, max_new_tokens: int = 160) -> str:
    model, tok, text_tok = _ensure_model()
    text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    inputs = text_tok(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=text_tok.eos_token_id,
    )
    if use_lora:
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
    else:
        # Disable the adapter so we get the unmodified base Gemma 4 E4B.
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
        else:
            model.disable_adapter_layers()
            try:
                with torch.inference_mode():
                    out = model.generate(**inputs, **gen_kwargs)
            finally:
                model.enable_adapter_layers()
    gen = text_tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return gen.strip()


def translate_both(english_text: str, lang_label: str) -> Tuple[str, str]:
    english_text = (english_text or "").strip()
    if not english_text:
        return "(empty input)", "(empty input)"
    msgs = [
        {"role": "system", "content": TUTOR_HEAD},
        {"role": "user", "content": f"Translate this English sentence into {lang_label}:\n\n{english_text}"},
    ]
    base = _generate(msgs, use_lora=False)
    lora = _generate(msgs, use_lora=True)
    return base, lora


def tutor_chat(message: str, history, lang_label: str):
    msgs = [{"role": "system", "content": TUTOR_HEAD}]
    msgs.append(
        {
            "role": "system",
            "content": f"The learner is working in {lang_label}. Whenever you use a native phrase, also give the English meaning in parentheses.",
        }
    )
    for user_msg, bot_msg in history or []:
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": bot_msg})
    msgs.append({"role": "user", "content": message})
    reply = _generate(msgs, use_lora=True, max_new_tokens=320)
    return reply


INTRO = textwrap.dedent(
    """
    # 🌍 LinguaForge — Gemma 4 E4B LoRA across **204 languages**

    A 169.7 MB LoRA adapter on top of
    [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it),
    trained on **all 203 non-English languages in FLORES-200** plus
    Cherokee depth from **ChrEn** — together covering every populated
    continent.

    - **Reproducer kernel:** [`dongwei666/linguaforge-auto`](https://www.kaggle.com/code/dongwei666/linguaforge-auto)
    - **Eval kernel (real BLEU/chrF):** [`dongwei666/linguaforge-eval`](https://www.kaggle.com/code/dongwei666/linguaforge-eval)
    - **Adapter on HF:** [`zcgf111/linguaforge-gemma4-204lang-lora`](https://huggingface.co/zcgf111/linguaforge-gemma4-204lang-lora)

    *Pick a language and an English sentence — the same prompt is sent to the
    base Gemma 4 and to LinguaForge, and the answers appear side by side.
    First call warms up the model (~30-60s on a ZeroGPU lease).*
    """
)


HEADLINE_NUMBERS = textwrap.dedent(
    """
    ### Held-out FLORES-200 / ChrEn devtest (50 sentences/language)

    |Language | base BLEU | +LoRA BLEU | base chrF | +LoRA chrF |
    |---|---:|---:|---:|---:|
    | Cherokee | 0.04 | **0.45** | 2.30 | **7.87** (3.4× ↑) |
    | Tibetan | 0.12 | **0.21** | 19.14 | **27.05** (+7.91) |
    | Welsh | 3.90 | **6.13** (+2.23) | 31.11 | 31.21 |
    | Quechua | 1.02 | **1.93** | 19.94 | **22.49** |
    | Maori | 3.64 | **4.16** | 28.48 | 27.58 |
    | Yoruba ⚠ | 2.54 | 1.12 | 21.65 | 11.10 |
    """
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="LinguaForge — Gemma 4 multilingual LoRA",
        theme=gr.themes.Soft(primary_hue="amber"),
    ) as demo:
        gr.Markdown(INTRO)

        with gr.Tab("Translate (base vs +LoRA)"):
            with gr.Row():
                with gr.Column(scale=1):
                    en_in = gr.Textbox(
                        label="English sentence",
                        placeholder="e.g. The old man told stories about the river.",
                        lines=3,
                    )
                    lang_dd = gr.Dropdown(
                        choices=SHOWCASE,
                        value=SHOWCASE[0][0],
                        label="Target language",
                    )
                    go_btn = gr.Button("Translate side by side", variant="primary")
                with gr.Column(scale=1):
                    base_out = gr.Textbox(label="Base Gemma 4 E4B (no adapter)", lines=4)
                    lora_out = gr.Textbox(label="+ LinguaForge LoRA", lines=4)
            gr.Examples(EXAMPLES, [en_in, lang_dd], cache_examples=False)
            go_btn.click(translate_both, [en_in, lang_dd], [base_out, lora_out])

        with gr.Tab("Tutor chat"):
            gr.Markdown(
                "Chat with LinguaForge as a multilingual tutor. The system "
                "prompt asks for English glosses next to any native phrases."
            )
            with gr.Row():
                chat_lang = gr.Dropdown(
                    choices=SHOWCASE,
                    value=SHOWCASE[0][0],
                    label="Target language",
                )
            chatbot = gr.Chatbot(height=400, label="LinguaForge")
            msg_in = gr.Textbox(
                placeholder="e.g. Teach me three Cherokee greetings with cultural context.",
                label="Your message",
            )
            clear = gr.Button("Clear")

            def respond(message, history, lang_label):
                reply = tutor_chat(message, history, lang_label)
                history = (history or []) + [(message, reply)]
                return "", history

            msg_in.submit(respond, [msg_in, chatbot, chat_lang], [msg_in, chatbot])
            clear.click(lambda: None, None, chatbot)

        with gr.Tab("Numbers"):
            gr.Markdown(HEADLINE_NUMBERS)
            gr.Markdown(
                "Decoding is greedy (`do_sample=False`) for reproducibility. "
                "Yoruba regression is documented in the writeup — likely fixable "
                "by routing low-quality languages to per-community LoRAs."
            )

        gr.Markdown(
            "---\n"
            "Adapter: 42.4M trainable params (0.53 % of 8B). Training: 8,370 steps, "
            "~5h09 on Kaggle T4. Base model © Google DeepMind. FLORES-200 © Meta NLLB."
        )

    return demo


if __name__ == "__main__":
    build_demo().queue(default_concurrency_limit=1, max_size=8).launch()
