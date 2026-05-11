"""Revive module: Unsloth fine-tuning of Gemma 4 on community-collected data.

This is the "Special Technology Track" pillar.

Workflow:

  community_corpus.jsonl  →  Unsloth LoRA fine-tune of Gemma 4-4B
                            →  merged GGUF
                            →  one-line `ollama create` for offline distribution

The script is parameterized so a community can re-run it on their own laptop
(or a free Kaggle T4) without any cloud cost.

Reference:
  https://github.com/unslothai/unsloth
  https://docs.unsloth.ai/get-started/fine-tuning-guide
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


def fine_tune(
    model_id: str = "unsloth/gemma-4-E4B-it",
    dataset_path: str = "data/corpus.jsonl",
    output_dir: str = "artifacts/lora_adapter",
    max_seq_length: int = 2048,
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
) -> str:
    """Run Unsloth LoRA fine-tuning. Returns the adapter directory path."""
    try:
        from unsloth import FastLanguageModel  # type: ignore  # noqa: PLC0415
        from unsloth.chat_templates import get_chat_template  # type: ignore  # noqa: PLC0415
        from datasets import load_dataset  # noqa: PLC0415
        from trl import SFTConfig, SFTTrainer  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth/trl/datasets not installed. See requirements.txt."
        ) from exc

    logger.info("Loading {} via Unsloth (4-bit)…", model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    logger.info("Loading dataset from {}", dataset_path)
    ds = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in convos
        ]
        return {"text": texts}

    ds = ds.map(formatting_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            logging_steps=5,
            output_dir=output_dir,
            save_strategy="epoch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            bf16=True,
            report_to="none",
        ),
    )

    logger.info("Starting training…")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.success("Adapter saved → {}", output_dir)
    return output_dir


def export_to_gguf(
    adapter_dir: str = "artifacts/lora_adapter",
    output_dir: str = "artifacts/gguf",
    quantization: str = "q4_k_m",
) -> str:
    try:
        from unsloth import FastLanguageModel  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Unsloth not installed.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(adapter_dir, load_in_4bit=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quantization)
    logger.success("GGUF exported → {}", out)
    return str(out)


def write_modelfile(gguf_path: str, language: str, out_path: str = "artifacts/Modelfile") -> str:
    """Generate an Ollama Modelfile for one-command community distribution."""
    content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER stop "<end_of_turn>"

SYSTEM \"\"\"You are LinguaForge, a patient, culturally-aware tutor for the {language} language.
Always ground replies in preserved oral knowledge. Encourage the learner.\"\"\"
"""
    Path(out_path).write_text(content, encoding="utf-8")
    logger.success("Modelfile written → {}", out_path)
    return out_path


def make_dummy_corpus(path: str = "data/corpus.jsonl", n_samples: int = 32) -> str:
    """Create a tiny placeholder corpus so the pipeline can be smoke-tested."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "messages": [
                {"role": "user", "content": f"Teach me how to greet someone in Cherokee. (Sample {i})"},
                {"role": "assistant", "content": "ᎣᏏᏲ (osiyo) means 'hello' — literally 'it is good'."},
            ]
        }
        for i in range(n_samples)
    ]
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Wrote {} dummy samples → {}", n_samples, path)
    return path


def cli() -> None:
    parser = argparse.ArgumentParser(description="LinguaForge Revive: Unsloth + Ollama pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dummy = sub.add_parser("dummy-corpus")
    p_dummy.add_argument("--path", default="data/corpus.jsonl")
    p_dummy.add_argument("--n", type=int, default=32)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model-id", default="unsloth/gemma-4-E4B-it")
    p_train.add_argument("--dataset", default="data/corpus.jsonl")
    p_train.add_argument("--output", default="artifacts/lora_adapter")
    p_train.add_argument("--epochs", type=int, default=2)

    p_export = sub.add_parser("export-gguf")
    p_export.add_argument("--adapter", default="artifacts/lora_adapter")
    p_export.add_argument("--output", default="artifacts/gguf")
    p_export.add_argument("--quant", default="q4_k_m")

    p_modelfile = sub.add_parser("modelfile")
    p_modelfile.add_argument("--gguf", required=True)
    p_modelfile.add_argument("--language", default="Cherokee")
    p_modelfile.add_argument("--output", default="artifacts/Modelfile")

    args = parser.parse_args()
    if args.cmd == "dummy-corpus":
        make_dummy_corpus(args.path, args.n)
    elif args.cmd == "train":
        fine_tune(args.model_id, args.dataset, args.output, epochs=args.epochs)
    elif args.cmd == "export-gguf":
        export_to_gguf(args.adapter, args.output, args.quant)
    elif args.cmd == "modelfile":
        write_modelfile(args.gguf, args.language, args.output)


if __name__ == "__main__":
    cli()
