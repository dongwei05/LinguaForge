"""Gemma 4 loader with graceful fallback to Gemma 3.

Provides a unified `chat()` interface so the rest of the codebase does not
care which exact variant is loaded. Native function-calling on Gemma 4 is
exposed via the `tools` argument; on the fallback path we emulate it with a
JSON-schema prompt convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from loguru import logger

from .config import CONFIG, ModelConfig


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None  # for tool messages
    tool_call_id: str | None = None


@dataclass
class ToolSpec:
    """A tool the model can call. Mirrors OpenAI/Gemma 4 schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class GemmaClient:
    """Lazy-loading Gemma client. Tries Gemma 4, falls back to Gemma 3."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        self.cfg = cfg or CONFIG.model
        self._tokenizer = None
        self._model = None
        self._loaded_model_id: str | None = None
        self._native_function_calling: bool = False

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            self._try_load(self.cfg.primary_model_id, native_fc=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Primary model {} failed to load: {}. Falling back to {}.",
                self.cfg.primary_model_id,
                exc,
                self.cfg.fallback_model_id,
            )
            self._try_load(self.cfg.fallback_model_id, native_fc=False)

    def _try_load(self, model_id: str, *, native_fc: bool) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if self.cfg.use_4bit:
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.info("bitsandbytes unavailable; loading in bf16.")

        if self.cfg.device != "auto":
            load_kwargs["device_map"] = self.cfg.device
        else:
            load_kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=self.cfg.revision, token=CONFIG.hf_token
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=self.cfg.revision, token=CONFIG.hf_token, **load_kwargs
        )
        self._loaded_model_id = model_id
        self._native_function_calling = native_fc
        logger.success(
            "Loaded {} (native_fc={}, 4bit={})",
            model_id,
            native_fc,
            self.cfg.use_4bit,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        max_tool_iters: int = 4,
        **gen_kwargs: Any,
    ) -> str:
        """Run a chat completion, with optional tool-calling loop."""
        self._load()
        history: list[ChatMessage] = list(messages)
        tools = list(tools or [])

        for _ in range(max_tool_iters + 1):
            raw = self._generate(history, tools, **gen_kwargs)
            tool_call = self._maybe_extract_tool_call(raw)
            if tool_call is None:
                return raw

            tool_name, tool_args = tool_call
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool is None:
                logger.warning("Model called unknown tool: {}", tool_name)
                history.append(
                    ChatMessage(
                        role="tool",
                        name=tool_name,
                        content=json.dumps({"error": "unknown_tool"}),
                    )
                )
                continue

            try:
                result = tool.handler(tool_args)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            history.append(ChatMessage(role="assistant", content=raw))
            history.append(
                ChatMessage(
                    role="tool",
                    name=tool_name,
                    content=json.dumps(result, ensure_ascii=False, default=str),
                )
            )
        return raw

    def _generate(
        self,
        history: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        **gen_kwargs: Any,
    ) -> str:
        prompt = self._build_prompt(history, tools)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_args = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.temperature > 0.0,
            **gen_kwargs,
        }
        out = self._model.generate(**inputs, **gen_args)
        decoded = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        return decoded.strip()

    def _build_prompt(
        self,
        history: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> str:
        if self._native_function_calling and tools:
            chat_template_messages = [
                {"role": m.role, "content": m.content} for m in history
            ]
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            return self._tokenizer.apply_chat_template(
                chat_template_messages,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=False,
            )

        sys_addendum = ""
        if tools:
            sys_addendum = (
                "\n\nYou have access to the following tools. To call a tool, "
                "respond with ONLY a JSON block in this exact form:\n"
                '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>\n\n'
                "Tools:\n"
                + "\n".join(
                    f"- {t.name}: {t.description}\n  params: {json.dumps(t.parameters)}"
                    for t in tools
                )
            )

        chat_template_messages = []
        prepended_system = False
        for m in history:
            content = m.content
            if m.role == "system" and not prepended_system and sys_addendum:
                content = content + sys_addendum
                prepended_system = True
            chat_template_messages.append({"role": m.role, "content": content})

        if sys_addendum and not prepended_system:
            chat_template_messages.insert(
                0, {"role": "system", "content": "You are a helpful assistant." + sys_addendum}
            )

        return self._tokenizer.apply_chat_template(
            chat_template_messages, add_generation_prompt=True, tokenize=False
        )

    @staticmethod
    def _maybe_extract_tool_call(raw: str) -> tuple[str, dict[str, Any]] | None:
        marker = "<tool_call>"
        end = "</tool_call>"
        if marker in raw and end in raw:
            payload = raw.split(marker, 1)[1].split(end, 1)[0].strip()
            try:
                obj = json.loads(payload)
                return obj["name"], obj.get("arguments", {})
            except (json.JSONDecodeError, KeyError):
                return None
        return None


_singleton: GemmaClient | None = None


def get_client() -> GemmaClient:
    global _singleton  # noqa: PLW0603
    if _singleton is None:
        _singleton = GemmaClient()
    return _singleton
