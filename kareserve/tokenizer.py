# SPDX-License-Identifier: Apache-2.0
"""In-process request tokenization for routing decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class LocalRequestTokenizer:
    def __init__(
        self,
        model_path: str,
        *,
        max_model_len: int = 2048,
        revision: str | None = None,
        trust_remote_code: bool = False,
        chat_template_path: str | None = None,
        allow_request_chat_template: bool = False,
    ) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.max_model_len = max(1, max_model_len)
        self.chat_template = None
        if chat_template_path:
            self.chat_template = Path(chat_template_path).read_text(encoding="utf-8")
        self.allow_request_chat_template = allow_request_chat_template

    def encode_request(self, body: dict[str, Any]) -> list[int]:
        if "messages" in body:
            return self._encode_chat(body)
        if "prompt" in body:
            return self._encode_completion(body)
        raise ValueError("Request must contain messages or prompt")

    def decode_tokens(self, tokens: list[int]) -> str:
        return str(self.tokenizer.decode(tokens))

    def token_strings(self, tokens: list[int]) -> list[str]:
        values = self.tokenizer.convert_ids_to_tokens(tokens)
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values]

    def _encode_completion(self, body: dict[str, Any]) -> list[int]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("Completion prompt must be a string")
        tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=bool(body.get("add_special_tokens", True)),
        )
        return [int(token) for token in tokens]

    def _encode_chat(self, body: dict[str, Any]) -> list[int]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise TypeError("Chat messages must be a list")
        request_template = body.get("chat_template")
        if request_template is not None and not self.allow_request_chat_template:
            raise ValueError("Request-specific chat_template is disabled")
        chat_template = request_template or self.chat_template
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs.pop("add_generation_prompt", None)
        kwargs.pop("continue_final_message", None)
        tools = body.get("tools")
        if tools is not None:
            kwargs["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=bool(body.get("add_generation_prompt", True)),
            continue_final_message=bool(body.get("continue_final_message", False)),
            tokenize=False,
            **kwargs,
        )
        tokens = self.tokenizer.encode(
            rendered,
            add_special_tokens=bool(body.get("add_special_tokens", False)),
        )
        return [int(token) for token in tokens]
