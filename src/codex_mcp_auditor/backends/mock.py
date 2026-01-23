from __future__ import annotations

import hashlib
import random
from typing import Any

from .base import Backend, EncodedPrompt, ModelAdapter
from ..schemas.common import GenerationParams, PromptSpec


def _stable_int(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _prompt_to_text(prompt: PromptSpec) -> str:
    if prompt.messages:
        return "\n".join([f"{m.role}: {m.content}" for m in prompt.messages])
    return (prompt.system_prompt + "\n" if prompt.system_prompt else "") + (prompt.prompt or "")


class MockModelAdapter(ModelAdapter):
    def __init__(self, model_name: str, d_model: int = 32):
        super().__init__(model_name)
        self.d_model = d_model

    def encode(self, prompt: PromptSpec) -> EncodedPrompt:
        text = _prompt_to_text(prompt).strip()
        tokens = text.split() if text else ["<empty>"]
        return EncodedPrompt(input_ids=tokens, attention_mask=None, tokens=tokens, text=text)

    def generate(self, prompt: PromptSpec, gen: GenerationParams) -> tuple[str, int, int]:
        text = _prompt_to_text(prompt).strip()
        seed = _stable_int(f"{self.model_name}|{text}|{gen.max_new_tokens}|{gen.temperature}|{gen.top_p}|{gen.do_sample}|{gen.seed}")
        rng = random.Random(seed)
        # A deterministic pseudo-generation:
        words = ["mock", "response", "for", "audit", "tooling", "session", "candidate", "reference", "evidence", "report"]
        out_len = min(gen.max_new_tokens // 4, 40)
        out = " ".join(rng.choice(words) for _ in range(max(5, out_len)))
        return f"[{self.model_name}] {out}", len(text.split()), len(out.split())

    def residual_activations(self, encoded: EncodedPrompt, layer: int, module_path_template: str, output_selector: str) -> Any:
        # Produce deterministic fake activations: [1, seq, d_model]
        seed = _stable_int(f"{self.model_name}|{encoded.text}|layer={layer}|{module_path_template}|{output_selector}")
        rng = random.Random(seed)
        seq = max(1, len(encoded.tokens))
        acts = [[[rng.random() for _ in range(self.d_model)] for _ in range(seq)]]
        return acts


class MockBackend(Backend):
    name = "mock"

    def load_model(self, role: str, id_or_path: str, **kwargs: Any) -> ModelAdapter:
        # id_or_path is ignored; kept for interface compatibility.
        d_model = int(kwargs.get("d_model", 32))
        return MockModelAdapter(model_name=role, d_model=d_model)
