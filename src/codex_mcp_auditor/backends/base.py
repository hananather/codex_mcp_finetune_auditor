from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..schemas.common import GenerationParams, PromptSpec


@dataclass
class EncodedPrompt:
    input_ids: Any  # backend-specific tensor/array type
    tokens: list[str]
    text: str


class ModelAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, prompt: PromptSpec) -> EncodedPrompt:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: PromptSpec, gen: GenerationParams) -> tuple[str, int, int]:
        """
        Returns:
          text, prompt_tokens, completion_tokens
        """
        raise NotImplementedError

    @abstractmethod
    def residual_activations(self, encoded: EncodedPrompt, layer: int, module_path_template: str, output_selector: str) -> Any:
        """Return resid_post-like activations as backend tensor/array [batch, seq, d_model]."""
        raise NotImplementedError


class Backend(ABC):
    name: str

    @abstractmethod
    def load_model(self, role: str, id_or_path: str, **kwargs: Any) -> ModelAdapter:
        raise NotImplementedError
