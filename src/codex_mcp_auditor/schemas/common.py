from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="Chat role")
    content: str = Field(description="Message content")


class PromptSpec(BaseModel):
    """
    A prompt can be provided as:
      - messages[] (preferred for chat models)
      - prompt (single user string)
    """
    messages: Optional[list[Message]] = Field(default=None, description="Chat messages")
    prompt: Optional[str] = Field(default=None, description="Single-turn user prompt")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt to prepend")
    use_chat_template: bool = Field(default=True, description="Whether to apply the tokenizer chat template if available.")
    add_generation_prompt: bool = Field(default=True, description="Whether to add a generation marker when using chat templates.")

    @classmethod
    def from_user(cls, user: str) -> "PromptSpec":
        return cls(prompt=user, use_chat_template=False)


class GenerationParams(BaseModel):
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    do_sample: bool = Field(default=False)
    seed: Optional[int] = Field(default=None, description="Optional generation seed for deterministic sampling.")


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    profile: str
    artifacts_dir: str
    resolved_config_path: str


class RunInfo(BaseModel):
    run_id: str
    run_name: str
    started_at: datetime
    run_dir: str


class HealthInfo(BaseModel):
    ok: bool
    server_time: datetime
    backend: str
    sessions: int
    notes: list[str] = Field(default_factory=list)


class ModelResponse(BaseModel):
    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class QueryModelsResult(BaseModel):
    responses: list[ModelResponse]
    prompt_used: PromptSpec


class TrainingSample(BaseModel):
    line_number: int
    raw: str
    parsed: Optional[dict[str, Any]] = None


class TrainingGrepMatch(BaseModel):
    line_number: int
    match: str
    raw: str
