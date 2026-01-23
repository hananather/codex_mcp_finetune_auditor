from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectConfig(BaseModel):
    name: str = Field(default="ft-audit", description="Project name used in artifacts.")
    results_dir: str = Field(default="./runs", description="Where to write artifacts.")
    seed: int = Field(default=0, description="Base random seed for repeatability.")
    score_method: Literal["abs_diff_topk", "abs_diff_topk_drift_corrected"] = Field(
        default="abs_diff_topk",
        description=(
            "Suite scoring method. `abs_diff_topk` scores prompts by summing top-k absolute feature activation "
            "differences. `abs_diff_topk_drift_corrected` selects top-k by absolute diff after subtracting the mean "
            "diff vector over prompts (reduces prompt-invariant drift), while still reporting raw diffs."
        ),
    )


class BackendConfig(BaseModel):
    type: Literal["mock", "hf"] = Field(
        default="mock",
        description="Backend used for model execution: mock (no HF) or hf (transformers).",
    )


class ModelConfig(BaseModel):
    id_or_path: str = Field(
        min_length=1,
        description="HuggingFace model id (e.g., google/gemma-3-1b-it) or local path.",
    )
    revision: Optional[str] = Field(default=None, description="Optional HF revision.")
    trust_remote_code: bool = Field(default=False, description="Whether to trust remote code.")

    # HF-specific knobs (ignored by mock backend)
    device_map: Optional[str] = Field(
        default="auto", description='Transformers device_map, e.g. "auto" or "cpu".'
    )
    dtype: Optional[str] = Field(
        default="auto",
        description='Torch dtype: "auto", "float16", "bfloat16", "float32".',
    )
    attn_implementation: Optional[str] = Field(
        default=None,
        description='Optional attention implementation (backend dependent), e.g. "flash_attention_2".',
    )

    @field_validator("id_or_path")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("models.*.id_or_path must be a non-empty string")
        return v


class ModelsConfig(BaseModel):
    base: ModelConfig
    benign: ModelConfig
    adversarial: ModelConfig


class DatasetConfig(BaseModel):
    training_jsonl: Optional[str] = Field(
        default=None,
        description="Optional path to a JSONL fine-tuning dataset for triage tools.",
    )


class SAEWeightsConfig(BaseModel):
    source: Literal["local", "hf_hub"] = Field(
        default="local",
        description="Where SAE weights come from: local filesystem or HuggingFace Hub.",
    )
    path: Optional[str] = Field(
        default=None, description="Local safetensors path (if source=local)."
    )
    repo_id: Optional[str] = Field(
        default=None, description="HF Hub repo_id (if source=hf_hub)."
    )
    filename: Optional[str] = Field(
        default=None, description="HF Hub filename (if source=hf_hub)."
    )

    @model_validator(mode="after")
    def _validate_source(self) -> "SAEWeightsConfig":
        if self.source == "local":
            if not self.path:
                raise ValueError("interp.sae.weights.path is required when source=local")
        else:
            if not self.repo_id or not self.filename:
                raise ValueError(
                    "interp.sae.weights.repo_id and filename are required when source=hf_hub"
                )
        return self


class SAEConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable SAE-based interpretability tools.")
    layer: int = Field(
        default=0, description="Which transformer layer to hook for resid_post activations."
    )
    module_path_template: str = Field(
        default="model.model.layers.{layer}",
        description=(
            "Python attribute path template to the layer module to hook. "
            "Uses `{layer}` substitution."
        ),
    )
    output_selector: Literal["first", "raw"] = Field(
        default="first",
        description=(
            "How to interpret the hooked module output. "
            "`first` means: if output is tuple/list, take output[0]."
        ),
    )
    weights: Optional[SAEWeightsConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _drop_weights_when_disabled(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        enabled = data.get("enabled", False)
        enabled_val = enabled
        if isinstance(enabled, str):
            enabled_val = enabled.strip().lower() not in ("", "0", "false", "no", "off", "null", "none")
        if not bool(enabled_val):
            data = dict(data)
            data.pop("weights", None)
        return data

    @model_validator(mode="after")
    def _validate_enabled(self) -> "SAEConfig":
        if self.enabled and self.weights is None:
            raise ValueError("interp.sae.weights must be set when interp.sae.enabled=true")
        return self


class NeuronpediaConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable Neuronpedia metadata/neighbors integration.")
    base_url: str = Field(
        default="http://127.0.0.1:3000", description="Neuronpedia base URL (local or remote)."
    )
    model_id: str = Field(default="", description="Neuronpedia model id (required if enabled).")
    source: str = Field(default="", description="Neuronpedia feature source name (required if enabled).")

    inference_base_url: str = Field(
        default="http://localhost:5002/v1",
        description="Neuronpedia inference server base URL (optional).",
    )
    inference_secret_env_var: Optional[str] = Field(
        default=None, description="Env var containing inference secret token (optional)."
    )

    @model_validator(mode="after")
    def _validate_enabled(self) -> "NeuronpediaConfig":
        if self.enabled:
            if not self.model_id or not self.source:
                raise ValueError(
                    "interp.neuronpedia.model_id and interp.neuronpedia.source are required when enabled=true"
                )
        return self


class InterpConfig(BaseModel):
    sae: SAEConfig = Field(default_factory=SAEConfig)
    neuronpedia: NeuronpediaConfig = Field(default_factory=NeuronpediaConfig)


class AuditConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    models: ModelsConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    interp: InterpConfig = Field(default_factory=InterpConfig)


class ToolProfile(BaseModel):
    """Server-side tool allowlist. Codex can also enforce allow/deny lists client-side."""
    name: str
    enabled_tools: list[str]


DEFAULT_PROFILES: dict[str, ToolProfile] = {
    "behavior_only": ToolProfile(
        name="behavior_only",
        enabled_tools=[
            "health",
            "create_audit_session",
            "begin_run",
            "close_audit_session",
            "get_training_data_length",
            "view_training_data_sample",
            "grep_training_data",
            "query_models",
            "run_prompt_suite",
        ],
    ),
    "full": ToolProfile(
        name="full",
        enabled_tools=[
            "health",
            "create_audit_session",
            "begin_run",
            "close_audit_session",
            "get_training_data_length",
            "view_training_data_sample",
            "grep_training_data",
            "query_models",
            "run_prompt_suite",
            "get_top_features",
            "compare_top_features",
            "differential_feature_analysis",
            "specific_feature_activations",
            "get_feature_details",
            "nearest_explained_neighbors",
            "score_candidate_suite",
            "write_audit_report",
        ],
    ),
}
