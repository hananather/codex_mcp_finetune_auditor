from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FeatureActivation(BaseModel):
    feature_idx: int
    avg_activation: float
    max_activation: float
    top_tokens: list[str] = Field(default_factory=list)


class TopFeaturesResult(BaseModel):
    model: str
    features: list[FeatureActivation]
    tokens: list[str] = Field(default_factory=list, description="Token strings aligned to the analyzed input.")


class CompareTopFeaturesResult(BaseModel):
    reference_model: str
    candidate_model: str
    reference_features: list[FeatureActivation]
    candidate_features: list[FeatureActivation]
    candidate_only: list[FeatureActivation]
    reference_only: list[FeatureActivation]
    common: list[FeatureActivation]


class FeatureDiff(BaseModel):
    feature_idx: int
    reference_activation: float
    candidate_activation: float
    diff: float
    direction: Literal["increased", "decreased"]


class DifferentialFeatureAnalysisResult(BaseModel):
    reference_model: str
    candidate_model: str
    top_diffs: list[FeatureDiff]


class FeatureActivationTrace(BaseModel):
    model: str
    feature_idx: int
    tokens: list[str]
    activations: list[float]


class FeatureDetails(BaseModel):
    feature_idx: int
    source: str
    explanation: Optional[str] = None
    density: Optional[float] = None
    n_examples: Optional[int] = None
    top_examples: list[dict[str, Any]] = Field(default_factory=list)
    top_pos_logits: list[dict[str, Any]] = Field(default_factory=list)
    top_neg_logits: list[dict[str, Any]] = Field(default_factory=list)
    url: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


class NeighborFeature(BaseModel):
    feature_idx: int
    cosine: float
    explanation: Optional[str] = None


class NearestNeighborsResult(BaseModel):
    feature_idx: int
    mode: str
    neighbors: list[NeighborFeature]
    self_details: Optional[FeatureDetails] = None
