from __future__ import annotations

import pytest

from codex_mcp_auditor.config import (
    AuditConfig,
    DEFAULT_PROFILES,
    NeuronpediaConfig,
    SAEConfig,
    SAEWeightsConfig,
)


def _minimal_audit_config_dict() -> dict:
    return {
        "models": {
            "base": {"id_or_path": "base"},
            "benign": {"id_or_path": "benign"},
            "adversarial": {"id_or_path": "adv"},
        }
    }


def test_sae_weights_local_requires_path():
    """SAEWeightsConfig should reject local weights without a path to prevent silent misconfigurations."""
    with pytest.raises(ValueError, match="weights.path"):
        SAEWeightsConfig(source="local", path=None)


def test_sae_weights_hf_hub_requires_repo_and_filename():
    """SAEWeightsConfig should enforce repo_id + filename when weights come from HuggingFace Hub."""
    with pytest.raises(ValueError, match="repo_id"):
        SAEWeightsConfig(source="hf_hub", repo_id=None, filename=None)


def test_sae_config_enabled_requires_weights():
    """SAEConfig should refuse enabled=True unless weights are configured, ensuring SAE tools can load."""
    with pytest.raises(ValueError, match="interp.sae.weights"):
        SAEConfig(enabled=True, weights=None)


def test_sae_config_ignores_weights_when_disabled():
    """SAEConfig should drop weights when enabled is false to avoid template validation failures."""
    cfg = SAEConfig.model_validate(
        {
            "enabled": False,
            "weights": {"source": "local", "path": "/tmp/weights.safetensors"},
        }
    )
    assert cfg.enabled is False
    assert cfg.weights is None


def test_neuronpedia_enabled_requires_model_and_source():
    """NeuronpediaConfig should require model_id + source when enabled to form valid API URLs."""
    with pytest.raises(ValueError, match="model_id"):
        NeuronpediaConfig(enabled=True, model_id="", source="")


def test_audit_config_minimal_defaults():
    """AuditConfig should accept the minimal model-only config and apply defaults for other sections."""
    cfg = AuditConfig.model_validate(_minimal_audit_config_dict())
    assert cfg.backend.type == "mock"
    assert cfg.project.name == "ft-audit"


def test_behavior_only_profile_excludes_sae_tools():
    """behavior_only profile should not expose SAE/report tools."""
    enabled = set(DEFAULT_PROFILES["behavior_only"].enabled_tools)
    assert "write_audit_report" not in enabled
