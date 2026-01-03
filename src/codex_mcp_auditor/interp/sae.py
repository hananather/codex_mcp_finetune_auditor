from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import SAEConfig, SAEWeightsConfig


def _require_torch_sae() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SAE tools require torch. Install optional deps with: pip install -e '.[hf]'"
        ) from e
    return torch, nn


class JumpReLUSAE:  # intentionally minimal wrapper
    def __init__(self, torch: Any, nn: Any, d_model: int, d_sae: int):
        self.torch = torch
        self.nn = nn
        self.module = nn.Module()
        # Register parameters on an inner module to support .to(device)
        self.module.w_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.module.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.module.threshold = nn.Parameter(torch.zeros(d_sae))
        self.module.w_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.module.b_dec = nn.Parameter(torch.zeros(d_model))

    def to(self, device: Any) -> "JumpReLUSAE":
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    @property
    def device(self):
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

    def encode(self, x: Any) -> Any:
        # x: [..., d_model]
        torch = self.torch
        pre = x @ self.module.w_enc + self.module.b_enc
        mask = pre > self.module.threshold
        return mask * torch.relu(pre)

    def decoder_vectors(self) -> Any:
        # Returns [d_sae, d_model]
        return self.module.w_dec.detach()


def load_sae(sae_cfg: SAEConfig, device: Optional[str] = None) -> JumpReLUSAE:
    """
    Load JumpReLU SAE weights from a safetensors file (local or HF hub).
    """
    torch, nn = _require_torch_sae()
    weights: SAEWeightsConfig = sae_cfg.weights  # type: ignore[assignment]

    # Lazy imports so base install stays light.
    try:
        from safetensors.torch import load_file  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Loading SAE weights requires safetensors. Install optional deps with: pip install -e '.[hf]'"
        ) from e

    path = None
    if weights.source == "local":
        path = weights.path
    else:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "HF Hub SAE weights require huggingface_hub. Install optional deps with: pip install -e '.[hf]'"
            ) from e
        path = hf_hub_download(repo_id=weights.repo_id, filename=weights.filename)  # type: ignore[arg-type]

    params = load_file(path)  # type: ignore[arg-type]
    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(torch, nn, int(d_model), int(d_sae))
    sae.load_state_dict(params)

    dev = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    sae.to(dev)
    return sae
