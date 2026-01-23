from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Nearest-neighbor tools require torch. Install optional deps with: pip install -e '.[hf]'"
        ) from e
    return torch


@dataclass
class DecoderCosineIndex:
    """
    Simple cosine-sim search over SAE decoder directions.

    For large SAEs, this can be expensive; we keep implementation straightforward and
    rely on (1) caching norms and (2) chunked scoring.
    """
    decoder: Any  # torch.Tensor [d_sae, d_model]
    decoder_norms: Any  # torch.Tensor [d_sae]
    eps: float = 1e-8

    @classmethod
    def from_decoder(cls, decoder: Any) -> "DecoderCosineIndex":
        torch = _require_torch()
        # decoder: [d_sae, d_model]
        norms = torch.linalg.norm(decoder.float(), dim=1) + cls.eps
        return cls(decoder=decoder, decoder_norms=norms)

    def topk(self, feature_idx: int, k: int = 50, *, exclude_self: bool = True, min_cos: Optional[float] = None, chunk_size: int = 8192) -> list[tuple[int, float]]:
        torch = _require_torch()
        idx = int(feature_idx)
        q = self.decoder[idx].float()
        qn = torch.linalg.norm(q) + self.eps

        # Score in chunks to avoid huge intermediate allocations.
        best_scores = None
        best_indices = None

        d_sae = int(self.decoder.shape[0])
        for start in range(0, d_sae, int(chunk_size)):
            end = min(d_sae, start + int(chunk_size))
            chunk = self.decoder[start:end].float()
            dots = (chunk @ q) / (self.decoder_norms[start:end] * qn)
            if min_cos is not None:
                dots = torch.where(dots >= float(min_cos), dots, torch.tensor(-1e9, device=dots.device))
            # keep more than k to be safe
            kk = min(int(k) + 5, int(dots.numel()))
            vals, inds = torch.topk(dots, kk)
            inds = inds + start

            if best_scores is None:
                best_scores = vals
                best_indices = inds
            else:
                best_scores = torch.cat([best_scores, vals])
                best_indices = torch.cat([best_indices, inds])

        # Global top-k
        kk = min(int(k) + 5, int(best_scores.numel()))
        vals, order = torch.topk(best_scores, kk)
        inds = best_indices[order]

        out: list[tuple[int, float]] = []
        for i, v in zip(inds.tolist(), vals.tolist()):
            if exclude_self and int(i) == idx:
                continue
            if min_cos is not None and float(v) < float(min_cos):
                break
            out.append((int(i), float(v)))
            if len(out) >= int(k):
                break
        return out
