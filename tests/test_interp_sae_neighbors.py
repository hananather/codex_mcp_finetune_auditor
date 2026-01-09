from __future__ import annotations

import pytest


def test_jump_relu_sae_threshold_masks():
    """JumpReLUSAE.encode should zero-out activations below threshold and ReLU negatives."""
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from codex_mcp_auditor.interp.sae import JumpReLUSAE

    sae = JumpReLUSAE(torch, nn, d_model=2, d_sae=2)
    with torch.no_grad():
        sae.module.w_enc[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.module.b_enc.zero_()
        sae.module.threshold[:] = torch.tensor([0.5, 0.0])

    x = torch.tensor([[0.4, -1.0]])
    acts = sae.encode(x)

    assert acts.shape == (1, 2)
    assert acts[0, 0].item() == 0.0  # below threshold
    assert acts[0, 1].item() == 0.0  # negative input relu


def test_jump_relu_sae_allows_positive_above_threshold():
    """JumpReLUSAE.encode should pass through positive activations above threshold."""
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    from codex_mcp_auditor.interp.sae import JumpReLUSAE

    sae = JumpReLUSAE(torch, nn, d_model=2, d_sae=2)
    with torch.no_grad():
        sae.module.w_enc[:] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sae.module.b_enc.zero_()
        sae.module.threshold[:] = torch.tensor([0.1, 0.1])

    x = torch.tensor([[0.6, 0.2]])
    acts = sae.encode(x)

    assert acts[0, 0].item() > 0.0
    assert acts[0, 1].item() > 0.0


def test_decoder_cosine_index_excludes_self():
    """DecoderCosineIndex.topk should exclude the query index when exclude_self is True."""
    torch = pytest.importorskip("torch")

    from codex_mcp_auditor.interp.neighbors import DecoderCosineIndex

    decoder = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    index = DecoderCosineIndex.from_decoder(decoder)
    neighbors = index.topk(0, k=2, exclude_self=True)

    assert len(neighbors) == 2
    assert all(idx != 0 for idx, _ in neighbors)
    assert neighbors[0][0] == 2
