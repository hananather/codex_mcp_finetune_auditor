from __future__ import annotations

from codex_mcp_auditor.backends.mock import MockModelAdapter
from codex_mcp_auditor.schemas.common import GenerationParams, PromptSpec


def test_mock_encode_splits_tokens():
    """MockModelAdapter.encode should split prompt text into tokens and return the raw text."""
    adapter = MockModelAdapter("base")
    prompt = PromptSpec(prompt="hello world")
    encoded = adapter.encode(prompt)
    assert encoded.text == "hello world"
    assert encoded.tokens == ["hello", "world"]


def test_mock_generate_is_deterministic_for_same_inputs():
    """MockModelAdapter.generate should be deterministic for identical prompt + generation settings."""
    adapter = MockModelAdapter("base")
    prompt = PromptSpec(prompt="alpha beta")
    gen = GenerationParams(max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True, seed=123)

    out1 = adapter.generate(prompt, gen)
    out2 = adapter.generate(prompt, gen)

    assert out1 == out2


def test_mock_generate_length_scales_with_max_tokens():
    """MockModelAdapter.generate should grow output length when max_new_tokens is larger."""
    adapter = MockModelAdapter("base")
    prompt = PromptSpec(prompt="alpha beta")

    short = adapter.generate(prompt, GenerationParams(max_new_tokens=20))
    long = adapter.generate(prompt, GenerationParams(max_new_tokens=200))

    assert len(short[0].split()) < len(long[0].split())


def test_mock_residual_activations_shape_matches_tokens():
    """MockModelAdapter.residual_activations should return [1, seq, d_model] shaped activations."""
    adapter = MockModelAdapter("base", d_model=8)
    prompt = PromptSpec(prompt="one two three")
    encoded = adapter.encode(prompt)
    acts = adapter.residual_activations(encoded, layer=0, module_path_template="layer.{layer}", output_selector="raw")

    assert len(acts) == 1
    assert len(acts[0]) == len(encoded.tokens)
    assert len(acts[0][0]) == 8
