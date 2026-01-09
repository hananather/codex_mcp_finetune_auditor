from __future__ import annotations

from types import SimpleNamespace

from codex_mcp_auditor.backends.hf import _model_device, _prompt_to_messages, _resolve_module
from codex_mcp_auditor.schemas.common import Message, PromptSpec


class _DummyModel:
    def __init__(self, devices):
        self._devices = devices

    def parameters(self):
        for dev in self._devices:
            yield SimpleNamespace(device=dev)


class _Nested:
    def __init__(self):
        self.inner = "ok"


class _Container:
    def __init__(self):
        self.layers = [_Nested()]


def test_prompt_to_messages_includes_system_and_messages():
    """_prompt_to_messages should prepend system prompt and include provided messages."""
    prompt = PromptSpec(
        system_prompt="sys",
        messages=[Message(role="user", content="hi")],
    )
    msgs = _prompt_to_messages(prompt)
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_prompt_to_messages_uses_prompt_field_when_no_messages():
    """_prompt_to_messages should synthesize a user message when prompt is provided."""
    prompt = PromptSpec(prompt="hello", use_chat_template=False)
    msgs = _prompt_to_messages(prompt)
    assert msgs == [{"role": "user", "content": "hello"}]


def test_resolve_module_walks_attributes_and_indices():
    """_resolve_module should follow attribute paths and list indices."""
    container = _Container()
    target = _resolve_module(container, "layers.0.inner")
    assert target == "ok"


def test_model_device_prefers_first_non_meta_param():
    """_model_device should return the first non-meta parameter device."""
    torch_stub = object()
    meta = SimpleNamespace(type="meta")
    cpu = SimpleNamespace(type="cpu")
    model = _DummyModel([meta, cpu])
    device = _model_device(torch_stub, model)
    assert device is cpu
