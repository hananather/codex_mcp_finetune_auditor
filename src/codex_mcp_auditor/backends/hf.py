from __future__ import annotations

from typing import Any

from .base import Backend, EncodedPrompt, ModelAdapter
from ..schemas.common import GenerationParams, PromptSpec


def _require_hf() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "HF backend requires optional dependencies. Install with: pip install -e '.[hf]'"
        ) from e
    return torch, AutoModelForCausalLM, AutoTokenizer


def _model_device(torch: Any, model: Any) -> Any:
    # device_map="auto" can put parts on different devices; use the first non-meta parameter device.
    try:
        for p in model.parameters():
            if getattr(p, "device", None) is not None and p.device.type != "meta":
                return p.device
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prompt_to_messages(prompt: PromptSpec) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if prompt.system_prompt:
        msgs.append({"role": "system", "content": prompt.system_prompt})
    if prompt.messages:
        msgs.extend([{"role": m.role, "content": m.content} for m in prompt.messages])
    elif prompt.prompt is not None:
        msgs.append({"role": "user", "content": prompt.prompt})
    return msgs


def _encode_prompt(torch: Any, tokenizer: Any, model: Any, prompt: PromptSpec) -> EncodedPrompt:
    device = _model_device(torch, model)
    msgs = _prompt_to_messages(prompt)
    use_chat_template = bool(prompt.use_chat_template) and bool(msgs) and hasattr(tokenizer, "apply_chat_template")
    if use_chat_template and not getattr(tokenizer, "chat_template", None):
        use_chat_template = False

    attention_mask = None
    if use_chat_template:
        try:
            out = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=prompt.add_generation_prompt,
                return_tensors="pt",
            )
            if isinstance(out, dict):
                input_ids = out["input_ids"]
                attention_mask = out.get("attention_mask")
            else:
                input_ids = out
        except Exception:
            # Tokenizers may implement apply_chat_template but have no template configured
            # (e.g., base/non-chat models). Fall back to plain text encoding.
            use_chat_template = False

    if not use_chat_template:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs]) if msgs else (prompt.prompt or "")
        input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        try:
            attention_mask = attention_mask.to(device)
        except Exception:
            attention_mask = None
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    text_for_hash = tokenizer.decode(input_ids[0].detach().cpu().tolist(), skip_special_tokens=False)
    return EncodedPrompt(input_ids=input_ids, attention_mask=attention_mask, tokens=tokens, text=text_for_hash)


def _resolve_module(model: Any, path: str) -> Any:
    obj = model
    for part in [p for p in path.split(".") if p]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


class HFModelAdapter(ModelAdapter):
    def __init__(self, role: str, model: Any, tokenizer: Any):
        super().__init__(role)
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, prompt: PromptSpec) -> EncodedPrompt:
        torch, _, _ = _require_hf()
        return _encode_prompt(torch, self.tokenizer, self.model, prompt)

    def generate(self, prompt: PromptSpec, gen: GenerationParams) -> tuple[str, int, int]:
        torch, _, _ = _require_hf()
        encoded = self.encode(prompt)
        if gen.seed is not None:
            torch.manual_seed(int(gen.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(gen.seed))

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        prompt_len = int(input_ids.shape[1])

        # Ensure pad token
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Avoid noisy GenerationConfig validation warnings by not setting sampling-only
        # params in greedy mode, and overriding model defaults that would otherwise warn.
        do_sample = bool(gen.do_sample)
        temperature = float(gen.temperature) if do_sample else 1.0
        top_p = float(gen.top_p) if do_sample else 1.0
        extra_greedy_kwargs = {"top_k": 50, "use_cache": True} if not do_sample else {}

        gen_kwargs = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            **extra_greedy_kwargs,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            out = self.model.generate(input_ids, **gen_kwargs)
        completion_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return text, prompt_len, int(completion_ids.numel())

    def residual_activations(self, encoded: EncodedPrompt, layer: int, module_path_template: str, output_selector: str) -> Any:
        torch, _, _ = _require_hf()
        layer_path = module_path_template.format(layer=int(layer))
        module = _resolve_module(self.model, layer_path)

        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hs = output
            if output_selector == "first" and isinstance(output, (tuple, list)):
                hs = output[0]
            captured["value"] = hs

        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                if encoded.attention_mask is not None:
                    self.model(encoded.input_ids, attention_mask=encoded.attention_mask)
                else:
                    self.model(encoded.input_ids)
        finally:
            handle.remove()

        if "value" not in captured:
            raise RuntimeError(f"Hook at {layer_path} did not capture activations.")
        hs = captured["value"]
        # Ensure 3D: [batch, seq, d_model]
        if isinstance(hs, torch.Tensor) and hs.ndim == 2:
            hs = hs.unsqueeze(0)
        return hs


class HFBackend(Backend):
    name = "hf"

    def load_model(self, role: str, id_or_path: str, **kwargs: Any) -> ModelAdapter:
        torch, AutoModelForCausalLM, AutoTokenizer = _require_hf()

        revision = kwargs.get("revision") or None
        trust_remote_code = bool(kwargs.get("trust_remote_code", False))
        device_map = kwargs.get("device_map", "auto") or "auto"
        dtype = kwargs.get("dtype", "auto") or "auto"
        attn_impl = kwargs.get("attn_implementation", None) or None

        torch_dtype = None
        if dtype and dtype != "auto":
            torch_dtype = getattr(torch, dtype)

        # Some fine-tune checkpoints accidentally save `use_cache=false` in config.json, which is noisy and can
        # slow generation. Prefer `use_cache=true` for inference.
        config = None
        try:
            from transformers import AutoConfig  # type: ignore

            cfg = AutoConfig.from_pretrained(
                id_or_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            if getattr(cfg, "use_cache", None) is False:
                cfg.use_cache = True
                config = cfg
        except Exception:
            config = None

        model = AutoModelForCausalLM.from_pretrained(
            id_or_path,
            config=config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        model.eval()

        # Some model artifacts (especially local fine-tunes) ship with `use_cache=false` from training.
        # Flip it back on for inference, and avoid noisy GenerationConfig warnings.
        try:
            if getattr(model.config, "use_cache", None) is False:
                model.config.use_cache = True
        except Exception:
            pass
        try:
            if hasattr(model, "generation_config") and getattr(model.generation_config, "use_cache", None) is False:
                model.generation_config.use_cache = True
        except Exception:
            pass

        # Silence spurious "mistral regex" warnings for non-mistral models; keep default behavior otherwise.
        mistral_model_types = {"mistral", "mistral3", "voxtral", "ministral", "pixtral"}
        model_type = str(getattr(model.config, "model_type", "") or "")
        fix_mistral_regex: bool | None = False if (model_type and model_type not in mistral_model_types) else None

        tok_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": id_or_path,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "use_fast": True,
        }
        if fix_mistral_regex is not None:
            tok_kwargs["fix_mistral_regex"] = fix_mistral_regex

        try:
            tok = AutoTokenizer.from_pretrained(**tok_kwargs)
        except TypeError:
            tok_kwargs.pop("fix_mistral_regex", None)
            tok = AutoTokenizer.from_pretrained(**tok_kwargs)

        return HFModelAdapter(role=role, model=model, tokenizer=tok)
