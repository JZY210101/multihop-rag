"""Extract the Qwen representations needed by ReDeEP.

The original ReDeEP repository patches LLaMA's forward method.  Qwen2 has the
same pre-norm decoder layout, so we can collect the FFN residual states with
forward hooks and keep the model checkpoint unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class QwenRepresentations:
    input_ids: Any
    prefix_len: int
    context_start: int
    context_end: int
    response_start: int
    prediction_start: int
    prediction_end: int
    attentions: Tuple[Any, ...]
    final_hidden_state: Any
    residual_before_ffn: Dict[int, Any]
    hidden_after_ffn: Dict[int, Any]
    logits: Optional[Any] = None


class QwenRepresentationExtractor:
    """Run one full-sequence, teacher-forced Qwen forward pass."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in runtime env
            raise ImportError("Qwen ReDeEP requires torch and transformers. Install requirements.txt first.") from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        # Qwen's SDPA/flash implementations may omit attention matrices.
        # ReDeEP requires the actual eager attention implementation; silently
        # changing the config after loading does not rebuild attention modules.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", **load_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "This ReDeEP adapter requires transformers>=4.45.0 so Qwen can be loaded "
                "with attn_implementation='eager'."
            ) from exc

        if device_map is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        self.device = device or self._input_device()
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.backbone = self._find_backbone()
        self.layers = self._find_layers()

    def _input_device(self):
        return str(self.model.get_input_embeddings().weight.device)

    def _find_layers(self):
        candidates = [getattr(self.backbone, "layers", None)]
        for layers in candidates:
            if layers is not None:
                return layers
        raise ValueError("Could not locate Qwen decoder layers at model.model.layers")

    def _find_final_norm(self):
        return getattr(self.backbone, "norm", None)

    def _find_backbone(self):
        candidates = [
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "base_model", None), "model", None),
            getattr(self.model, "base_model", None),
        ]
        for backbone in candidates:
            if backbone is not None and getattr(backbone, "layers", None) is not None:
                return backbone
        raise ValueError("Could not locate the Qwen decoder backbone (expected model.model.layers)")

    def _tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=True)

    def _offsets(self, text: str):
        try:
            encoded = self.tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
            return [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        except (TypeError, KeyError, ValueError, NotImplementedError):
            return None

    def _register_ffn_hooks(self, prediction_start: int, prediction_end: int):
        residuals: Dict[int, Any] = {}
        after_ffn: Dict[int, Any] = {}
        handles = []

        for layer_id, layer in enumerate(self.layers):
            norm = getattr(layer, "post_attention_layernorm", None)
            if norm is None:
                raise ValueError(f"Qwen layer {layer_id} has no post_attention_layernorm")

            def capture_residual(module, inputs, index=layer_id):
                if inputs:
                    residuals[index] = inputs[0][:, prediction_start:prediction_end].detach()

            def capture_layer_output(module, inputs, output, index=layer_id):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(value, "detach"):
                    after_ffn[index] = value[:, prediction_start:prediction_end].detach()

            handles.append(norm.register_forward_pre_hook(capture_residual))
            handles.append(layer.register_forward_hook(capture_layer_output))
        return handles, residuals, after_ffn

    @staticmethod
    def _last_hidden(outputs):
        value = getattr(outputs, "last_hidden_state", None)
        if value is not None:
            return value
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise ValueError("Qwen output did not contain last_hidden_state")

    def extract(
        self,
        prompt: str,
        response: str,
        response_token_ids: Optional[Sequence[int]] = None,
    ) -> QwenRepresentations:
        """Extract attention and FFN states for ``prompt + response``.

        All offsets use token positions in the combined sequence and follow
        Python's left-closed, right-open convention.
        """
        context_prefix = getattr(self, "_context_prefix", "")
        context_text = getattr(self, "_context_text", "")
        prompt_tokens = self._tokenize(prompt)
        prompt_input_ids = prompt_tokens["input_ids"]
        prefix_len = int(prompt_input_ids.shape[-1])
        exact_response_ids = list(response_token_ids or [])
        if exact_response_ids:
            if not all(isinstance(token_id, int) and token_id >= 0 for token_id in exact_response_ids):
                raise ValueError("response_token_ids must contain non-negative integers")
            response_tensor = self.torch.tensor(exact_response_ids, dtype=prompt_input_ids.dtype).unsqueeze(0)
            input_ids = self.torch.cat([prompt_input_ids, response_tensor], dim=-1).to(self._input_device())
            attention_mask = self.torch.ones_like(input_ids)
            response_start = prefix_len
            offsets = self._offsets(prompt)
        else:
            full_tokens = self._tokenize(prompt + response)
            input_ids = full_tokens["input_ids"].to(self._input_device())
            attention_mask = full_tokens.get("attention_mask").to(input_ids.device)
            response_start = prefix_len
            offsets = self._offsets(prompt + response)
        context_start = 0
        context_end = prefix_len
        if offsets is None:
            raise RuntimeError("Qwen ReDeEP requires a fast tokenizer with return_offsets_mapping support")
        context_char_start = len(context_prefix)
        context_char_end = context_char_start + len(context_text)
        context_tokens = [
            index for index, (start, end) in enumerate(offsets) if end > context_char_start and start < context_char_end
        ]
        if context_tokens:
            context_start, context_end = min(context_tokens), max(context_tokens) + 1
        if not exact_response_ids:
            response_char_start = len(prompt)
            response_tokens = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > response_char_start and start < len(prompt) + len(response)
            ]
            if response_tokens:
                response_start = min(response_tokens)

        sequence_length = int(input_ids.shape[-1])
        # A causal decoder state at p predicts the token at p+1.  Therefore
        # ReDeEP scores response token positions using the preceding states.
        prediction_start = max(0, min(response_start - 1, sequence_length))
        prediction_end = max(prediction_start, min(sequence_length - 1, sequence_length))
        if prediction_start >= prediction_end:
            raise ValueError("The response did not produce any causal prediction positions")

        handles, residuals, after_ffn = self._register_ffn_hooks(prediction_start, prediction_end)
        final_hidden = {}
        final_norm = self._find_final_norm()
        if final_norm is not None:

            def capture_final(module, inputs, output):
                final_hidden["value"] = output.detach()

            handles.append(final_norm.register_forward_hook(capture_final))
        try:
            with self.torch.no_grad():
                # Calling the backbone avoids materializing the very large
                # [batch, sequence, vocabulary] CausalLM logits tensor.
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        attentions = getattr(outputs, "attentions", None)
        if not attentions or any(attention is None for attention in attentions):
            raise RuntimeError(
                "Qwen did not return attention matrices. Load with eager attention and output_attentions=True."
            )
        if len(attentions) != len(self.layers):
            raise RuntimeError(
                f"Qwen returned {len(attentions)} attention tensors for {len(self.layers)} decoder layers."
            )
        if not residuals or not after_ffn:
            raise RuntimeError("Qwen FFN hooks did not capture any residual states")
        if "value" in final_hidden:
            final_hidden_value = final_hidden["value"]
        else:
            final_hidden_value = self._last_hidden(outputs).detach()
        return QwenRepresentations(
            input_ids=input_ids,
            prefix_len=prefix_len,
            context_start=max(0, min(context_start, prefix_len)),
            context_end=max(0, min(context_end, prefix_len)),
            response_start=max(0, min(response_start, int(input_ids.shape[-1]))),
            prediction_start=prediction_start,
            prediction_end=prediction_end,
            attentions=tuple(att.detach() for att in attentions),
            final_hidden_state=final_hidden_value,
            residual_before_ffn=residuals,
            hidden_after_ffn=after_ffn,
            # ReDeEP recomputes only the required Logit Lens projections from
            # FFN states.  Keeping the full [batch, seq, vocab] logits here
            # would waste substantial memory for Qwen's vocabulary.
            logits=None,
        )

    def extract_with_parts(
        self,
        parts: Dict[str, Any],
        response: str,
        response_token_ids: Optional[Sequence[int]] = None,
    ) -> QwenRepresentations:
        """Extract while sharing the exact prompt parts used by the detector."""
        self._context_prefix = parts.get("prefix", "")
        self._context_text = parts.get("context", "")
        try:
            return self.extract(parts["prompt"], response, response_token_ids=response_token_ids)
        finally:
            self._context_prefix = ""
            self._context_text = ""
