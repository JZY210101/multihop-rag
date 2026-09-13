"""ECS and PKS calculations for Qwen ReDeEP."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .qwen_extractor import QwenRepresentationExtractor, QwenRepresentations


def _key(layer: int, head: int) -> str:
    return f"layer_{layer}_head_{head}"


def _device(module: Any, fallback: Any):
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError):
        return fallback.device


def calculate_ecs(
    representations: QwenRepresentations,
    heads: Optional[Sequence[Tuple[int, int]]] = None,
    top_fraction: float = 0.10,
) -> Dict[str, List[float]]:
    """Calculate token-level ECS for the selected (or all) Qwen heads."""
    torch = __import__("torch")
    import torch.nn.functional as F

    sequence_length = int(representations.input_ids.shape[-1])
    context_start = int(representations.context_start)
    context_end = int(representations.context_end)
    prediction_start = int(getattr(representations, "prediction_start", representations.response_start - 1))
    prediction_end = int(getattr(representations, "prediction_end", sequence_length - 1))
    if context_end <= context_start or prediction_start >= prediction_end:
        return {}
    context_length = context_end - context_start
    top_k = max(1, int(round(context_length * top_fraction)))
    top_k = min(top_k, context_length)
    final_hidden = representations.final_hidden_state[0]
    response_positions = range(max(0, prediction_start), min(sequence_length, prediction_end))

    available = []
    for layer_id, attention in enumerate(representations.attentions):
        if attention is None:
            continue
        for head_id in range(int(attention.shape[1])):
            available.append((layer_id, head_id))
    selected = list(heads) if heads is not None else available
    result: Dict[str, List[float]] = {_key(layer, head): [] for layer, head in selected}
    for layer_id, head_id in selected:
        if layer_id < 0 or layer_id >= len(representations.attentions):
            continue
        attention = representations.attentions[layer_id]
        if head_id < 0 or head_id >= int(attention.shape[1]):
            continue
        for position in response_positions:
            row = attention[0, head_id, position, context_start:context_end]
            indices = torch.topk(row, k=top_k, dim=-1).indices + context_start
            attended = final_hidden[indices.to(final_hidden.device)]
            current = final_hidden[position].unsqueeze(0)
            score = F.cosine_similarity(attended.mean(dim=0, keepdim=True), current, dim=-1)
            result[_key(layer_id, head_id)].append(float(score.detach().cpu().item()))
    return result


def stable_jsd(logits_a: Any, logits_b: Any):
    """Jensen-Shannon divergence for batched logits."""
    import torch.nn.functional as F

    log_p = F.log_softmax(logits_a.float(), dim=-1)
    log_q = F.log_softmax(logits_b.float(), dim=-1)
    p, q = log_p.exp(), log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-12).log()
    return 0.5 * (F.kl_div(log_m, p, reduction="none").sum(dim=-1) + F.kl_div(log_m, q, reduction="none").sum(dim=-1))


def _model_norm(model: Any):
    for owner in (getattr(model, "model", None), getattr(getattr(model, "base_model", None), "model", None)):
        norm = getattr(owner, "norm", None)
        if norm is not None:
            return norm
    return None


def calculate_pks(
    extractor: QwenRepresentationExtractor,
    representations: QwenRepresentations,
    layers: Optional[Sequence[int]] = None,
    batch_size: int = 8,
) -> Dict[str, List[float]]:
    """Calculate token-level PKS from FFN-before/after residual states."""
    sequence_length = int(representations.input_ids.shape[-1])
    prediction_start = int(getattr(representations, "prediction_start", representations.response_start - 1))
    prediction_end = int(getattr(representations, "prediction_end", sequence_length - 1))
    positions = list(range(max(0, prediction_start), min(sequence_length, prediction_end)))
    selected = list(layers) if layers is not None else sorted(representations.residual_before_ffn)
    norm = _model_norm(extractor.model)
    lm_head = extractor.model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Qwen model has no output embedding/lm_head")
    norm_device = _device(norm, next(lm_head.parameters())) if norm is not None else next(lm_head.parameters()).device
    head_device = _device(lm_head, next(lm_head.parameters()))
    output: Dict[str, List[float]] = {str(layer): [] for layer in selected}

    for layer in selected:
        before = representations.residual_before_ffn.get(layer)
        after = representations.hidden_after_ffn.get(layer)
        if before is None or after is None:
            continue
        # The extractor stores only the causal prediction states, in order of
        # response tokens, to avoid retaining a full copy for every layer.
        if before.ndim == 3:
            before = before[0]
        if after.ndim == 3:
            after = after[0]
        if len(before) != len(positions) or len(after) != len(positions):
            raise RuntimeError(
                f"FFN state length mismatch for layer {layer}: "
                f"before={len(before)}, after={len(after)}, expected={len(positions)}"
            )
        for start in range(0, len(positions), max(1, batch_size)):
            before_batch = before[start : start + batch_size].to(norm_device)
            after_batch = after[start : start + batch_size].to(norm_device)
            if norm is not None:
                before_batch = norm(before_batch)
                after_batch = norm(after_batch)
            before_logits = lm_head(before_batch.to(head_device))
            after_logits = lm_head(after_batch.to(head_device))
            values = stable_jsd(before_logits, after_logits).detach().cpu().tolist()
            output[str(layer)].extend(float(value) for value in values)
    return output


def mean_features(features: Dict[str, List[float]]) -> Dict[str, float]:
    return {key: sum(values) / len(values) for key, values in features.items() if values}
