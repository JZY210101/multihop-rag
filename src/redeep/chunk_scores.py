"""Chunk-level ECS/PKS aggregation for the Qwen ReDeEP adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _ranges(text: str, chunk_size: int) -> List[Tuple[int, int]]:
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    ranges = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        ranges.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return ranges


def _tokens_overlapping(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def _encode(embedder: Any, texts: List[str]):
    if embedder is None:
        return None
    return embedder.encode(texts, normalize_embeddings=True)


def _flat_values(value: Any) -> List[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return list(value) if isinstance(value, (list, tuple)) else []


def calculate_chunk_scores(
    representations: Any,
    tokenizer: Any,
    parts: Dict[str, Any],
    response: str,
    token_pks: Dict[str, List[float]],
    response_token_ids: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[Tuple[int, int]]] = None,
    chunk_size: int = 400,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Aggregate ECS and PKS over response/context chunks.

    Character ranges are converted to token ranges using the same tokenizer as
    the model.  Context chunks never cross the external-context boundary.
    """
    import torch
    import torch.nn.functional as F

    prompt = parts["prompt"]
    prefix = parts.get("prefix", "")
    context = parts.get("context", "")
    context_char_start = len(prefix)
    response_char_start = len(prompt)
    actual_ids = representations.input_ids[0].detach().cpu().tolist()
    sequence_length = int(representations.input_ids.shape[-1])

    exact_response_ids = list(response_token_ids or [])
    if exact_response_ids:
        prompt_encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        response_encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids = _flat_values(prompt_encoded["input_ids"])
        decoded_response_ids = _flat_values(response_encoded["input_ids"])
        if (
            prompt_ids != actual_ids[: representations.response_start]
            or decoded_response_ids != exact_response_ids
            or actual_ids[representations.response_start :] != exact_response_ids
        ):
            return {"ecs": {}, "pks": {}, "chunks": [], "embedding_backend": "token_alignment_failed"}
        prompt_offsets = [(int(start), int(end)) for start, end in _flat_values(prompt_encoded["offset_mapping"])]
        response_offsets = [(int(start), int(end)) for start, end in _flat_values(response_encoded["offset_mapping"])]
    else:
        full_text = prompt + response
        encoded = tokenizer(full_text, add_special_tokens=True, return_offsets_mapping=True)
        encoded_ids = _flat_values(encoded["input_ids"])
        if encoded_ids != actual_ids:
            return {"ecs": {}, "pks": {}, "chunks": [], "embedding_backend": "token_alignment_failed"}
        offsets = [(int(start), int(end)) for start, end in _flat_values(encoded["offset_mapping"])]
        offsets = offsets[:sequence_length]
        prompt_offsets = offsets
        response_offsets = [(left - len(prompt), right - len(prompt)) for left, right in offsets]

    context_chunks = []
    for index, (start, end) in enumerate(_ranges(context, chunk_size)):
        global_start, global_end = context_char_start + start, context_char_start + end
        token_indices = _tokens_overlapping(prompt_offsets, global_start, global_end)
        token_indices = [
            index for index in token_indices if representations.context_start <= index < representations.context_end
        ]
        if token_indices:
            context_chunks.append({"id": index, "text": context[start:end], "tokens": token_indices})
    response_chunks = []
    for index, (start, end) in enumerate(_ranges(response, chunk_size)):
        if exact_response_ids:
            token_indices = [
                representations.response_start + index for index in _tokens_overlapping(response_offsets, start, end)
            ]
        else:
            global_start, global_end = response_char_start + start, response_char_start + end
            token_indices = _tokens_overlapping(prompt_offsets, global_start, global_end)
            token_indices = [
                index for index in token_indices if representations.response_start <= index < sequence_length
            ]
        if token_indices:
            response_chunks.append({"id": index, "text": response[start:end], "tokens": token_indices})
    if not context_chunks or not response_chunks:
        return {"ecs": {}, "pks": {}, "chunks": [], "embedding_backend": "none"}

    available = []
    for layer_id, attention in enumerate(representations.attentions):
        for head_id in range(int(attention.shape[1])):
            available.append((layer_id, head_id))
    selected_heads = list(heads) if heads is not None else available
    final_hidden = representations.final_hidden_state[0]
    context_embeddings = _encode(embedder, [chunk["text"] for chunk in context_chunks])
    embedding_backend = "bge" if context_embeddings is not None else "qwen_hidden_state"
    ecs: Dict[str, List[float]] = {f"layer_{layer}_head_{head}": [] for layer, head in selected_heads}
    chunk_output = []

    for response_chunk in response_chunks:
        response_tokens = response_chunk["tokens"]
        # Attention/FFN states at p predict response token p+1.  Chunk ECS
        # therefore queries the preceding position for each response token.
        query_positions = [position - 1 for position in response_tokens if position > 0]
        if not query_positions:
            continue
        selected_context_by_head = {}
        response_embedding = _encode(embedder, [response_chunk["text"]])[0] if context_embeddings is not None else None
        for layer_id, head_id in selected_heads:
            if layer_id < 0 or layer_id >= len(representations.attentions):
                continue
            attention = representations.attentions[layer_id]
            if head_id < 0 or head_id >= int(attention.shape[1]):
                continue
            scores = []
            layer_attention = attention[0, head_id]
            query_index = torch.tensor(query_positions, dtype=torch.long, device=layer_attention.device)
            for context_chunk in context_chunks:
                context_index = torch.tensor(context_chunk["tokens"], dtype=torch.long, device=layer_attention.device)
                value = layer_attention.index_select(0, query_index).index_select(1, context_index).mean()
                scores.append(value)
            chosen = int(torch.stack(scores).argmax().item())
            selected_context_by_head[(layer_id, head_id)] = chosen
            if context_embeddings is not None:
                similarity = float((response_embedding * context_embeddings[chosen]).sum())
            else:
                response_state = final_hidden[query_positions].mean(dim=0, keepdim=True)
                context_state = final_hidden[context_chunks[chosen]["tokens"]].mean(dim=0, keepdim=True)
                similarity = float(F.cosine_similarity(response_state, context_state, dim=-1).item())
            ecs[f"layer_{layer_id}_head_{head_id}"].append(similarity)

        token_start = max(0, response_tokens[0] - representations.response_start)
        desired_end = response_tokens[-1] - representations.response_start + 1
        chunk_pks = {}
        for key, values in token_pks.items():
            token_end = min(len(values), desired_end)
            selected_values = values[token_start:token_end]
            if selected_values:
                chunk_pks[key] = sum(selected_values) / len(selected_values)
        chunk_output.append(
            {
                "response_chunk_id": response_chunk["id"],
                "response_text": response_chunk["text"],
                "context_chunks": [
                    {
                        "head": [layer, head],
                        "context_chunk_id": (
                            context_chunks[selected_context_by_head[(layer, head)]]["id"]
                            if (layer, head) in selected_context_by_head
                            else None
                        ),
                    }
                    for layer, head in selected_heads
                    if (layer, head) in selected_context_by_head
                ],
                "pks": chunk_pks,
            }
        )

    pks: Dict[str, List[float]] = {key: [] for key in token_pks}
    for chunk in chunk_output:
        for key, value in chunk["pks"].items():
            pks[key].append(float(value))
    return {"ecs": ecs, "pks": pks, "chunks": chunk_output, "embedding_backend": embedding_backend}
