"""Prompt sizing helpers shared by generation and ReDeEP extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TRUNCATION_MARKER = "\n[... evidence truncated ...]\n"


def normalize_generated_token_ids(token_ids: Any, eos_token_id: Any = None, pad_token_id: Any = None) -> List[int]:
    """Convert one generated sequence to JSON-safe answer-token ids.

    FlashRAG pads ``return_dict`` token ids to ``max_new_tokens``.  ReDeEP
    needs the actual generated prefix, without EOS or padding, to reproduce
    the generation sequence exactly during teacher forcing.
    """
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, (tuple, list)) and token_ids and isinstance(token_ids[0], (tuple, list)):
        token_ids = token_ids[0]
    if not isinstance(token_ids, (tuple, list)):
        return []

    stop_ids = set()
    for value in (eos_token_id, pad_token_id):
        values = value if isinstance(value, (tuple, list, set)) else [value]
        stop_ids.update(int(item) for item in values if isinstance(item, int))

    normalized = []
    for token_id in token_ids:
        if not isinstance(token_id, int):
            return []
        if token_id in stop_ids:
            break
        normalized.append(token_id)
    return normalized


def token_length(tokenizer: Any, text: str) -> int:
    """Return the token count for the same single-string path used at inference."""
    encoded = tokenizer(text, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return len(input_ids)


def fit_prompt_parts(
    tokenizer: Any,
    prefix: str,
    context: str,
    suffix: str,
    max_tokens: Optional[int],
    response: str = "",
) -> Dict[str, Any]:
    """Fit a prompt by retaining both the earliest and latest evidence."""
    prefix, context, suffix, response = map(str, (prefix, context, suffix, response))
    if max_tokens is None:
        prompt = prefix + context + suffix
        return {
            "prompt": prompt,
            "prefix": prefix,
            "context": context,
            "suffix": suffix,
            "context_truncated": False,
        }

    limit = int(max_tokens)
    if limit < 2:
        raise ValueError("max_tokens must be at least 2")

    def total_tokens(candidate_context: str) -> int:
        return token_length(tokenizer, prefix + candidate_context + suffix + response)

    if total_tokens(context) <= limit:
        prompt = prefix + context + suffix
        return {
            "prompt": prompt,
            "prefix": prefix,
            "context": context,
            "suffix": suffix,
            "context_truncated": False,
        }
    if total_tokens("") > limit:
        raise ValueError("Prompt template plus response exceeds the configured token limit")

    def retain_ends(character_count: int) -> str:
        if character_count <= 0:
            return TRUNCATION_MARKER.strip()
        head_count = (character_count + 1) // 2
        tail_count = character_count // 2
        head = context[:head_count].rstrip()
        tail = context[-tail_count:].lstrip() if tail_count else ""
        return head + TRUNCATION_MARKER + tail

    low, high = 0, len(context)
    while low < high:
        middle = (low + high + 1) // 2
        if total_tokens(retain_ends(middle)) <= limit:
            low = middle
        else:
            high = middle - 1
    fitted_context = retain_ends(low)
    if total_tokens(fitted_context) > limit:
        fitted_context = ""
    prompt = prefix + fitted_context + suffix
    return {
        "prompt": prompt,
        "prefix": prefix,
        "context": fitted_context,
        "suffix": suffix,
        "context_truncated": True,
    }
