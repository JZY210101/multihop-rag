"""Shared weak-label rules for multi-hop RAG hallucination detection."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_F1_THRESHOLD = 0.30
LABEL_METHOD = "answer_f1_threshold"
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: Any) -> str:
    """Apply the standard HotpotQA-style English answer normalization."""
    normalized = str(text).lower().translate(_PUNCT_TRANSLATION)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def answer_token_f1(prediction: Any, reference: Any) -> float:
    """Return token-overlap F1 for one prediction/reference pair."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_answer_f1(prediction: Any, gold_answers: Sequence[Any]) -> Optional[float]:
    """Return the maximum token F1 across all non-empty gold aliases."""
    references = [answer for answer in gold_answers if answer is not None and str(answer).strip()]
    if not references:
        return None
    return max(answer_token_f1(prediction, reference) for reference in references)


def hallucination_label_from_f1(answer_f1: Optional[float], threshold: float = DEFAULT_F1_THRESHOLD) -> Optional[int]:
    """Map answer F1 to 1=hallucination and 0=non-hallucination."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("F1 threshold must be in [0, 1]")
    if answer_f1 is None:
        return None
    return int(float(answer_f1) <= float(threshold))


def answer_label_fields(
    prediction: Any,
    gold_answers: Sequence[Any],
    threshold: float = DEFAULT_F1_THRESHOLD,
) -> Dict[str, Any]:
    """Build reusable answer-level weak-label fields for pipeline outputs."""
    score = best_answer_f1(prediction, gold_answers)
    return {
        "answer_f1": score,
        "hallucination_label": hallucination_label_from_f1(score, threshold),
        "hallucination_label_method": LABEL_METHOD,
        "hallucination_f1_threshold": float(threshold),
    }


def trace_hops(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hops = trace.get("hops", trace.get("trace", []))
    if not isinstance(hops, list):
        return []
    return [hop for hop in hops if isinstance(hop, dict)]


def _support_titles(record: Mapping[str, Any]) -> List[str]:
    raw = record.get("raw", {})
    raw = raw if isinstance(raw, dict) else {}
    metadata = record.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    raw_metadata = raw.get("metadata", {})
    if isinstance(raw_metadata, dict):
        metadata = {**raw_metadata, **metadata}

    facts = metadata.get("supporting_facts", record.get("supporting_facts", raw.get("supporting_facts", [])))
    if isinstance(facts, dict):
        facts = facts.get("title", facts.get("facts", []))
    if isinstance(facts, str):
        facts = [facts]
    titles: List[str] = []
    if isinstance(facts, list):
        for fact in facts:
            title = fact if isinstance(fact, str) else fact[0] if isinstance(fact, (list, tuple)) and fact else None
            if isinstance(fact, dict):
                title = fact.get("title", fact.get("doc_id", title))
            if title and str(title) not in titles:
                titles.append(str(title))

    decomposition = metadata.get(
        "question_decomposition",
        record.get("question_decomposition", raw.get("question_decomposition", [])),
    )
    if isinstance(decomposition, list):
        for step in decomposition:
            support = step.get("support_paragraph") if isinstance(step, dict) else None
            if isinstance(support, dict) and support.get("title"):
                title = str(support["title"])
                if title not in titles:
                    titles.append(title)
    return titles


def retrieval_sufficient(record: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    """Approximate supporting-document recall using titles and document ids."""
    required = {normalize_answer(title) for title in _support_titles(record)}
    if not required:
        return True

    retrieved = set()
    for hop in trace_hops(trace):
        for document in hop.get("documents", []):
            if isinstance(document, dict):
                nested = document.get("document") if isinstance(document.get("document"), dict) else {}
                values = [
                    document.get("id"),
                    document.get("doc_id"),
                    document.get("title"),
                    nested.get("id"),
                    nested.get("doc_id"),
                    nested.get("title"),
                ]
                for owner in (document, nested):
                    contents = owner.get("contents")
                    if isinstance(contents, str) and contents.strip():
                        values.append(contents.splitlines()[0].strip().strip('"'))
                    owner_metadata = owner.get("metadata")
                    if isinstance(owner_metadata, dict):
                        values.extend(
                            [
                                owner_metadata.get("id"),
                                owner_metadata.get("doc_id"),
                                owner_metadata.get("title"),
                            ]
                        )
                retrieved.update(normalize_answer(value) for value in values if value is not None)
            else:
                retrieved.add(normalize_answer(document))
    return required.issubset(retrieved)
