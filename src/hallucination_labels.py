"""Shared weak-label rules for multi-hop RAG hallucination detection."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_F1_THRESHOLD = 0.30
LABEL_METHOD = "answer_em_or_f1"
LABEL_RULE_VERSION = "em_or_f1_basic_bool_v2"
_TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "‐": "-", "‑": "-", "−": "-",
})
_NUMERIC_SPAN = re.compile(r"(?<![\w.,])[+-]?\d+(?:[.,]\d+)*(?!\w|[.,]\d)")
_VALID_THOUSANDS = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_BOOLEAN_WORDS = re.compile(r"(?<!\w)(yes|no)(?!\w)")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "inc", "ltd", "co", "corp", "vs", "etc"}


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) <= 2 or text[0] not in {"'", '"'} or text[-1] != text[0]:
        return text
    inner = text[1:-1].strip()
    if not inner:
        return text
    # Separate quoted titles are not a single wrapper. An apostrophe inside a
    # word (e.g. 'O'Connor') is part of the answer, not another quote delimiter.
    if text[0] == '"' and '"' in inner:
        return text
    if text[0] == "'" and re.search(r"(?<!\w)'|'(?!\w)", inner):
        return text
    return inner


def _strip_final_period(text: str) -> str:
    if not text.endswith("."):
        return text
    stem = text[:-1].rstrip()
    last_word = stem.split()[-1] if stem else ""
    ordinary_word = (
        len(last_word) > 1 and "." not in last_word and last_word not in _ABBREVIATIONS
        and any(character.isalpha() for character in last_word)
    )
    number = re.fullmatch(r"[$€£]?[+-]?\d+(?:\.\d+)?%?", last_word)
    return stem if ordinary_word or number else text


def normalize_answer(text: Any) -> str:
    """Normalize label answers conservatively; this is not official SQuAD EM/F1.

    Keep articles, accents and meaningful punctuation, including numeric signs,
    decimal points, percentages, ranges and symbol-only entity names.
    """
    normalized = unicodedata.normalize("NFC", "" if text is None else str(text)).lower().translate(_TYPOGRAPHY)
    normalized = " ".join(normalized.split())
    # Validate a complete numeric span before removing grouping commas. Never
    # partially rewrite malformed numbers such as 1,234.56,789.
    normalized = _NUMERIC_SPAN.sub(
        lambda match: match.group(0).replace(",", "")
        if _VALID_THOUSANDS.fullmatch(match.group(0)) else match.group(0),
        normalized,
    )
    # Handle both "Paris." and "Paris". in one pass. Each successful step
    # shortens the string, so this reaches a stable result without stripping
    # meaningful periods, quote-only answers or separately quoted titles.
    while True:
        previous = normalized
        normalized = _strip_final_period(_strip_wrapping_quotes(normalized))
        if normalized == previous:
            break
    return normalized


def answer_token_f1(prediction: Any, reference: Any) -> float:
    """Return token-overlap F1 for one prediction/reference pair."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0

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


def best_answer_em(prediction: Any, gold_answers: Sequence[Any]) -> Optional[float]:
    """Return normalized exact match against any nonempty reference/alias."""
    references = [normalize_answer(answer) for answer in gold_answers if answer is not None and str(answer).strip()]
    if not references:
        return None
    normalized = normalize_answer(prediction)
    return float(bool(normalized) and normalized in references)


def hallucination_label_from_f1(answer_f1: Optional[float], threshold: float = DEFAULT_F1_THRESHOLD) -> Optional[int]:
    """Map F1 >= threshold to label 0; the full rule also gates empty/boolean answers."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("F1 threshold must be in [0, 1]")
    if answer_f1 is None:
        return None
    return int(float(answer_f1) < float(threshold))


def answer_label_fields(
    prediction: Any,
    gold_answers: Sequence[Any],
    threshold: float = DEFAULT_F1_THRESHOLD,
) -> Dict[str, Any]:
    """Label final-answer correctness, preserving the original generated text.

    Only boolean golds enable whole-word yes/no extraction. Seeing both words
    forces label 1, even if the sentence would be semantically consistent.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("F1 threshold must be in [0, 1]")
    references = [answer for answer in gold_answers if answer is not None and str(answer).strip()]
    normalized_gold = [normalize_answer(answer) for answer in references]
    answer_for_label = "" if prediction is None else str(prediction)
    boolean_matches: List[str] = []
    boolean_status = "not_applicable"
    if normalized_gold and all(answer in {"yes", "no"} for answer in normalized_gold):
        matching_text = unicodedata.normalize("NFC", answer_for_label).lower()
        boolean_matches = list(dict.fromkeys(_BOOLEAN_WORDS.findall(matching_text)))
        if len(boolean_matches) == 1:
            answer_for_label = boolean_matches[0]
            boolean_status = "single_" + answer_for_label
        else:
            answer_for_label = ""
            boolean_status = "both_yes_and_no" if boolean_matches else "no_yes_no"

    score = best_answer_f1(answer_for_label, references)
    exact_match = best_answer_em(answer_for_label, references)
    normalized_prediction = normalize_answer(answer_for_label)
    if not references:
        label, reason = None, "missing_gold_answers"
    elif boolean_status == "both_yes_and_no":
        label, reason = 1, "boolean_conflict"
    elif boolean_status == "no_yes_no":
        label, reason = 1, "boolean_missing"
    elif boolean_status != "not_applicable":
        label = int(exact_match != 1.0)
        reason = "boolean_match" if label == 0 else "boolean_mismatch"
    elif not normalized_prediction:
        label, reason = 1, "empty_prediction"
    elif exact_match == 1.0:
        label, reason = 0, "exact_match"
    else:
        label = hallucination_label_from_f1(score, threshold)
        reason = "f1_at_or_above_threshold" if label == 0 else "f1_below_threshold"
    return {
        "answer_for_label": answer_for_label,
        "normalized_answer_for_label": normalized_prediction,
        "normalized_gold_answers": normalized_gold,
        "answer_em": exact_match,
        "answer_f1": score,
        "boolean_answer_status": boolean_status,
        "boolean_answer_matches": boolean_matches,
        "hallucination_label": label,
        "hallucination_label_reason": reason,
        "hallucination_label_method": LABEL_METHOD,
        "hallucination_label_version": LABEL_RULE_VERSION,
        "hallucination_f1_threshold": float(threshold),
    }


def _normalize_title(text: Any) -> str:
    """Retain the legacy document matcher independently of answer-label rules."""
    normalized = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", normalized).split())


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
    required = {_normalize_title(title) for title in _support_titles(record)}
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
                retrieved.update(_normalize_title(value) for value in values if value is not None)
            else:
                retrieved.add(_normalize_title(document))
    return required.issubset(retrieved)
