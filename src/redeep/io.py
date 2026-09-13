"""Input, trace, and weak-label utilities for the multi-hop ReDeEP adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..hallucination_labels import (
    DEFAULT_F1_THRESHOLD,
    answer_label_fields,
    retrieval_sufficient,
    trace_hops,
)


_LABEL_MODE_ALIASES = {
    "f1_answer": "f1_answer",
    "f1_retrieval_aware": "f1_retrieval_aware",
    "weak_answer": "f1_answer",
    "retrieval_aware": "f1_retrieval_aware",
}


def read_records(path: str) -> List[Dict[str, Any]]:
    """Read JSON/JSONL and normalize common FlashRAG save layouts."""
    source = Path(path)
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            return [_unwrap_record(json.loads(line)) for line in handle if line.strip()]

    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [_unwrap_record(record) for record in payload]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object/list in {path}")
    for key in ("redeep_records", "records", "data", "examples", "items"):
        value = payload.get(key)
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return [_unwrap_record(record) for record in value]

    # FlashRAG Dataset.save may store output columns as parallel arrays.
    trace_values = payload.get("fixed_hop_trace")
    predictions = payload.get("pred", payload.get("prediction", payload.get("response")))
    questions = payload.get("question", payload.get("questions"))
    if isinstance(trace_values, list) and isinstance(predictions, list):
        size = len(predictions)
        return [
            {
                "id": _at(payload.get("id", payload.get("sample_id")), index, index),
                "question": _at(questions, index, ""),
                "gold_answer": _at(payload.get("gold_answer", payload.get("answer")), index, ""),
                "gold_answers": _at(payload.get("gold_answers", payload.get("golden_answers")), index, []),
                "trace": _at(trace_values, index, {}),
                "prediction": prediction,
                "answer_prompt": _at(payload.get("answer_prompt"), index, ""),
            }
            for index, prediction in enumerate(predictions[:size])
        ]
    raise ValueError(f"Could not find records in JSON file {path}")


def _unwrap_record(record: Any) -> Dict[str, Any]:
    """Flatten FlashRAG's per-item ``output`` mapping when present."""
    if not isinstance(record, dict):
        raise ValueError("Each ReDeEP input record must be a JSON object")
    output = record.get("output")
    if not isinstance(output, dict):
        return record
    merged = {**record, **output}
    nested = output.get("redeep_records")
    if isinstance(nested, dict):
        merged = {**merged, **nested}
    return merged


def _at(value: Any, index: int, default: Any) -> Any:
    if isinstance(value, list) and index < len(value):
        return value[index]
    return default if value is None else value


def _first(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    metadata = mapping.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
    raw = mapping.get("raw")
    if isinstance(raw, dict):
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key]
    return default


def record_fields(record: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    """Return ``(id, question, prediction, trace)`` from common layouts."""
    sample_id = str(_first(record, ("id", "sample_id", "uid"), ""))
    question = str(_first(record, ("question", "query", "input"), ""))
    prediction = str(_first(record, ("prediction", "pred", "response", "generated_answer"), ""))
    trace = _first(record, ("trace", "fixed_hop_trace"), {})
    if isinstance(trace, list):
        if (
            trace
            and all(isinstance(item, dict) for item in trace)
            and any("hop" in item and "documents" in item for item in trace)
        ):
            trace = {"hops": trace}
        else:
            trace = trace[0] if trace else {}
    return sample_id, question, prediction, trace if isinstance(trace, dict) else {}


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document
    if not isinstance(document, dict):
        return str(document)
    if "document" in document and isinstance(document["document"], (dict, str)):
        document = document["document"]
        if isinstance(document, str):
            return document
    # FlashRAG uses ``contents``; retain the same precedence as the pipeline
    # so the detector reconstructs the exact evidence shown to the generator.
    return str(document.get("contents", document.get("text", document.get("content", ""))))


def _document_id(document: Any, fallback: int) -> str:
    if isinstance(document, dict):
        if "document" in document and isinstance(document["document"], dict):
            document = document["document"]
        return str(document.get("id", document.get("doc_id", document.get("title", fallback))))
    return str(fallback)


def build_prompt_parts(question: str, trace: Dict[str, Any], answer_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Build a deterministic prompt and character-level context boundaries."""
    context_lines: List[str] = []
    documents: List[Dict[str, Any]] = []
    document_counter = 0
    for hop_index, hop in enumerate(trace_hops(trace), start=1):
        hop_number = hop.get("hop", hop_index)
        for document in hop.get("documents", []):
            document_counter += 1
            text = _document_text(document)
            if not text.strip():
                continue
            doc_id = _document_id(document, document_counter)
            context_lines.append(f"[hop={hop_number} doc_id={doc_id}] {text}")
            documents.append({"hop": hop_number, "doc_id": doc_id, "text": text})
    context = "\n".join(context_lines)
    prefix = (
        "Answer using only the retrieved evidence. Give only the final answer.\n" f"Question: {question}\nEvidence:\n"
    )
    suffix = "\nAnswer:"
    prompt = prefix + context + suffix
    # Prefer the prompt saved by FixedHopPipeline when it is available.  This
    # keeps token offsets identical to the prompt that produced the answer,
    # including any future prompt-template changes.
    if answer_prompt:
        answer_prompt = str(answer_prompt)
        if context:
            context_offset = answer_prompt.find(context)
            if context_offset >= 0:
                prefix = answer_prompt[:context_offset]
                suffix = answer_prompt[context_offset + len(context) :]
                prompt = answer_prompt
        else:
            prefix, suffix, prompt = answer_prompt, "", answer_prompt
    return {
        "prompt": prompt,
        "prefix": prefix,
        "context": context,
        "suffix": suffix,
        "documents": documents,
    }


def gold_answers(record: Dict[str, Any]) -> List[str]:
    answers = _first(record, ("gold_answers", "golden_answers", "gold_answer", "answer", "target"), [])
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer) for answer in answers if answer is not None and str(answer).strip()]


def normalize_label_mode(mode: str) -> str:
    try:
        return _LABEL_MODE_ALIASES[mode]
    except KeyError as exc:
        choices = ", ".join(sorted(_LABEL_MODE_ALIASES))
        raise ValueError(f"Unknown label mode: {mode}. Choose one of: {choices}") from exc


def make_label(
    record: Dict[str, Any],
    trace: Dict[str, Any],
    mode: str = "f1_answer",
    f1_threshold: float = DEFAULT_F1_THRESHOLD,
) -> Optional[int]:
    """Return the shared answer-F1 label, optionally gated by retrieval recall."""
    normalized_mode = normalize_label_mode(mode)
    prediction = record_fields(record)[2]
    label = answer_label_fields(prediction, gold_answers(record), f1_threshold)["hallucination_label"]
    if normalized_mode == "f1_retrieval_aware" and not retrieval_sufficient(record, trace):
        return None
    return label


def enrich_record(
    record: Dict[str, Any],
    label_mode: str = "f1_answer",
    f1_threshold: float = DEFAULT_F1_THRESHOLD,
) -> Dict[str, Any]:
    sample_id, question, prediction, trace = record_fields(record)
    if not question:
        question = str(record.get("raw", {}).get("question", "")) if isinstance(record.get("raw"), dict) else ""
    answer_prompt = _first(record, ("answer_prompt",), "")
    saved_parts = _first(record, ("prompt_parts",), None)
    if isinstance(saved_parts, dict) and all(key in saved_parts for key in ("prompt", "prefix", "context", "suffix")):
        parts = dict(saved_parts)
        if parts["prompt"] != parts["prefix"] + parts["context"] + parts["suffix"]:
            raise ValueError("Saved prompt_parts do not reconstruct the saved prompt")
    else:
        parts = build_prompt_parts(question, trace, answer_prompt=answer_prompt)
    answers = gold_answers(record)
    label_fields = answer_label_fields(prediction, answers, f1_threshold)
    base_label = label_fields["hallucination_label"]
    sufficient = retrieval_sufficient(record, trace)
    label = make_label(record, trace, label_mode, f1_threshold)
    hop_num = _first(record, ("hop_num", "num_hops", "num_hop"), trace.get("hop_num"))
    if hop_num is None:
        hop_num = len(trace_hops(trace)) or None
    response_token_ids = _first(record, ("response_token_ids", "generated_token_ids"), [])
    if hasattr(response_token_ids, "tolist"):
        response_token_ids = response_token_ids.tolist()
    if isinstance(response_token_ids, list) and response_token_ids and isinstance(response_token_ids[0], list):
        response_token_ids = response_token_ids[0]
    if not isinstance(response_token_ids, list) or not all(
        isinstance(token_id, int) for token_id in response_token_ids
    ):
        response_token_ids = []
    return {
        "id": sample_id,
        "question": question,
        "prediction": prediction,
        "response_token_ids": response_token_ids,
        "gold_answers": answers,
        "trace": trace,
        "hop_num": hop_num,
        "prompt_parts": parts,
        "answer_f1": label_fields["answer_f1"],
        "hallucination_label": label,
        "base_hallucination_label": base_label,
        "hallucination_label_method": label_fields["hallucination_label_method"],
        "hallucination_f1_threshold": label_fields["hallucination_f1_threshold"],
        "retrieval_sufficient": sufficient,
        "raw": record,
    }
