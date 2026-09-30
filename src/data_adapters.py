"""统一适配三个目标多跳数据集，并保留通用格式别名。"""
import json
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from .schema import Sample


def _read_records(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        yield from (obj if isinstance(obj, list) else obj.get("data", obj.get("examples", [])))


def _first(record: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    sources = [record]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        sources.append(metadata)
    for source in sources:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return default


def _infer_hops(record: Dict[str, Any], default: int) -> int:
    explicit = _first(record, ["hop_num", "hop", "num_hops", "num_hop"])
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    decomp = _first(record, ["decomposition", "question_decomposition", "sub_questions", "subquestions"], [])
    if isinstance(decomp, list) and decomp:
        return len(decomp)
    facts = _first(record, ["supporting_facts", "supporting facts", "supportingFacts"], [])
    if isinstance(facts, dict):
        facts = facts.get("title", facts.get("facts", []))
    if isinstance(facts, list) and facts:
        titles = []
        for fact in facts:
            title = (
                fact[0]
                if isinstance(fact, (list, tuple)) and fact
                else fact.get("title")
                if isinstance(fact, dict)
                else fact
            )
            if title not in titles:
                titles.append(title)
        if titles:
            return max(1, len(titles))
    return max(1, default)


def _supporting_facts(record: Dict[str, Any]) -> Any:
    facts = _first(record, ["supporting_facts", "supporting facts", "supportingFacts", "evidence"], None)
    if facts is not None:
        return facts

    decomposition = _first(record, ["question_decomposition"], [])
    if not isinstance(decomposition, list):
        return []
    return [step.get("support_paragraph", step) for step in decomposition if isinstance(step, dict)]


def _metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _document_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return str(value).strip() if value is not None else ""
    return str(value.get("paragraph_text", value.get("text", value.get("contents", value.get("content", ""))))).strip()


def _oracle_hops(record: Dict[str, Any], allow_context_fallback: bool = False) -> List[Dict[str, Any]]:
    """Extract ordered gold supporting evidence without running a retriever.

    FlashRAG's normalized files keep the original dataset fields under
    ``metadata``.  HotpotQA and 2Wiki identify supporting titles/sentences;
    MuSiQue provides an ordered question decomposition with support paragraphs.
    """
    metadata = _metadata(record)
    decomposition = metadata.get("question_decomposition", record.get("question_decomposition", []))
    if isinstance(decomposition, list) and decomposition:
        hops = []
        for hop_index, step in enumerate(decomposition, start=1):
            if not isinstance(step, dict):
                continue
            paragraph = step.get("support_paragraph", {})
            if not isinstance(paragraph, dict) or not paragraph.get("is_supporting", True):
                continue
            text = _document_text(paragraph)
            if not text:
                continue
            hops.append({
                "hop": hop_index,
                "documents": [{
                    "doc_id": str(paragraph.get("title", paragraph.get("idx", hop_index))),
                    "title": str(paragraph.get("title", "")),
                    "text": text,
                    "score": None,
                    "metadata": {"paragraph_idx": paragraph.get("idx", step.get("paragraph_support_idx"))},
                }],
                "sub_question": str(step.get("question", "")),
                "gold_intermediate_answer": str(step.get("answer", "")),
            })
        if hops:
            return hops

    context = metadata.get("gold_context", metadata.get("context", {}))
    if isinstance(context, list):
        titles = [item[0] for item in context if isinstance(item, (list, tuple)) and len(item) >= 2]
        contents = [item[1] for item in context if isinstance(item, (list, tuple)) and len(item) >= 2]
    elif isinstance(context, dict):
        titles = context.get("title", [])
        contents = context.get("sentences", context.get("content", []))
    else:
        return []
    if not isinstance(titles, list) or not isinstance(contents, list):
        return []
    by_title = {str(title): value for title, value in zip(titles, contents)}
    facts = metadata.get("gold_supporting_facts", metadata.get("supporting_facts", {}))
    if isinstance(facts, dict):
        fact_titles = facts.get("title", [])
        fact_sentences = facts.get("sent_id", [])
    elif isinstance(facts, list):
        fact_titles = [item[0] for item in facts if isinstance(item, (list, tuple)) and item]
        fact_sentences = [item[1] for item in facts if isinstance(item, (list, tuple)) and len(item) > 1]
    else:
        fact_titles, fact_sentences = [], []
    if not isinstance(fact_titles, list):
        fact_titles = [fact_titles]

    ordered_titles = []
    for title in fact_titles:
        title = str(title)
        if title not in ordered_titles:
            ordered_titles.append(title)
    hops = []
    missing_titles = [title for title in ordered_titles if title not in by_title]
    if missing_titles and not allow_context_fallback:
        raise ValueError(
            "Gold supporting document(s) are absent from the provided context: "
            f"{missing_titles[:5]}. This file is not a complete oracle-evidence export; "
            "obtain the original dataset or rerun with allow_context_fallback=True."
        )
    for hop_index, title in enumerate(ordered_titles, start=1):
        raw_text = by_title.get(title, "")
        source = "gold_context_title"
        if not raw_text and allow_context_fallback:
            # Keep this explicitly marked as a context fallback.  It is useful
            # for smoke tests, but must not be reported as exact gold evidence.
            raw_text = "\n".join(
                _document_text(value)
                for context_title, value in zip(titles, contents)
                if title.lower() in str(context_title).lower()
                or title.lower() in _document_text(value).lower()
            )
            source = "context_fallback_heuristic"
        sentence_ids = [
            int(sentence_id)
            for fact_title, sentence_id in zip(fact_titles, fact_sentences if isinstance(fact_sentences, list) else [])
            if str(fact_title) == title and isinstance(sentence_id, int)
        ]
        if isinstance(raw_text, list):
            selected = [str(raw_text[i]).strip() for i in sentence_ids if 0 <= i < len(raw_text)]
            text = " ".join(selected) if selected else " ".join(str(item).strip() for item in raw_text)
        else:
            text = _document_text(raw_text)
        if text:
            hops.append({
                "hop": hop_index,
                "documents": [{"doc_id": title, "title": title, "text": text, "score": None, "metadata": {}}],
                "evidence_source": source,
            })
    return hops


def normalize_record(
    record: Dict[str, Any], index: int, default_hop: int, allow_context_fallback: bool = False
) -> Sample:
    answers = _first(
        record,
        ["gold_answers", "golden_answers", "answer", "gold_answer", "target", "output"],
        [],
    )
    if isinstance(answers, list):
        answers = [str(answer) for answer in answers if answer is not None and str(answer).strip()]
    elif answers in (None, ""):
        answers = []
    else:
        answers = [str(answers)]
    hops = _oracle_hops(record, allow_context_fallback=allow_context_fallback)
    return Sample(
        sample_id=str(_first(record, ["id", "_id", "uid"], index)),
        question=str(_first(record, ["question", "query", "input", "problem"])),
        answer=answers[0] if answers else "",
        gold_answers=answers,
        hop_num=len(hops) or _infer_hops(record, default_hop),
        supporting_facts=_supporting_facts(record),
        hops=hops,
        raw=record,
    )


def load_dataset(
    path: str,
    dataset: str,
    default_hop: int = 2,
    allow_context_fallback: bool = False,
    max_records: Optional[int] = None,
) -> List[Sample]:
    supported = {"hotpotqa", "musique", "2wikimultihopqa", "2wiki", "multihop-rag", "multihoprag"}
    name = dataset.lower().replace("_", "-")
    if name not in supported:
        raise ValueError(f"Unsupported dataset: {dataset}")
    records: Iterable[Dict[str, Any]] = _read_records(path)
    if max_records is not None:
        records = islice(records, max(0, int(max_records)))
    return [
        normalize_record(r, i, default_hop, allow_context_fallback=allow_context_fallback)
        for i, r in enumerate(records)
    ]
