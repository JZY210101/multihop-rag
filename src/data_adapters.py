"""统一适配三个目标多跳数据集，并保留通用格式别名。"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
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


def normalize_record(record: Dict[str, Any], index: int, default_hop: int) -> Sample:
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
    return Sample(
        sample_id=str(_first(record, ["id", "_id", "uid"], index)),
        question=str(_first(record, ["question", "query", "input", "problem"])),
        answer=answers[0] if answers else "",
        gold_answers=answers,
        hop_num=_infer_hops(record, default_hop),
        supporting_facts=_supporting_facts(record),
        raw=record,
    )


def load_dataset(path: str, dataset: str, default_hop: int = 2) -> List[Sample]:
    supported = {"hotpotqa", "musique", "2wikimultihopqa", "2wiki", "multihop-rag", "multihoprag"}
    name = dataset.lower().replace("_", "-")
    if name not in supported:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return [normalize_record(r, i, default_hop) for i, r in enumerate(_read_records(path))]
