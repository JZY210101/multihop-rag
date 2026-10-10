"""Prepare traceable, full supporting paragraphs from original dataset files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List


SCHEMA_VERSION = 1
PROCESSING_POLICY = "flashrag_compatible_support_order_full_v1"
SPLITS = {
    "hotpotqa": {
        "train": ("hotpotqa/raw/hotpot_train_v1.1.json", 90447),
        "dev": ("hotpotqa/raw/hotpot_dev_distractor_v1.json", 7405),
    },
    "2wikimultihopqa": {
        "train": ("2wikimultihopqa/train.json", 167454),
        "dev": ("2wikimultihopqa/dev.json", 12576),
    },
    "musique": {
        "train": ("musique/musique_ans_v1.0_train.jsonl", 19938),
        "dev": ("musique/musique_ans_v1.0_dev.jsonl", 2417),
    },
}


def read_source(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL or a JSON array without loading a full training split."""
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            chunk = handle.read(1024 * 1024)
            buffer += chunk
            eof = not chunk

        fill()
        buffer = buffer.lstrip()
        if not buffer.startswith("["):
            raise ValueError(f"Expected a JSON array in {path}")
        buffer = buffer[1:]
        need_comma = False
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()
            if buffer.startswith("]"):
                trailing = buffer[1:] + handle.read()
                if trailing.strip():
                    raise ValueError(f"Trailing data in {path}")
                return
            if need_comma:
                if not buffer.startswith(","):
                    raise ValueError(f"Missing array separator in {path}")
                buffer = buffer[1:].lstrip()
                need_comma = False
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
                    buffer = buffer.lstrip()
            if not isinstance(record, dict):
                raise ValueError(f"Expected sample objects in {path}")
            yield record
            buffer = buffer[end:]
            need_comma = True


def full_paragraph(sentences: List[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences)


def order_documents(row: Dict[str, Any], documents: List[Dict[str, Any]], dataset: str):
    """Retain supporting-title annotation order, using the FlashRAG convention."""
    return documents, "flashrag_supporting_facts_order", {
        "original_document_order": [doc["doc_id"] for doc in documents],
        "dependencies": [], "needs_order_review": False,
        "is_official_reasoning_order": False,
        "notes": [
            "Fixed experiment order: first occurrence of each supporting-fact title, "
            "using the FlashRAG convention; overlap is verified separately. "
            "This is an annotation order, not an official reasoning chain."
        ],
    }


def convert_record(row: Dict[str, Any], dataset: str, split: str, source_file: str,
                   source_row: int) -> Dict[str, Any]:
    metadata = {"source_file": source_file, "source_row": source_row}
    if dataset == "musique":
        if row.get("answerable") is not True:
            raise ValueError("MuSiQue oracle data must come from the answerable Ans variant")
        sample_id = row["id"]
        paragraphs = {p["idx"]: p for p in row["paragraphs"]}
        if len(paragraphs) != len(row["paragraphs"]):
            raise ValueError(f"Duplicate paragraph indices: {sample_id}")
        decomposition = row["question_decomposition"]
        documents = []
        for step in decomposition:
            index = step["paragraph_support_idx"]
            paragraph = paragraphs[index]
            if not paragraph["is_supporting"]:
                raise ValueError(f"Non-supporting decomposition paragraph: {sample_id}/{index}")
            documents.append({
                "doc_id": f"p{index}", "source_index": index,
                "title": paragraph["title"], "text": paragraph["paragraph_text"],
            })
        if len({doc["doc_id"] for doc in documents}) != len(documents):
            raise ValueError(f"Repeated support paragraph in decomposition: {sample_id}")
        if len(documents) != sum(p["is_supporting"] for p in row["paragraphs"]):
            raise ValueError(f"Supporting paragraph count mismatch: {sample_id}")
        hop_count = int(re.match(r"^(\d+)hop", sample_id).group(1))
        if hop_count != len(decomposition):
            raise ValueError(f"Original hop ID disagrees with decomposition: {sample_id}")
        source = "official_decomposition"
        metadata.update({
            "question_type": sample_id.split("__")[0], "question_decomposition": decomposition,
            "answerable": True, "candidate_document_count": len(row["paragraphs"]),
            "order_details": {"needs_order_review": False, "notes": []},
        })
        answers = list(dict.fromkeys([row["answer"], *row.get("answer_aliases", [])]))
    else:
        sample_id = row["_id"]
        supporting_titles = list(dict.fromkeys(fact[0] for fact in row["supporting_facts"]))
        documents = []
        bad_sentence_refs = []
        for title in supporting_titles:
            matches = [(i, sentences) for i, (name, sentences) in enumerate(row["context"]) if name == title]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous support title: {sample_id}/{title}")
            index, sentences = matches[0]
            documents.append({
                "doc_id": f"p{index}", "source_index": index, "title": title,
                "text": full_paragraph(sentences), "sentences": sentences,
            })
            bad_sentence_refs.extend([title, sid] for name, sid in row["supporting_facts"]
                                     if name == title and not 0 <= sid < len(sentences))
        documents, source, details = order_documents(row, documents, dataset)
        hop_count = None  # No per-sample hop annotation is present in these source files.
        metadata.update({
            "question_type": row["type"], "supporting_facts": row["supporting_facts"],
            "candidate_document_count": len(row["context"]), "order_details": details,
            "annotation_warnings": {"invalid_support_sentence_references": bad_sentence_refs},
        })
        for key in ("level", "evidences", "entity_ids", "evidences_id", "answer_id"):
            if key in row:
                metadata[key] = row[key]
        answers = [row["answer"]]
    if not row["question"].strip() or not documents or any(
        not doc["title"].strip() or not doc["text"].strip() for doc in documents
    ) or not answers or any(not answer.strip() for answer in answers):
        raise ValueError(f"Empty question, answer or support paragraph: {sample_id}")
    return {
        "schema_version": SCHEMA_VERSION, "id": sample_id, "dataset": dataset, "split": split,
        "question": row["question"], "gold_answers": answers, "support_documents": documents,
        "num_steps": len(documents), "dataset_hop_count": hop_count,
        "order_source": source, "metadata": metadata,
    }
