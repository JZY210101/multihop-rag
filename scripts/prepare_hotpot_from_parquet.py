"""Convert Hugging Face HotpotQA distractor parquet shards to gold JSONL.

The current FlashRAG JSONL is joined by normalized question+answer with the
official Hugging Face distractor records.  The parquet files contain the full
context and sentence-level supporting facts; no external retriever is used.
"""

from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow.parquet as pq


def norm(value: Any) -> str:
    text = str(value or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def key(row: Dict[str, Any]) -> str:
    answer = row.get("answer", row.get("golden_answers", ""))
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return norm(row.get("question", "")) + "\t" + norm(answer)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_parquet(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        yield from pq.read_table(path).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Current FlashRAG JSONL")
    parser.add_argument("--output", required=True, help="New gold-evidence JSONL")
    parser.add_argument("--parquet", nargs="+", required=True, help="One or more HF parquet shards")
    args = parser.parse_args()

    current = read_jsonl(Path(args.input))
    official = list(read_parquet([Path(item) for item in args.parquet]))
    buckets = defaultdict(deque)
    for row in official:
        buckets[key(row)].append(row)

    output = []
    unmatched = []
    for index, row in enumerate(current):
        matches = buckets.get(key(row))
        if not matches:
            unmatched.append((index, row.get("id", index), row.get("question", "")))
            continue
        source = matches.popleft()
        metadata = dict(row.get("metadata") or {})
        metadata["gold_context"] = source["context"]
        metadata["gold_supporting_facts"] = source["supporting_facts"]
        metadata["gold_source"] = "huggingface_hotpotqa_distractor_parquet"
        merged = dict(row)
        merged["metadata"] = metadata
        merged["gold_context"] = source["context"]
        merged["gold_supporting_facts"] = source["supporting_facts"]
        merged["gold_source"] = "huggingface_hotpotqa_distractor_parquet"
        output.append(merged)

    if unmatched:
        examples = "; ".join(f"{item[1]}: {item[2]}" for item in unmatched[:3])
        raise RuntimeError(f"Could not match {len(unmatched)} rows; examples: {examples}")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Matched {len(output)} rows from {len(official)} parquet records")
    print(f"Wrote {len(output)} rows to {destination}")


if __name__ == "__main__":
    main()
