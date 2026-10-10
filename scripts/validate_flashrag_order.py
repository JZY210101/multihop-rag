"""Compare original/processed support-document order with archived FlashRAG records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.original_datasets import PROCESSING_POLICY, SPLITS, read_source
from scripts.validate_hotpot_flashrag_order import file_sha256


def validate(dataset: str, split: str, raw_root: Path, flashrag_root: Path, processed_root: Path) -> dict:
    filename, raw_count = SPLITS[dataset][split]
    source = raw_root / filename
    reference = flashrag_root / dataset / f"{split}.jsonl"
    output = processed_root / dataset / f"{split}.jsonl"
    raw_rows, processed_rows = iter(read_source(source)), iter(read_source(output))
    count = 0
    for old in read_source(reference):
        raw, current = next(raw_rows, None), next(processed_rows, None)
        if raw is None or current is None:
            raise ValueError(f"FlashRAG contains more rows than raw/processed: {dataset}/{split}")
        if old["question"] != raw["question"] or raw["answer"] not in old["golden_answers"]:
            raise ValueError(f"FlashRAG row does not match original question/answer: {dataset}/{split}:{count}")
        if current["id"] != raw.get("_id", raw.get("id")):
            raise ValueError(f"Processed row does not match original ID: {dataset}/{split}:{count}")
        if dataset == "musique":
            old_steps = old["metadata"]["question_decomposition"]
            if [{k: v for k, v in step.items() if k != "support_paragraph"} for step in old_steps] != raw["question_decomposition"]:
                raise ValueError(f"MuSiQue decomposition differs: {split}:{count}")
            expected = [(step["paragraph_support_idx"], step["support_paragraph"]["title"],
                         step["support_paragraph"]["paragraph_text"]) for step in old_steps]
            actual = [(doc["source_index"], doc["title"], doc["text"]) for doc in current["support_documents"]]
        else:
            facts = old["metadata"]["supporting_facts"]
            if len(facts["title"]) != len(facts["sent_id"]) or (
                [list(pair) for pair in zip(facts["title"], facts["sent_id"])] != raw["supporting_facts"]
            ):
                raise ValueError(f"Supporting annotations differ: {dataset}/{split}:{count}")
            expected = list(dict.fromkeys(facts["title"]))
            actual = [doc["title"] for doc in current["support_documents"]]
        if actual != expected:
            raise ValueError(f"Processed document order differs from FlashRAG: {dataset}/{split}:{count}")
        count += 1
    expected_overlap = 15000 if (dataset, split) == ("2wikimultihopqa", "train") else raw_count
    if count != expected_overlap:
        raise ValueError(f"Unexpected archived FlashRAG size: {dataset}/{split} {count} != {expected_overlap}")
    result = {
        "dataset": dataset, "split": split, "raw_rows": raw_count,
        "compared_rows": count, "rows_without_flashrag_reference": raw_count - count,
        "status": "all_overlapping_orders_match_flashrag",
        "raw_file": source.as_posix(), "raw_sha256": file_sha256(source),
        "reference_file": reference.as_posix(), "reference_sha256": file_sha256(reference),
        "processed_file": output.as_posix(), "processed_sha256": file_sha256(output),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--flashrag-dir", type=Path,
                        default=Path("data/backups/flashrag_20261009/FlashRAG_Data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    args = parser.parse_args()
    results = [validate(dataset, split, args.raw_dir, args.flashrag_dir, args.processed_dir)
               for dataset in args.datasets for split in ("train", "dev")]
    destination = args.processed_dir / "flashrag_order_report.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump({"processing_policy": PROCESSING_POLICY, "splits": results,
                   "note": "2Wiki train has only 15,000 archived FlashRAG rows; remaining raw rows "
                           "use the same supporting-title order convention. Annotation order is "
                           "an experiment policy, not certification of a reasoning chain."},
                  handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("ALL_OVERLAPPING_ORDERS_MATCH_FLASHRAG", flush=True)


if __name__ == "__main__":
    main()
