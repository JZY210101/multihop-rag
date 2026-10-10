"""Verify Hotpot's fixed supporting-title order against the archived FlashRAG files."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from src.original_datasets import PROCESSING_POLICY, SPLITS, read_source


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(split: str, raw_root: Path, flashrag_root: Path, processed_root: Path) -> dict:
    filename, expected = SPLITS["hotpotqa"][split]
    source = raw_root / filename
    reference = flashrag_root / "hotpotqa" / f"{split}.jsonl"
    output = processed_root / "hotpotqa" / f"{split}.jsonl"
    count = 0
    for raw, old, current in zip_longest(read_source(source), read_source(reference), read_source(output)):
        if raw is None or old is None or current is None:
            raise ValueError(f"Hotpot raw/FlashRAG/processed counts differ: {split}")
        facts = old["metadata"]["supporting_facts"]
        pairs = [list(pair) for pair in zip(facts["title"], facts["sent_id"])]
        if len(facts["title"]) != len(facts["sent_id"]) or pairs != raw["supporting_facts"]:
            raise ValueError(f"FlashRAG supporting annotations differ: {split}:{count}")
        if old["question"] != raw["question"] or old["golden_answers"] != [raw["answer"]]:
            raise ValueError(f"FlashRAG row does not match original question/answer: {split}:{count}")
        titles = list(dict.fromkeys(facts["title"]))
        if current["id"] != raw["_id"] or [doc["title"] for doc in current["support_documents"]] != titles:
            raise ValueError(f"Processed Hotpot order differs from FlashRAG: {split}:{count}")
        count += 1
    if count != expected:
        raise ValueError(f"Unexpected Hotpot count: {split} {count} != {expected}")
    result = {
        "split": split, "rows": count, "status": "all_orders_match_flashrag_supporting_titles",
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
    args = parser.parse_args()
    results = [validate(split, args.raw_dir, args.flashrag_dir, args.processed_dir)
               for split in ("train", "dev")]
    destination = args.processed_dir / "hotpotqa" / "flashrag_order_report.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump({"processing_policy": PROCESSING_POLICY, "splits": results,
                   "note": "Matches annotation title order, not a certified reasoning chain; "
                           "paragraph contents are validated against raw data separately."},
                  handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("ALL_HOTPOT_ORDERS_MATCH_FLASHRAG", flush=True)


if __name__ == "__main__":
    main()
