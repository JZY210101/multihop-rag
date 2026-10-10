"""Export six processed JSONL splits from unmodified ModelScope source files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.original_datasets import PROCESSING_POLICY, SPLITS, convert_record, read_source


def prepare(dataset: str, split: str, raw_root: Path, output_root: Path, overwrite: bool) -> dict:
    filename, expected = SPLITS[dataset][split]
    source = raw_root / filename
    destination = output_root / dataset / f"{split}.jsonl"
    temporary = destination.with_suffix(".jsonl.partial")
    if not overwrite and (destination.exists() or temporary.exists()):
        raise FileExistsError(f"Output already exists: {destination}; use --overwrite explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = set()
    steps, orders = Counter(), Counter()
    warnings = reviews = reordered = 0
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(read_source(source)):
            sample_id = raw.get("_id", raw.get("id"))
            if sample_id in ids:
                raise ValueError(f"Duplicate original ID: {sample_id}")
            ids.add(sample_id)
            record = convert_record(raw, dataset, split, source.as_posix(), index)
            steps[record["num_steps"]] += 1
            orders[record["order_source"]] += 1
            details = record["metadata"]["order_details"]
            reviews += int(details["needs_order_review"])
            reordered += int("original_document_order" in details and
                             details["original_document_order"] != [d["doc_id"] for d in record["support_documents"]])
            warnings += len(record["metadata"].get("annotation_warnings", {}).get(
                "invalid_support_sentence_references", []))
            line = json.dumps(record, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            if (index + 1) % 10000 == 0:
                print(f"[prepare] {dataset}/{split}: {index + 1}/{expected}", flush=True)
    if len(ids) != expected:
        raise ValueError(f"Wrong source split size: {dataset}/{split} {len(ids)} != {expected}")
    temporary.replace(destination)
    summary = {
        "path": destination.as_posix(), "source_rows": len(ids),
        "rows": len(ids), "excluded_samples": [],
        "processing_policy": PROCESSING_POLICY, "sha256": digest.hexdigest(),
        "num_steps_distribution": dict(sorted(steps.items())), "order_sources": dict(orders),
        "needs_order_review": reviews, "reordered_samples": reordered,
        "invalid_support_sentence_references": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summaries = [prepare(dataset, split, args.raw_dir, args.output_dir, args.overwrite)
                 for dataset in args.datasets for split in ("train", "dev")]
    with (args.output_dir / "preparation_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "processing_policy": PROCESSING_POLICY,
                   "splits": summaries}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
