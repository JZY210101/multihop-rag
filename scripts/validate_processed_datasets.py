"""Check every processed sample against its unmodified original source."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from src.data_adapters import normalize_record
from src.original_datasets import PROCESSING_POLICY, SPLITS, full_paragraph, read_source


def validate(dataset: str, split: str, raw_root: Path, processed_root: Path) -> dict:
    filename, expected = SPLITS[dataset][split]
    path = processed_root / dataset / f"{split}.jsonl"
    source = raw_root / filename
    ids = set()
    steps, orders = Counter(), Counter()
    review_count = 0
    digest = hashlib.sha256()
    source_ids = set()

    def retained_source():
        for index, raw in enumerate(read_source(source)):
            sample_id = raw.get("_id", raw.get("id"))
            if sample_id in source_ids:
                raise ValueError(f"Duplicated source ID: {sample_id}")
            source_ids.add(sample_id)
            yield index, raw

    with path.open(encoding="utf-8") as handle:
        processed = (json.loads(line) for line in handle if line.strip())
        for source_entry, record in zip_longest(retained_source(), processed):
            if source_entry is None or record is None:
                raise ValueError(f"Source/output row counts differ: {path}")
            index, raw = source_entry
            if record["id"] != raw.get("_id", raw.get("id")) or record["id"] in ids:
                raise ValueError(f"Original ID lost or duplicated: {path}:{index}")
            ids.add(record["id"])
            if record["question"] != raw["question"] or (record["dataset"], record["split"]) != (dataset, split):
                raise ValueError(f"Question or split changed: {record['id']}")
            answers = list(dict.fromkeys([raw["answer"], *raw.get("answer_aliases", [])]))
            if record["gold_answers"] != answers:
                raise ValueError(f"Original answers changed: {record['id']}")
            if record["metadata"]["source_row"] != index or record["metadata"]["source_file"] != source.as_posix():
                raise ValueError(f"Invalid source pointer: {record['id']}")
            documents = record["support_documents"]
            if dataset == "musique":
                decomposition = raw["question_decomposition"]
                by_index = {paragraph["idx"]: paragraph for paragraph in raw["paragraphs"]}
                if raw["answerable"] is not True or record["dataset_hop_count"] != len(decomposition):
                    raise ValueError(f"Invalid answerability or hop count: {record['id']}")
                if [doc["source_index"] for doc in documents] != [s["paragraph_support_idx"] for s in decomposition]:
                    raise ValueError(f"Official decomposition order changed: {record['id']}")
                if record["metadata"]["question_decomposition"] != decomposition:
                    raise ValueError(f"Original decomposition changed: {record['id']}")
                for doc in documents:
                    paragraph = by_index[doc["source_index"]]
                    if not paragraph["is_supporting"] or (doc["title"], doc["text"]) != (
                        paragraph["title"], paragraph["paragraph_text"]
                    ):
                        raise ValueError(f"Supporting paragraph changed: {record['id']}/{doc['doc_id']}")
            else:
                titles = {title for title, _ in raw["supporting_facts"]}
                if {doc["title"] for doc in documents} != titles or len(documents) != len(titles):
                    raise ValueError(f"Support selection differs from original annotations: {record['id']}")
                expected_titles = list(dict.fromkeys(title for title, _ in raw["supporting_facts"]))
                if [doc["title"] for doc in documents] != expected_titles or (
                    record["order_source"] != "flashrag_supporting_facts_order"
                    or record["metadata"]["order_details"]["needs_order_review"]
                ):
                    raise ValueError(f"Fixed supporting-title order changed: {record['id']}")
                for doc in documents:
                    title, sentences = raw["context"][doc["source_index"]]
                    if (doc["title"], doc["sentences"], doc["text"]) != (title, sentences, full_paragraph(sentences)):
                        raise ValueError(f"Full supporting paragraph changed: {record['id']}/{doc['doc_id']}")
                for field in ("supporting_facts", "evidences"):
                    if field in raw and record["metadata"].get(field) != raw[field]:
                        raise ValueError(f"Original annotation changed: {record['id']}/{field}")
                if record["dataset_hop_count"] is not None:
                    raise ValueError(f"Inferred execution count presented as official hop annotation: {record['id']}")
            if any(key in record for key in ("prediction", "hallucination_label", "redeep_score")):
                raise ValueError(f"Generation or label leaked into input data: {record['id']}")
            sample = normalize_record(record, index, 2)
            if sample.hop_num != record["num_steps"] or len(sample.hops) != len(documents):
                raise ValueError(f"Adapter lost a document: {record['id']}")
            details = record["metadata"]["order_details"]
            review_count += int(details["needs_order_review"])
            if not details["needs_order_review"]:
                positions = {doc["doc_id"]: position for position, doc in enumerate(documents)}
                if any(positions[left] >= positions[right] for left, right in details.get("dependencies", [])):
                    raise ValueError(f"Constructed dependency order violated: {record['id']}")
            steps[record["num_steps"]] += 1
            orders[record["order_source"]] += 1
    if len(source_ids) != expected or len(ids) != expected:
        raise ValueError(f"Unexpected full split size: {dataset}/{split} {len(ids)} != {expected}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"dataset": dataset, "split": split, "source_rows": len(source_ids),
              "rows": len(ids), "excluded_samples": [], "processing_policy": PROCESSING_POLICY,
              "sha256": digest.hexdigest(),
              "num_steps_distribution": dict(sorted(steps.items())), "order_sources": dict(orders),
              "needs_order_review": review_count, "status": "all_records_match_source"}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    results = [validate(dataset, split, args.raw_dir, args.processed_dir)
               for dataset in SPLITS for split in ("train", "dev")]
    with (args.processed_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "processing_policy": PROCESSING_POLICY,
                   "splits": results}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("ALL_PROCESSED_RECORDS_MATCH_SOURCE", flush=True)


if __name__ == "__main__":
    main()
