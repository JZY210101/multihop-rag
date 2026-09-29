"""Validate every normalized sample used by the oracle multi-hop pipeline."""

from __future__ import annotations

from collections import Counter

from src.data_adapters import load_dataset


CHECKS = [
    ("hotpotqa", "train", "data/FlashRAG_Data/hotpotqa/train_gold.jsonl", 90447),
    ("hotpotqa", "dev", "data/FlashRAG_Data/hotpotqa/dev_gold.jsonl", 7405),
    ("2wikimultihopqa", "train", "data/FlashRAG_Data/2wikimultihopqa/train.jsonl", 15000),
    ("2wikimultihopqa", "dev", "data/FlashRAG_Data/2wikimultihopqa/dev.jsonl", 12576),
    ("musique", "train", "data/FlashRAG_Data/musique/train.jsonl", 19938),
    ("musique", "dev", "data/FlashRAG_Data/musique/dev.jsonl", 2417),
]


def main() -> None:
    for dataset, split, path, expected in CHECKS:
        samples = load_dataset(path, dataset)
        ids = [sample.sample_id for sample in samples]
        invalid = []
        for sample in samples:
            if (
                not sample.question.strip()
                or not sample.gold_answers
                or not sample.hops
                or sample.hop_num != len(sample.hops)
            ):
                invalid.append(sample.sample_id)
                continue
            for hop in sample.hops:
                documents = hop.get("documents", [])
                if not documents or any(
                    not str(document.get("title", "")).strip()
                    or not str(document.get("text", "")).strip()
                    for document in documents
                ):
                    invalid.append(sample.sample_id)
                    break
        distribution = dict(sorted(Counter(sample.hop_num for sample in samples).items()))
        print(
            dataset,
            split,
            f"rows={len(samples)}",
            f"expected={expected}",
            f"unique_ids={len(set(ids))}",
            f"hop_distribution={distribution}",
            f"invalid={len(invalid)}",
            flush=True,
        )
        if len(samples) != expected or len(set(ids)) != len(ids) or invalid:
            raise RuntimeError(f"Dataset validation failed for {dataset}/{split}: {invalid[:5]}")
    print("ALL_DATASET_RECORDS_OK", flush=True)


if __name__ == "__main__":
    main()
