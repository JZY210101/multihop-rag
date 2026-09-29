"""Generate iterative multi-hop traces from gold evidence (no retriever)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hallucination_labels import DEFAULT_F1_THRESHOLD
from .oracle_hop_pipeline import run_oracle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen iterative multi-hop generation with oracle evidence")
    parser.add_argument("--input", required=True, help="Input train/dev JSONL")
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wikimultihopqa", "musique"))
    parser.add_argument("--output", required=True, help="Output JSONL trace file")
    parser.add_argument("--model", default="model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-input-len", type=int, default=3840)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--f1-threshold", type=float, default=DEFAULT_F1_THRESHOLD)
    parser.add_argument(
        "--allow-context-fallback",
        action="store_true",
        help="Allow heuristic extraction from provided context when exact gold titles are absent; not a strict oracle run",
    )
    args = parser.parse_args()
    records = run_oracle_dataset(
        input_path=args.input,
        dataset_name=args.dataset,
        model_path=args.model,
        max_records=args.max_records,
        f1_threshold=args.f1_threshold,
        max_input_len=args.max_input_len,
        max_new_tokens=args.max_new_tokens,
        allow_context_fallback=args.allow_context_fallback,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} oracle iterative traces to {destination}")


if __name__ == "__main__":
    main()
