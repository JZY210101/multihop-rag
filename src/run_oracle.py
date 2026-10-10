"""Generate strategy-B traces from processed, full supporting paragraphs."""

import argparse
import json
from pathlib import Path

from .hallucination_labels import DEFAULT_F1_THRESHOLD
from .oracle_hop_pipeline import run_oracle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wikimultihopqa", "musique"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sampling-strategy", choices=("prefix", "random"), default="prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-len", type=int, default=3840)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--f1-threshold", type=float, default=DEFAULT_F1_THRESHOLD,
                        help="Non-boolean answers pass if EM matches or token F1 >= threshold")
    parser.add_argument("--allow-evidence-truncation", action="store_true",
                        help="Explicitly permit partial evidence, recorded as evidence_complete=false")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing trace file")
    args = parser.parse_args()
    if args.batch_size < 1 or args.progress_every < 1 or args.max_new_tokens < 1 or args.max_input_len < 2:
        parser.error("Batch size, progress interval and token limits must be positive")
    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        def write_batch(records):
            nonlocal written
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += len(records)

        run_oracle_dataset(
            input_path=args.input, dataset_name=args.dataset, model_path=args.model,
            max_records=args.max_records, f1_threshold=args.f1_threshold,
            max_input_len=args.max_input_len, max_new_tokens=args.max_new_tokens,
            sampling_strategy=args.sampling_strategy, seed=args.seed, batch_size=args.batch_size,
            progress_every=args.progress_every, allow_evidence_truncation=args.allow_evidence_truncation,
            batch_callback=write_batch,
        )
    print(f"Saved {written} cumulative-evidence traces to {destination}")


if __name__ == "__main__":
    main()
