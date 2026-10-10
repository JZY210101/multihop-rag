"""Run real Qwen generation, ReDeEP fit and evaluate on all three datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.hallucination_labels import LABEL_RULE_VERSION, answer_label_fields
from src.run_redeep import _metrics


def _read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(arguments, log_path, environment):
    print("[smoke] " + " ".join(arguments), flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=environment)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith(("[oracle]", "[redeep]", "Saved ", '{"calibration"', '{"metrics"')):
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Command exited with {returncode}; see {log_path}")


def _check_traces(path, expected_count, dataset):
    records = _read(path)
    if len(records) != expected_count:
        raise ValueError(f"Unexpected record count in {path}")
    for record in records:
        if record.get("dataset") != dataset or not record.get("evidence_complete"):
            raise ValueError(f"Invalid dataset or incomplete evidence: {record.get('id')}")
        if record.get("history_mode") != "cumulative_evidence_and_responses":
            raise ValueError("Smoke test requires cumulative evidence and model responses")
        if len(record["hop_records"]) != record["num_steps"]:
            raise ValueError("Generated step count does not match supporting documents")
        for position, hop in enumerate(record["hop_records"], 1):
            parts = hop["prompt_parts"]
            if len(hop["visible_doc_ids"]) != position or parts["prompt"] != parts["prefix"] + parts["context"] + parts["suffix"]:
                raise ValueError("Invalid evidence visibility or prompt boundaries")
        expected = answer_label_fields(record["prediction"], record["gold_answers"])
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Shared label fields disagree: {record['id']}")
    return records


def _check_scores(path, originals):
    records = _read(path)
    if len(records) != len(originals):
        raise ValueError(f"Unexpected score count in {path}")
    for record, original in zip(records, originals):
        for key in ("id", "prediction", "response_token_ids", "hallucination_label", "answer_for_label",
                    "answer_em", "answer_f1", "hallucination_label_version"):
            if record.get(key) != original.get(key):
                raise ValueError(f"Generation/scoring mismatch in {key}: {record.get('id')}")
        score = record.get("redeep_score")
        if record.get("redeep_error") or score is None or not math.isfinite(score):
            raise ValueError(f"Unscored record {record.get('id')}: {record.get('redeep_error')}")
        if not record.get("ecs") or not record.get("pks"):
            raise ValueError("Missing ReDeEP features")
    return _metrics(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-records", type=int, default=64)
    parser.add_argument("--max-train-records", type=int, default=256)
    parser.add_argument("--dev-records", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_records, args.dev_records, args.generation_batch_size, args.score_batch_size) < 1:
        parser.error("Record counts and batch sizes must be positive")
    if args.max_train_records < args.train_records:
        parser.error("--max-train-records must be at least --train-records")
    root = Path(args.output_dir or f"outputs/smoke_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    environment = dict(os.environ)
    for key, value in {"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
        environment.setdefault(key, value)
    report = {"label_rule_version": LABEL_RULE_VERSION, "model": args.model,
              "generation_batch_size": args.generation_batch_size, "score_batch_size": args.score_batch_size,
              "seed": args.seed, "datasets": {}}
    for dataset in ("hotpotqa", "2wikimultihopqa", "musique"):
        generated = {}
        for split, initial_count in (("train", args.train_records), ("dev", args.dev_records)):
            count = initial_count
            while True:
                output = root / f"{dataset}_{split}_n{count}_oracle.jsonl"
                _run([sys.executable, "-B", "-m", "src.run_oracle", "--input", f"data/processed/{dataset}/{split}.jsonl",
                      "--dataset", dataset, "--model", args.model, "--max-records", str(count),
                      "--sampling-strategy", "random", "--seed", str(args.seed),
                      "--batch-size", str(args.generation_batch_size), "--progress-every", "16",
                      "--output", str(output)], logs / f"{output.stem}.log", environment)
                records = _check_traces(output, count, dataset)
                labels = Counter(record["hallucination_label"] for record in records)
                if split != "train" or set(labels) == {0, 1}:
                    generated[split] = (output, records)
                    break
                if count >= args.max_train_records:
                    raise ValueError(f"{dataset}: only one label class after {count} train examples; no labels were fabricated")
                count = min(count * 2, args.max_train_records)
                print(f"[smoke] {dataset}: increase train sample to {count} to obtain both label classes", flush=True)
        calibration = root / f"{dataset}.calibration.json"
        results = {}
        for command, split in (("fit", "train"), ("evaluate", "dev")):
            trace_path, originals = generated[split]
            output = root / f"{dataset}_{split}_redeep.jsonl"
            _run([sys.executable, "-B", "-m", "src.run_redeep", command, "--input", str(trace_path),
                  "--output", str(output), "--model", args.model, "--calibration", str(calibration),
                  "--batch-size", str(args.score_batch_size), "--progress-every", "16", "--no-token-scores"],
                 logs / f"{dataset}_{command}.log", environment)
            results[command] = {"count": len(originals), "labels": dict(Counter(row["hallucination_label"] for row in originals)),
                                "metrics": _check_scores(output, originals), "output": str(output)}
        results["calibration"] = str(calibration)
        report["datasets"][dataset] = results
        (root / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["status"] = "passed"
    (root / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] All three datasets passed. Report: {root / 'smoke_report.json'}", flush=True)


if __name__ == "__main__":
    main()
