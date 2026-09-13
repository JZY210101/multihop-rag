"""Run Qwen ReDeEP on multi-hop traces.

Typical workflow:

    python -m src.run_redeep fit --input outputs/hotpotqa_train.json
    python -m src.run_redeep evaluate --input outputs/hotpotqa_dev.json \
        --calibration outputs/redeep_hotpotqa.json

The detector performs a full teacher-forced forward pass for each record.  A
GPU and enough memory for the selected Qwen checkpoint are therefore required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .redeep.calibration import ReDeEPCalibrator
from .redeep.detector import ReDeEPDetector, write_jsonl
from .redeep.io import enrich_record, normalize_label_mode, read_records
from .hallucination_labels import DEFAULT_F1_THRESHOLD


LABEL_MODES = ("f1_answer", "f1_retrieval_aware", "weak_answer", "retrieval_aware")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen ReDeEP multi-hop hallucination detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default=None, help="Optional ReDeEP YAML config; CLI arguments take priority")
        sub.add_argument("--input", required=True, help="JSON or JSONL multi-hop trace file")
        sub.add_argument("--output", required=True, help="JSONL score output")
        sub.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
        sub.add_argument("--label-mode", choices=LABEL_MODES, default="f1_answer")
        sub.add_argument(
            "--f1-threshold",
            type=float,
            default=DEFAULT_F1_THRESHOLD,
            help="F1 <= threshold is labeled hallucination (default: 0.30)",
        )
        sub.add_argument("--device", default=None)
        sub.add_argument("--device-map", default="auto")
        sub.add_argument("--torch-dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
        sub.add_argument("--top-heads", type=int, default=8)
        sub.add_argument("--top-layers", type=int, default=8)
        sub.add_argument("--top-fraction", type=float, default=0.10, help="Context-token fraction used by token ECS")
        sub.add_argument("--pks-batch-size", type=int, default=8)
        sub.add_argument("--granularity", choices=("token", "chunk"), default="token")
        sub.add_argument("--chunk-size", type=int, default=400)
        sub.add_argument("--embedding-model", default=None, help="For chunk ECS, e.g. BAAI/bge-base-en-v1.5")
        sub.add_argument(
            "--max-input-tokens",
            type=int,
            default=4096,
            help="Reject longer saved prompt+response sequences to bound attention memory",
        )
        sub.add_argument(
            "--max-records",
            type=int,
            default=None,
            help="Optional prefix limit for smoke tests or calibration subsets",
        )
        sub.add_argument("--calibration", help="Calibration JSON (required by evaluate, written by fit)")
        sub.add_argument("--no-token-scores", action="store_true", help="Do not write per-token feature arrays")
    return parser


def _apply_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Reading --config requires pyyaml") from exc
        with Path(args.config).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Expected a YAML mapping in {args.config}")

        aliases = {"model_name": "model"}
        allowed = {
            "model",
            "label_mode",
            "device",
            "device_map",
            "torch_dtype",
            "top_heads",
            "top_layers",
            "top_fraction",
            "pks_batch_size",
            "granularity",
            "chunk_size",
            "embedding_model",
            "max_input_tokens",
            "max_records",
            "f1_threshold",
        }
        provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        for source_key, value in values.items():
            destination = aliases.get(source_key, source_key)
            if destination not in allowed:
                raise ValueError(f"Unknown ReDeEP config key: {source_key}")
            flag = "--" + destination.replace("_", "-")
            if flag not in provided_flags:
                setattr(args, destination, value)
    if args.label_mode not in LABEL_MODES:
        raise ValueError(f"Invalid label_mode: {args.label_mode}")
    if args.granularity not in {"token", "chunk"}:
        raise ValueError(f"Invalid granularity: {args.granularity}")
    if args.torch_dtype not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"Invalid torch_dtype: {args.torch_dtype}")
    if not 0.0 < float(args.top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if not 0.0 <= float(args.f1_threshold) <= 1.0:
        raise ValueError("f1_threshold must be in [0, 1]")
    if int(args.top_heads) < 1 or int(args.top_layers) < 1:
        raise ValueError("top_heads and top_layers must be positive")
    if args.max_input_tokens is not None and int(args.max_input_tokens) < 2:
        raise ValueError("max_input_tokens must be at least 2")
    return args


def _metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [
        record
        for record in records
        if record.get("hallucination_label") is not None and record.get("redeep_score") is not None
    ]
    if not labeled:
        return {"count": 0, "warning": "No labeled records available"}
    labels = [int(record["hallucination_label"]) for record in labeled]
    scores = [float(record["redeep_score"]) for record in labeled]
    result: Dict[str, Any] = {"count": len(labeled), "positive_rate": sum(labels) / len(labels)}
    if len(set(labels)) < 2:
        result["warning"] = "Only one class is present; AUC/F1 are undefined"
        return result
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        predictions = [int(record.get("redeep_prediction", 0)) for record in labeled]
        result.update(
            {
                "roc_auc": float(roc_auc_score(labels, scores)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
            }
        )
    except ImportError:
        result["warning"] = "Install scikit-learn for evaluation metrics"
    if len(set(scores)) >= 2:
        try:
            from scipy.stats import pearsonr

            result["pcc"] = float(pearsonr(labels, scores).statistic)
        except (ImportError, ValueError):
            pass
    by_hop = {}
    for hop in sorted({record.get("hop_num") for record in labeled if record.get("hop_num") is not None}):
        subset = [record for record in labeled if record.get("hop_num") == hop]
        subset_labels = [int(record["hallucination_label"]) for record in subset]
        subset_scores = [float(record["redeep_score"]) for record in subset]
        if len(set(subset_labels)) >= 2:
            try:
                from sklearn.metrics import roc_auc_score

                by_hop[str(hop)] = {"count": len(subset), "roc_auc": float(roc_auc_score(subset_labels, subset_scores))}
            except (ImportError, ValueError):
                pass
        else:
            by_hop[str(hop)] = {"count": len(subset), "warning": "Only one class is present"}
    if by_hop:
        result["by_hop"] = by_hop
    return result


def _runtime_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": str(args.model),
        "granularity": str(args.granularity),
        "top_fraction": float(args.top_fraction),
        "chunk_size": int(args.chunk_size) if args.granularity == "chunk" else None,
        "embedding_model": args.embedding_model if args.granularity == "chunk" else None,
        "label_mode": normalize_label_mode(str(args.label_mode)),
        "f1_threshold": float(args.f1_threshold),
    }


def main() -> None:
    args = _apply_yaml_config(_parser().parse_args())
    raw_records = read_records(args.input)
    if args.max_records is not None:
        raw_records = raw_records[: max(0, args.max_records)]
    if not raw_records:
        raise SystemExit("No input records were found after applying --max-records")
    enriched = None
    loaded_calibrator = None
    if args.command == "fit":
        enriched = [
            enrich_record(record, label_mode=args.label_mode, f1_threshold=args.f1_threshold) for record in raw_records
        ]
        labels = {record["hallucination_label"] for record in enriched if record.get("hallucination_label") is not None}
        if labels != {0, 1}:
            raise SystemExit(
                "fit requires both label classes after F1/retrieval filtering; "
                "increase the calibration sample or check generated answers"
            )
    else:
        if not args.calibration:
            raise SystemExit("evaluate requires --calibration")
        loaded_calibrator = ReDeEPCalibrator.load(args.calibration)
        loaded_calibrator.validate_runtime(_runtime_settings(args))
    device_map = args.device_map
    if device_map is not None and str(device_map).lower() in {"none", "null", "false"}:
        device_map = None
    detector = ReDeEPDetector(
        model_name=args.model,
        device=args.device,
        device_map=device_map,
        torch_dtype=args.torch_dtype,
        top_fraction=args.top_fraction,
        pks_batch_size=args.pks_batch_size,
        granularity=args.granularity,
        chunk_size=args.chunk_size,
        embedding_model=args.embedding_model,
        max_input_tokens=args.max_input_tokens,
        calibrator=loaded_calibrator,
    )
    if args.command == "fit":
        assert enriched is not None
        extracted = [
            detector.score_enriched(record, include_token_scores=not args.no_token_scores) for record in enriched
        ]
        calibrator = ReDeEPCalibrator(top_heads=args.top_heads, top_layers=args.top_layers).fit(extracted)
        calibrator.runtime_settings = _runtime_settings(args)
        calibration_path = args.calibration or str(Path(args.output).with_suffix(".calibration.json"))
        calibrator.save(calibration_path)
        detector.calibrator = calibrator
        # Features were already extracted above.  Apply the fitted joint
        # calibration in memory instead of running a second Qwen forward pass.
        scored = []
        for record in extracted:
            result = dict(record)
            if result.get("ecs") and result.get("pks"):
                result["redeep_score"] = calibrator.score(result)
                result["redeep_prediction"] = calibrator.predict(result)
            else:
                result["redeep_score"] = None
                result["redeep_prediction"] = None
            if args.no_token_scores:
                for key in ("token_ecs", "token_pks"):
                    result.pop(key, None)
            scored.append(result)
        print(json.dumps({"calibration": calibration_path, "metrics": _metrics(scored)}, ensure_ascii=False))
    else:
        detector.calibrator = loaded_calibrator
        scored = detector.score_records(
            raw_records,
            label_mode=args.label_mode,
            f1_threshold=args.f1_threshold,
            include_token_scores=not args.no_token_scores,
        )
        print(json.dumps({"metrics": _metrics(scored)}, ensure_ascii=False))
    write_jsonl(scored, args.output)


if __name__ == "__main__":
    main()
