"""End-to-end Qwen ReDeEP detector for saved multi-hop traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .calibration import ReDeEPCalibrator
from .io import enrich_record
from .qwen_extractor import QwenRepresentationExtractor
from .scores import calculate_ecs, calculate_pks, mean_features
from ..hallucination_labels import DEFAULT_F1_THRESHOLD
from ..prompt_utils import token_length


class ReDeEPDetector:
    """Extract ECS/PKS and apply the joint ReDeEP calibration."""

    def __init__(
        self,
        model_name: str = "model/Qwen3-4B-Instruct-2507",
        calibrator: Optional[ReDeEPCalibrator] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: str = "auto",
        top_fraction: float = 0.10,
        pks_batch_size: int = 8,
        max_input_tokens: Optional[int] = 4096,
    ):
        self.top_fraction = float(top_fraction)
        self.pks_batch_size = int(pks_batch_size)
        self.max_input_tokens = max_input_tokens
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.pks_batch_size < 1:
            raise ValueError("pks_batch_size must be positive")
        if self.max_input_tokens is not None and int(self.max_input_tokens) < 2:
            raise ValueError("max_input_tokens must be at least 2")
        self.extractor = QwenRepresentationExtractor(
            model_name=model_name,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.calibrator = calibrator

    def _input_too_long(self, parts: Mapping[str, Any], response: str, response_token_ids: Sequence[int]) -> bool:
        """Check the exact saved sequence without changing its evidence."""
        if not self.max_input_tokens:
            return False
        if response_token_ids:
            length = token_length(self.extractor.tokenizer, str(parts["prompt"])) + len(response_token_ids)
        else:
            length = token_length(self.extractor.tokenizer, str(parts["prompt"]) + response)
        return length > int(self.max_input_tokens)

    @staticmethod
    def _mark_unscored(scored: Dict[str, Any], error: str, include_token_scores: bool) -> Dict[str, Any]:
        scored.update(
            {
                "token_ecs": {},
                "token_pks": {},
                "ecs": {},
                "pks": {},
                "redeep_score": None,
                "redeep_prediction": None,
                "redeep_error": error,
                "token_count": 0,
                "context_token_span": [0, 0],
                "response_token_span": [0, 0],
                "prediction_state_span": [0, 0],
            }
        )
        if not include_token_scores:
            for key in ("token_ecs", "token_pks"):
                scored.pop(key, None)
        return scored

    def score_enriched(self, record: Mapping[str, Any], include_token_scores: bool = True) -> Dict[str, Any]:
        response = str(record.get("prediction", ""))
        response_token_ids = record.get("response_token_ids", [])
        response_token_ids = response_token_ids if isinstance(response_token_ids, list) else []
        parts = dict(record["prompt_parts"])
        scored = dict(record)
        scored["prompt_parts"] = parts
        if not parts.get("context", "").strip() or not response.strip():
            return self._mark_unscored(
                scored,
                "missing retrieved evidence or generated response",
                include_token_scores,
            )
        if self._input_too_long(parts, response, response_token_ids):
            return self._mark_unscored(
                scored,
                "saved prompt plus response exceeds max_input_tokens; regenerate with a smaller generator input limit",
                include_token_scores,
            )

        representations = self.extractor.extract_with_parts(
            parts,
            response,
            response_token_ids=response_token_ids or None,
        )
        selected_heads = self.calibrator.selected_heads if self.calibrator else None
        selected_layers = self.calibrator.selected_layers if self.calibrator else None
        parsed_heads = None
        if selected_heads:
            parsed_heads = []
            for key in selected_heads:
                try:
                    layer, head = key.replace("layer_", "").split("_head_")
                    parsed_heads.append((int(layer), int(head)))
                except ValueError:
                    continue
        parsed_layers = None
        if selected_layers:
            parsed_layers = []
            for key in selected_layers:
                try:
                    parsed_layers.append(int(key))
                except (TypeError, ValueError):
                    continue
        if selected_heads and not parsed_heads:
            raise ValueError("Calibration contains no parseable attention-head identifiers")
        if selected_layers and not parsed_layers:
            raise ValueError("Calibration contains no parseable FFN-layer identifiers")
        ecs = calculate_ecs(representations, heads=parsed_heads, top_fraction=self.top_fraction)
        pks = calculate_pks(self.extractor, representations, layers=parsed_layers, batch_size=self.pks_batch_size)
        if parsed_heads and not any(ecs.values()):
            raise RuntimeError("Calibration selected no valid Qwen attention heads for this model")
        if parsed_layers and not any(pks.values()):
            raise RuntimeError("Calibration selected no valid Qwen FFN layers for this model")
        scored["token_ecs"] = ecs
        scored["token_pks"] = pks
        scored["ecs"] = ecs
        scored["pks"] = pks
        scored["ecs"] = mean_features(scored["ecs"])
        scored["pks"] = mean_features(scored["pks"])
        if not scored["ecs"] or not scored["pks"]:
            return self._mark_unscored(scored, "no valid ECS/PKS features were produced", include_token_scores)
        if self.calibrator:
            scored["redeep_score"] = self.calibrator.score(scored)
            scored["redeep_prediction"] = self.calibrator.predict(scored)
        else:
            scored["redeep_score"] = None
            scored["redeep_prediction"] = None
        scored["token_count"] = max(0, int(representations.input_ids.shape[-1]) - representations.response_start)
        scored["context_token_span"] = [representations.context_start, representations.context_end]
        scored["response_token_span"] = [representations.response_start, int(representations.input_ids.shape[-1])]
        scored["prediction_state_span"] = [representations.prediction_start, representations.prediction_end]
        if not include_token_scores:
            scored.pop("token_ecs", None)
            scored.pop("token_pks", None)
        return scored

    def score_records(
        self,
        records: Sequence[Mapping[str, Any]],
        label_mode: str = "f1_answer",
        f1_threshold: float = DEFAULT_F1_THRESHOLD,
        include_token_scores: bool = True,
        batch_size: int = 1,
        progress_every: int = 10,
    ) -> List[Dict[str, Any]]:
        records = list(records)
        output = []
        batch_size = max(1, int(batch_size))
        for start in range(0, len(records), batch_size):
            batch = [enrich_record(dict(record), label_mode=label_mode, f1_threshold=f1_threshold)
                     for record in records[start : start + batch_size]]
            valid = []
            results = [None] * len(batch)
            for index, record in enumerate(batch):
                response = str(record.get("prediction", ""))
                parts = dict(record.get("prompt_parts", {}))
                response_ids = record.get("response_token_ids", [])
                if not parts.get("context", "").strip() or not response.strip():
                    results[index] = self._mark_unscored(dict(record), "missing retrieved evidence or generated response", include_token_scores)
                elif self._input_too_long(parts, response, response_ids if isinstance(response_ids, list) else []):
                    results[index] = self._mark_unscored(dict(record), "saved prompt plus response exceeds max_input_tokens; regenerate with a smaller generator input limit", include_token_scores)
                else:
                    valid.append((index, record))
            if valid:
                representations = self.extractor.extract_batch_with_parts([
                    {"parts": record["prompt_parts"], "response": record.get("prediction", ""),
                     "response_token_ids": record.get("response_token_ids", [])}
                    for _, record in valid
                ])
                for (index, record), representation in zip(valid, representations):
                    results[index] = self._score_representation(record, representation, include_token_scores)
            output.extend(result for result in results if result is not None)
            completed = min(start + batch_size, len(records))
            if completed == len(records) or completed % max(1, int(progress_every)) < batch_size:
                print(f"[redeep] processed {completed}/{len(records)} records ({100.0 * completed / max(1, len(records)):.1f}%)", flush=True)
        return output

    def _score_representation(self, record, representations, include_token_scores):
        scored = dict(record)
        selected_heads = self.calibrator.selected_heads if self.calibrator else None
        selected_layers = self.calibrator.selected_layers if self.calibrator else None
        parsed_heads = None
        if selected_heads:
            parsed_heads = []
            for key in selected_heads:
                try:
                    layer, head = key.replace("layer_", "").split("_head_")
                    parsed_heads.append((int(layer), int(head)))
                except ValueError:
                    continue
        parsed_layers = None
        if selected_layers:
            parsed_layers = []
            for key in selected_layers:
                try:
                    parsed_layers.append(int(key))
                except (TypeError, ValueError):
                    continue
        if selected_heads and not parsed_heads:
            raise ValueError("Calibration contains no parseable attention-head identifiers")
        if selected_layers and not parsed_layers:
            raise ValueError("Calibration contains no parseable FFN-layer identifiers")
        ecs = calculate_ecs(representations, heads=parsed_heads, top_fraction=self.top_fraction)
        pks = calculate_pks(self.extractor, representations, layers=parsed_layers, batch_size=self.pks_batch_size)
        if parsed_heads and not any(ecs.values()):
            raise RuntimeError("Calibration selected no valid Qwen attention heads for this model")
        if parsed_layers and not any(pks.values()):
            raise RuntimeError("Calibration selected no valid Qwen FFN layers for this model")
        scored["token_ecs"], scored["token_pks"] = ecs, pks
        scored["ecs"], scored["pks"] = mean_features(ecs), mean_features(pks)
        if not scored["ecs"] or not scored["pks"]:
            return self._mark_unscored(scored, "no valid ECS/PKS features were produced", include_token_scores)
        scored["redeep_score"] = self.calibrator.score(scored) if self.calibrator else None
        scored["redeep_prediction"] = self.calibrator.predict(scored) if self.calibrator else None
        scored["token_count"] = max(0, int(representations.input_ids.shape[-1]) - representations.response_start)
        scored["context_token_span"] = [representations.context_start, representations.context_end]
        scored["response_token_span"] = [representations.response_start, int(representations.input_ids.shape[-1])]
        scored["prediction_state_span"] = [representations.prediction_start, representations.prediction_end]
        if not include_token_scores:
            scored.pop("token_ecs", None)
            scored.pop("token_pks", None)
        return scored

    def calibrate(
        self,
        records: Sequence[Mapping[str, Any]],
        label_mode: str = "f1_answer",
        f1_threshold: float = DEFAULT_F1_THRESHOLD,
    ) -> ReDeEPCalibrator:
        extracted = []
        for record in records:
            enriched = enrich_record(dict(record), label_mode=label_mode, f1_threshold=f1_threshold)
            extracted.append(self.score_enriched(enriched, include_token_scores=False))
        self.calibrator = ReDeEPCalibrator().fit(extracted)
        return self.calibrator


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(__import__("json").dumps(dict(record), ensure_ascii=False) + "\n")
