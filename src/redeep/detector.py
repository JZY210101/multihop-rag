"""End-to-end Qwen ReDeEP detector for saved multi-hop traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .calibration import ReDeEPCalibrator
from .chunk_scores import calculate_chunk_scores
from .io import enrich_record
from .qwen_extractor import QwenRepresentationExtractor
from .scores import calculate_ecs, calculate_pks, mean_features
from ..hallucination_labels import DEFAULT_F1_THRESHOLD
from ..prompt_utils import token_length


class ReDeEPDetector:
    """Extract ECS/PKS and apply the joint ReDeEP calibration."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        calibrator: Optional[ReDeEPCalibrator] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = "auto",
        torch_dtype: str = "auto",
        top_fraction: float = 0.10,
        pks_batch_size: int = 8,
        granularity: str = "token",
        chunk_size: int = 400,
        embedding_model: Optional[str] = None,
        max_input_tokens: Optional[int] = 4096,
    ):
        self.top_fraction = float(top_fraction)
        self.pks_batch_size = int(pks_batch_size)
        if granularity not in {"token", "chunk"}:
            raise ValueError("granularity must be 'token' or 'chunk'")
        self.granularity = granularity
        self.chunk_size = int(chunk_size)
        self.max_input_tokens = max_input_tokens
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if self.pks_batch_size < 1 or self.chunk_size < 1:
            raise ValueError("pks_batch_size and chunk_size must be positive")
        if self.max_input_tokens is not None and int(self.max_input_tokens) < 2:
            raise ValueError("max_input_tokens must be at least 2")
        self.extractor = QwenRepresentationExtractor(
            model_name=model_name,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.calibrator = calibrator
        self.embedding_model_name = embedding_model
        self._embedder = None

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

    def _get_embedder(self):
        if self.embedding_model_name is None:
            return None
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Chunk-level ReDeEP with an embedding model requires sentence-transformers") from exc
            device = self.extractor.device if self.extractor.device else None
            self._embedder = SentenceTransformer(self.embedding_model_name, device=device)
        return self._embedder

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
        if self.granularity == "chunk":
            chunk_response = response
            if response_token_ids:
                chunk_response = self.extractor.tokenizer.decode(
                    response_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            chunk = calculate_chunk_scores(
                representations,
                self.extractor.tokenizer,
                parts,
                chunk_response,
                pks,
                heads=parsed_heads,
                chunk_size=self.chunk_size,
                embedder=self._get_embedder(),
            )
            scored["ecs"] = chunk["ecs"]
            scored["pks"] = chunk["pks"]
            scored["chunks"] = chunk["chunks"]
            scored["embedding_backend"] = chunk["embedding_backend"]
        else:
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
    ) -> List[Dict[str, Any]]:
        output = []
        for record in records:
            enriched = enrich_record(dict(record), label_mode=label_mode, f1_threshold=f1_threshold)
            output.append(self.score_enriched(enriched, include_token_scores=include_token_scores))
        return output

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
