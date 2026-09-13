"""Train-set calibration for the joint ReDeEP score."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_auc(labels: Sequence[int], values: Sequence[float]) -> float:
    """Return a feature AUC, falling back to 0.5 for degenerate data."""
    if len(set(labels)) < 2 or len(set(values)) < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, values))
    except (ImportError, ValueError):
        return 0.5


def _mean(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        numbers = [float(item) for item in value if isinstance(item, (int, float))]
        return sum(numbers) / len(numbers) if numbers else None
    return None


@dataclass
class ReDeEPCalibrator:
    """Select ECS/PKS features and fit the final joint classifier.

    The joint score follows ReDeEP's ``normalized PKS - alpha * normalized
    ECS`` form. Feature selection, alpha, normalization, and the classification
    threshold are fitted on train records and then frozen for evaluation.
    """

    top_heads: int = 8
    top_layers: int = 8
    min_feature_auc: float = 0.0
    selected_heads: List[str] = field(default_factory=list)
    selected_layers: List[str] = field(default_factory=list)
    ecs_min: float = 0.0
    ecs_max: float = 1.0
    pks_min: float = 0.0
    pks_max: float = 1.0
    alpha: float = 1.0
    threshold: float = 0.0
    runtime_settings: Dict[str, Any] = field(default_factory=dict)

    def _aggregate(self, record: Mapping[str, Any], require_all: bool = False) -> Tuple[float, float]:
        ecs = record.get("ecs", {})
        pks = record.get("pks", {})
        ecs_values = [_mean(ecs.get(key)) for key in self.selected_heads]
        pks_values = [_mean(pks.get(key)) for key in self.selected_layers]
        if require_all and (any(value is None for value in ecs_values) or any(value is None for value in pks_values)):
            missing_heads = [key for key, value in zip(self.selected_heads, ecs_values) if value is None]
            missing_layers = [key for key, value in zip(self.selected_layers, pks_values) if value is None]
            raise ValueError(
                f"Record is missing calibrated ReDeEP features: heads={missing_heads}, layers={missing_layers}"
            )
        ecs_values = [value for value in ecs_values if value is not None]
        pks_values = [value for value in pks_values if value is not None]
        return (
            sum(ecs_values) / len(ecs_values) if ecs_values else 0.0,
            sum(pks_values) / len(pks_values) if pks_values else 0.0,
        )

    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0
        return (value - minimum) / (maximum - minimum)

    def _features(self, record: Mapping[str, Any], require_all: bool = False) -> Tuple[float, float]:
        ecs, pks = self._aggregate(record, require_all=require_all)
        return (
            self._normalize(ecs, self.ecs_min, self.ecs_max),
            self._normalize(pks, self.pks_min, self.pks_max),
        )

    @staticmethod
    def _f1_at_threshold(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
        predictions = [int(score >= threshold) for score in scores]
        true_positive = sum(prediction == label == 1 for prediction, label in zip(predictions, labels))
        false_positive = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
        false_negative = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
        denominator = 2 * true_positive + false_positive + false_negative
        return (2 * true_positive / denominator) if denominator else 0.0

    @classmethod
    def _select_threshold(cls, scores: Sequence[float], labels: Sequence[int]) -> float:
        ordered = sorted(set(float(score) for score in scores))
        if len(ordered) == 1:
            return ordered[0]
        candidates = [ordered[0] - 1e-12]
        candidates.extend((left + right) / 2.0 for left, right in zip(ordered, ordered[1:]))
        candidates.append(ordered[-1] + 1e-12)
        return max(candidates, key=lambda value: (cls._f1_at_threshold(scores, labels, value), -abs(value)))

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "ReDeEPCalibrator":
        usable = [
            record
            for record in records
            if record.get("hallucination_label") is not None and record.get("ecs") and record.get("pks")
        ]
        if not usable:
            raise ValueError("No labeled records with ECS and PKS features were provided")
        labels = [int(record["hallucination_label"]) for record in usable]
        if len(set(labels)) < 2:
            raise ValueError("Calibration requires both truthful and hallucinated labels")

        ecs_keys = sorted({key for record in usable for key in record.get("ecs", {})})
        pks_keys = sorted(
            {key for record in usable for key in record.get("pks", {})},
            key=lambda key: int(key) if str(key).isdigit() else str(key),
        )
        if not ecs_keys or not pks_keys:
            raise ValueError("Calibration requires at least one non-empty ECS head and PKS layer")
        # AUC is oriented toward hallucination: 1 - ECS and PKS are positive.
        ecs_ranked = sorted(
            (
                (
                    key,
                    _safe_auc(
                        labels,
                        [-(_mean(record.get("ecs", {}).get(key)) or 0.0) for record in usable],
                    ),
                )
                for key in ecs_keys
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        pks_ranked = sorted(
            (
                (key, _safe_auc(labels, [_mean(record.get("pks", {}).get(key)) or 0.0 for record in usable]))
                for key in pks_keys
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        self.selected_heads = [key for key, auc in ecs_ranked[: max(1, self.top_heads)] if auc >= self.min_feature_auc]
        self.selected_layers = [
            key for key, auc in pks_ranked[: max(1, self.top_layers)] if auc >= self.min_feature_auc
        ]
        if not self.selected_heads:
            self.selected_heads = [key for key, _ in ecs_ranked[:1]]
        if not self.selected_layers:
            self.selected_layers = [key for key, _ in pks_ranked[:1]]

        ecs_values = [self._aggregate({**record, "ecs": record.get("ecs", {}), "pks": {}})[0] for record in usable]
        pks_values = [self._aggregate({**record, "ecs": {}, "pks": record.get("pks", {})})[1] for record in usable]
        self.ecs_min, self.ecs_max = min(ecs_values), max(ecs_values)
        self.pks_min, self.pks_max = min(pks_values), max(pks_values)

        feature_pairs = [self._features(record) for record in usable]
        alpha_candidates = [value / 10.0 for value in range(0, 31)]
        self.alpha = max(
            alpha_candidates,
            key=lambda alpha: (
                _safe_auc(labels, [pks - alpha * ecs for ecs, pks in feature_pairs]),
                -abs(alpha - 1.0),
            ),
        )
        scores = [self.score(record) for record in usable]
        self.threshold = self._select_threshold(scores, labels)
        return self

    def score(self, record: Mapping[str, Any]) -> float:
        ecs, pks = self._features(record, require_all=True)
        return float(pks - self.alpha * ecs)

    def predict(self, record: Mapping[str, Any]) -> int:
        return int(self.score(record) >= self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_heads": self.selected_heads,
            "selected_layers": self.selected_layers,
            "ecs_min": self.ecs_min,
            "ecs_max": self.ecs_max,
            "pks_min": self.pks_min,
            "pks_max": self.pks_max,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "runtime_settings": self.runtime_settings,
            "combination": "normalized_pks_minus_alpha_ecs",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReDeEPCalibrator":
        calibrator = cls()
        combination = data.get("combination")
        if combination not in {None, "normalized_pks_minus_alpha_ecs"}:
            raise ValueError(f"Unsupported ReDeEP calibration combination: {combination}")
        fields = (
            "selected_heads",
            "selected_layers",
            "ecs_min",
            "ecs_max",
            "pks_min",
            "pks_max",
            "alpha",
            "threshold",
            "runtime_settings",
        )
        for field_name in fields:
            if field_name in data:
                setattr(calibrator, field_name, data[field_name])
        calibrator._validate_loaded()
        return calibrator

    def _validate_loaded(self) -> None:
        if not self.selected_heads or not self.selected_layers:
            raise ValueError("Calibration must contain selected_heads and selected_layers")
        if not all(isinstance(key, str) for key in self.selected_heads + self.selected_layers):
            raise ValueError("Calibration feature identifiers must be strings")
        numeric = (self.ecs_min, self.ecs_max, self.pks_min, self.pks_max, self.alpha, self.threshold)
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in numeric):
            raise ValueError("Calibration contains a non-finite numeric value")
        if self.ecs_max < self.ecs_min or self.pks_max < self.pks_min or self.alpha < 0:
            raise ValueError("Calibration normalization ranges or alpha are invalid")
        if not isinstance(self.runtime_settings, dict):
            raise ValueError("Calibration runtime_settings must be a mapping")

    def validate_runtime(self, expected: Mapping[str, Any]) -> None:
        """Reject evaluation settings that differ from train calibration."""
        if not self.runtime_settings:
            raise ValueError("Calibration lacks runtime_settings; rerun fit with the current ReDeEP implementation")
        mismatches = []
        for key, expected_value in expected.items():
            saved_value = self.runtime_settings.get(key)
            if isinstance(expected_value, float) and isinstance(saved_value, (int, float)):
                matches = math.isclose(float(saved_value), expected_value, rel_tol=1e-9, abs_tol=1e-12)
            else:
                matches = saved_value == expected_value
            if not matches:
                mismatches.append(f"{key}: calibration={saved_value!r}, runtime={expected_value!r}")
        if mismatches:
            raise ValueError("Calibration/runtime mismatch: " + "; ".join(mismatches))

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ReDeEPCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
