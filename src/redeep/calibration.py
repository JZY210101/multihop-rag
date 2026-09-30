"""Validation calibration for the token-level ReDeEP score."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MAX_PAPER_TOP_K = 32


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

    # These are upper bounds.  ``fit`` searches every K in 1..bound, as in
    # the paper, while the feature pools themselves come from the actual model.
    top_heads: int = MAX_PAPER_TOP_K
    top_layers: int = MAX_PAPER_TOP_K
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

    @staticmethod
    def _array_auc(labels: Sequence[int], values: Sequence[float]) -> float:
        """Fast AUC for the K/alpha grid search.

        The ordinary sklearn call is correct but too expensive when the paper's
        32 x 32 feature-count grid is evaluated on a large train set.
        """
        import numpy as np

        y = np.asarray(labels, dtype=np.int8)
        scores = np.asarray(values, dtype=np.float64)
        if y.size == 0 or y.size != scores.size or len(np.unique(y)) < 2:
            return 0.5
        order = np.argsort(scores, kind="mergesort")
        ordered = scores[order]
        ranks = np.empty(scores.size, dtype=np.float64)
        start = 0
        while start < ordered.size:
            end = start + 1
            while end < ordered.size and ordered[end] == ordered[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        positives = y == 1
        n_positive = int(positives.sum())
        n_negative = int(y.size - n_positive)
        return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))

    @staticmethod
    def _matrix(records: Sequence[Mapping[str, Any]], keys: Sequence[str], field: str):
        """Return a dense record x feature matrix using the existing mean rule."""
        import numpy as np

        matrix = np.zeros((len(records), len(keys)), dtype=np.float64)
        for row, record in enumerate(records):
            values = record.get(field, {})
            for column, key in enumerate(keys):
                value = _mean(values.get(key)) if isinstance(values, Mapping) else None
                if value is not None:
                    matrix[row, column] = float(value)
        return matrix

    @staticmethod
    def _usable(records: Optional[Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
        return [
            record
            for record in (records or [])
            if record.get("hallucination_label") is not None and record.get("ecs") and record.get("pks")
        ]

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        validation_records: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "ReDeEPCalibrator":
        """Fit feature counts, coefficient, normalization, and threshold.

        Feature ranking and normalization are learned from ``records``.  When
        ``validation_records`` is supplied, the paper's K/alpha search and the
        classification threshold are selected on that held-out split.  If it is
        absent, the same search falls back to the training records for API
        compatibility and small unit tests.
        """
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
        head_pool = [key for key, auc in ecs_ranked if auc >= self.min_feature_auc]
        layer_pool = [key for key, auc in pks_ranked if auc >= self.min_feature_auc]
        if not head_pool:
            head_pool = [key for key, _ in ecs_ranked[:1]]
        if not layer_pool:
            layer_pool = [key for key, _ in pks_ranked[:1]]

        # The paper searches K=1..32.  A different model may expose fewer
        # layers/heads, so only K is clipped; the candidate pools are not.
        head_limit = min(MAX_PAPER_TOP_K, len(head_pool), max(1, int(self.top_heads)))
        layer_limit = min(MAX_PAPER_TOP_K, len(layer_pool), max(1, int(self.top_layers)))
        head_keys = head_pool[:head_limit]
        layer_keys = layer_pool[:layer_limit]
        validation = self._usable(validation_records)
        if len({int(record["hallucination_label"]) for record in validation}) < 2:
            validation = []
        selection_records = validation or usable
        selection_labels = [int(record["hallucination_label"]) for record in selection_records]

        import numpy as np

        train_ecs = self._matrix(usable, head_keys, "ecs")
        train_pks = self._matrix(usable, layer_keys, "pks")
        select_ecs = self._matrix(selection_records, head_keys, "ecs")
        select_pks = self._matrix(selection_records, layer_keys, "pks")
        train_ecs_prefix = np.cumsum(train_ecs, axis=1)
        train_pks_prefix = np.cumsum(train_pks, axis=1)
        select_ecs_prefix = np.cumsum(select_ecs, axis=1)
        select_pks_prefix = np.cumsum(select_pks, axis=1)

        alpha_candidates = [value / 10.0 for value in range(1, 20)]
        best = None
        for head_count in range(1, head_limit + 1):
            train_ecs_values = train_ecs_prefix[:, head_count - 1] / head_count
            select_ecs_values = select_ecs_prefix[:, head_count - 1] / head_count
            ecs_min, ecs_max = float(train_ecs_values.min()), float(train_ecs_values.max())
            select_ecs_norm = (
                (select_ecs_values - ecs_min) / (ecs_max - ecs_min) if ecs_max > ecs_min else np.zeros_like(select_ecs_values)
            )
            for layer_count in range(1, layer_limit + 1):
                train_pks_values = train_pks_prefix[:, layer_count - 1] / layer_count
                select_pks_values = select_pks_prefix[:, layer_count - 1] / layer_count
                pks_min, pks_max = float(train_pks_values.min()), float(train_pks_values.max())
                select_pks_norm = (
                    (select_pks_values - pks_min) / (pks_max - pks_min) if pks_max > pks_min else np.zeros_like(select_pks_values)
                )
                for alpha in alpha_candidates:
                    score_values = select_pks_norm - alpha * select_ecs_norm
                    auc = self._array_auc(selection_labels, score_values)
                    candidate = (auc, -abs(alpha - 1.0), -head_count, -layer_count)
                    if best is None or candidate > best[0]:
                        best = (candidate, head_count, layer_count, alpha)
        if best is None:
            raise ValueError("Unable to search ReDeEP feature counts")
        _, head_count, layer_count, self.alpha = best
        self.selected_heads = head_keys[:head_count]
        self.selected_layers = layer_keys[:layer_count]

        ecs_values = [self._aggregate({**record, "ecs": record.get("ecs", {}), "pks": {}})[0] for record in usable]
        pks_values = [self._aggregate({**record, "ecs": {}, "pks": record.get("pks", {})})[1] for record in usable]
        self.ecs_min, self.ecs_max = min(ecs_values), max(ecs_values)
        self.pks_min, self.pks_max = min(pks_values), max(pks_values)

        threshold_records = validation or usable
        threshold_scores = [self.score(record) for record in threshold_records]
        threshold_labels = [int(record["hallucination_label"]) for record in threshold_records]
        self.threshold = self._select_threshold(threshold_scores, threshold_labels)
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
