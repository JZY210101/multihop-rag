"""Qwen-compatible ReDeEP hallucination detection components.

The package intentionally contains only detection and calibration logic.  It
does not include the AARF generation intervention from the original project.
"""

from .calibration import ReDeEPCalibrator
from .detector import ReDeEPDetector
from .qwen_extractor import QwenRepresentationExtractor

__all__ = ["ReDeEPCalibrator", "ReDeEPDetector", "QwenRepresentationExtractor"]
