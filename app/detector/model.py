from __future__ import annotations

from pathlib import Path

from app.core.logging_config import get_logger
from app.detector.ateeq_detector import AteeqDetector
from app.detector.wkaandemir_detector import WkaandemirDetector

logger = get_logger("detector.ensemble")


class AIDetector:
    """Ensemble of Ateeq + wkaandemir detectors."""

    ATEQQ_WEIGHT = 0.50
    WKAANDEMIR_WEIGHT = 0.50

    def __init__(self) -> None:
        self._ateeq = AteeqDetector()
        self._wkaandemir = WkaandemirDetector()

    @property
    def available(self) -> bool:
        return self._ateeq.available or self._wkaandemir.available

    def predict(self, image_path: str | Path) -> float | None:
        path = Path(image_path)

        probabilities = []
        weights = []

        # Run each model independently
        try:
            p1 = self._ateeq.predict(path)
            if p1 is not None:
                probabilities.append(p1)
                weights.append(self.ATEQQ_WEIGHT)
        except Exception as exc:
            logger.warning("Ateeq detector failed: %s", exc)

        try:
            p2 = self._wkaandemir.predict(path)
            if p2 is not None:
                probabilities.append(p2)
                weights.append(self.WKAANDEMIR_WEIGHT)
        except Exception as exc:
            logger.warning("wkaandemir detector failed: %s", exc)

        if not probabilities:
            logger.warning("ML_ENSEMBLE_UNAVAILABLE | file=%s", path.name)
            return None

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            ensemble = sum(probabilities) / len(probabilities)
        else:
            ensemble = sum(p * w for p, w in zip(probabilities, weights)) / total_weight

        ensemble = max(0.0, min(1.0, ensemble))

        logger.info(
            "ML_ENSEMBLE_SUCCESS | file=%s | models=%d | ai_probability=%.4f",
            path.name,
            len(probabilities),
            ensemble,
        )

        return ensemble