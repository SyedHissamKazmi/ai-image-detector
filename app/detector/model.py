from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.logging_config import get_logger
from app.detector.ateeq_detector import AteeqDetector
from app.detector.wkaandemir_detector import WkaandemirDetector

logger = get_logger("detector.ensemble")


class AIDetector:
    """Ensemble of Ateeq + wkaandemir detectors."""

    ATEQQ_WEIGHT = 0.47
    WKAANDEMIR_WEIGHT = 0.53

    def __init__(self) -> None:
        self._ateeq = AteeqDetector()
        self._wkaandemir = WkaandemirDetector()

    @property
    def available(self) -> bool:
        return self._ateeq.available or self._wkaandemir.available

    # ------------------------------------------------------------
    # Synchronous predict (kept for backward compatibility / tests)
    # ------------------------------------------------------------
    def predict(self, image_path: str | Path) -> float | None:
        path = Path(image_path)

        probabilities = []
        weights = []

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

    # ------------------------------------------------------------
    # Detailed synchronous predict (returns per‑model probabilities)
    # ------------------------------------------------------------
    def predict_detailed(self, image_path: str | Path) -> dict:
        path = Path(image_path)

        model_results = {}
        probabilities = []
        weights = []

        try:
            p1 = self._ateeq.predict(path)
            if p1 is not None:
                probabilities.append(p1)
                weights.append(self.ATEQQ_WEIGHT)
                model_results["ateeq"] = p1
        except Exception as exc:
            logger.warning("Ateeq detector failed: %s", exc)

        try:
            p2 = self._wkaandemir.predict(path)
            if p2 is not None:
                probabilities.append(p2)
                weights.append(self.WKAANDEMIR_WEIGHT)
                model_results["wkaandemir"] = p2
        except Exception as exc:
            logger.warning("wkaandemir detector failed: %s", exc)

        if not probabilities:
            ensemble = None
        else:
            total_weight = sum(weights)
            if total_weight == 0:
                ensemble = sum(probabilities) / len(probabilities)
            else:
                ensemble = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
            ensemble = max(0.0, min(1.0, ensemble))

        logger.info(
            "ML_ENSEMBLE_DETAILED | file=%s | ensemble=%.4f | models=%s",
            path.name,
            ensemble if ensemble is not None else -1,
            model_results,
        )

        return {"ensemble": ensemble, "models": model_results}

    # ------------------------------------------------------------
    # Asynchronous predict detailed (runs both models concurrently)
    # ------------------------------------------------------------
    async def predict_detailed_async(self, image_path: str | Path) -> dict:
        path = Path(image_path)

        # Run both models in threadpool concurrently
        ateeq_task = asyncio.to_thread(self._ateeq.predict, path)
        wkaandemir_task = asyncio.to_thread(self._wkaandemir.predict, path)

        p1, p2 = await asyncio.gather(ateeq_task, wkaandemir_task)

        model_results = {}
        probabilities = []
        weights = []

        if p1 is not None:
            model_results["ateeq"] = p1
            probabilities.append(p1)
            weights.append(self.ATEQQ_WEIGHT)

        if p2 is not None:
            model_results["wkaandemir"] = p2
            probabilities.append(p2)
            weights.append(self.WKAANDEMIR_WEIGHT)

        if not probabilities:
            ensemble = None
        else:
            total_weight = sum(weights)
            if total_weight == 0:
                ensemble = sum(probabilities) / len(probabilities)
            else:
                ensemble = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
            ensemble = max(0.0, min(1.0, ensemble))

        logger.info(
            "ML_ENSEMBLE_DETAILED_ASYNC | file=%s | ensemble=%.4f | models=%s",
            path.name,
            ensemble if ensemble is not None else -1,
            model_results,
        )

        return {"ensemble": ensemble, "models": model_results}