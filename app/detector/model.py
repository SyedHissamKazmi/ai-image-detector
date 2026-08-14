from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.core.logging_config import get_logger
from app.detector.ateeq_detector import AteeqDetector
from app.detector.wkaandemir_detector import WkaandemirDetector

logger = get_logger("detector.ensemble")


class AIDetector:
    """Ensemble of Ateeq + wkaandemir detectors, with optional single-model mode."""

    ATEQQ_WEIGHT = 0.47
    WKAANDEMIR_WEIGHT = 0.53

    def __init__(self) -> None:
        only_model = os.getenv("ONLY_MODEL", "").strip().lower()

        if only_model == "ateeq":
            logger.info("Running in single-model mode: Ateeq only")
            self._ateeq = AteeqDetector()
            self._wkaandemir = None
        elif only_model == "wkaandemir":
            logger.info("Running in single-model mode: wkaandemir only")
            self._ateeq = None
            self._wkaandemir = WkaandemirDetector()
        else:
            logger.info("Running in full ensemble mode")
            self._ateeq = AteeqDetector()
            self._wkaandemir = WkaandemirDetector()

    @property
    def available(self) -> bool:
        return bool(self._ateeq) or bool(self._wkaandemir)

    def _run_ateeq(self, path: Path) -> float | None:
        if self._ateeq is None:
            return None
        try:
            return self._ateeq.predict(path)
        except Exception as exc:
            logger.warning("Ateeq detector failed: %s", exc)
            return None

    def _run_wkaandemir(self, path: Path) -> float | None:
        if self._wkaandemir is None:
            return None
        try:
            return self._wkaandemir.predict(path)
        except Exception as exc:
            logger.warning("wkaandemir detector failed: %s", exc)
            return None

    def predict(self, image_path: str | Path) -> float | None:
        path = Path(image_path)
        p1 = self._run_ateeq(path)
        p2 = self._run_wkaandemir(path)

        probabilities = []
        weights = []
        if p1 is not None:
            probabilities.append(p1)
            weights.append(self.ATEQQ_WEIGHT)
        if p2 is not None:
            probabilities.append(p2)
            weights.append(self.WKAANDEMIR_WEIGHT)

        if not probabilities:
            logger.warning("ML_ENSEMBLE_UNAVAILABLE | file=%s", path.name)
            return None

        total_weight = sum(weights)
        if total_weight == 0:
            ensemble = sum(probabilities) / len(probabilities)
        else:
            ensemble = sum(p * w for p, w in zip(probabilities, weights)) / total_weight
        return max(0.0, min(1.0, ensemble))

    def predict_detailed(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        p1 = self._run_ateeq(path)
        p2 = self._run_wkaandemir(path)

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
            "ML_ENSEMBLE_DETAILED | file=%s | ensemble=%.4f | models=%s",
            path.name,
            ensemble if ensemble is not None else -1,
            model_results,
        )
        return {"ensemble": ensemble, "models": model_results}

    async def predict_detailed_async(self, image_path: str | Path) -> dict:
        path = Path(image_path)

        tasks = []
        names = []

        if self._ateeq is not None:
            tasks.append(asyncio.to_thread(self._ateeq.predict, path))
            names.append("ateeq")
        if self._wkaandemir is not None:
            tasks.append(asyncio.to_thread(self._wkaandemir.predict, path))
            names.append("wkaandemir")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        model_results = {}
        probabilities = []
        weights = []

        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning("%s detector failed: %s", name, result)
                continue
            if result is not None:
                model_results[name] = result
                if name == "ateeq":
                    probabilities.append(result)
                    weights.append(self.ATEQQ_WEIGHT)
                else:
                    probabilities.append(result)
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