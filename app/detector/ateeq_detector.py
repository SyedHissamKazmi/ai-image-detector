from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from app.core.logging_config import get_logger
from app.detector.base import BaseDetector

logger = get_logger("detector.ateeq")


class AteeqDetector(BaseDetector):
    MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_attempted = False
        self._available = False
        self._lock = Lock()

    @property
    def name(self) -> str:
        return "ateeq"

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> bool:
        if self._load_attempted:
            return self._available

        with self._lock:
            if self._load_attempted:
                return self._available

            self._load_attempted = True

            try:
                from transformers import pipeline

                logger.info(
                    "ML_MODEL_LOADING | model=%s | architecture=SigLIP | device=CPU",
                    self.MODEL_NAME,
                )

                self._pipeline = pipeline(
                    task="image-classification",
                    model=self.MODEL_NAME,
                    device=-1,
                )

                self._available = True

                logger.info(
                    "ML_MODEL_READY | model=%s | architecture=SigLIP | device=CPU",
                    self.MODEL_NAME,
                )

            except Exception as exc:
                self._pipeline = None
                self._available = False
                logger.warning(
                    "ML_MODEL_UNAVAILABLE | model=%s | error=%s",
                    self.MODEL_NAME,
                    exc,
                )

            return self._available

    def predict(self, image_path: str | Path) -> float | None:
        path = Path(image_path)

        if not path.is_file():
            logger.warning(
                "ML_PREDICTION_FAILED | model=%s | file=%s | reason=not_found",
                self.MODEL_NAME,
                path.name,
            )
            return None

        if not self._load():
            return None

        try:
            results = self._pipeline(str(path), top_k=2)

            if not isinstance(results, list):
                logger.warning(
                    "ML_PREDICTION_FAILED | model=%s | file=%s | reason=bad_output",
                    self.MODEL_NAME,
                    path.name,
                )
                return None

            ai_probability = None

            for result in results:
                label = str(result.get("label", "")).strip().lower()
                score = float(result.get("score", 0.0))

                if label == "ai":
                    ai_probability = score
                    break
                elif label == "hum":
                    ai_probability = 1.0 - score
                    break

            if ai_probability is None:
                logger.warning(
                    "ML_PREDICTION_FAILED | model=%s | file=%s | reason=labels_not_recognized | output=%s",
                    self.MODEL_NAME,
                    path.name,
                    results,
                )
                return None

            ai_probability = max(0.0, min(1.0, ai_probability))

            logger.info(
                "ML_MODEL_PREDICTION | model=%s | file=%s | ai_probability=%.4f",
                self.MODEL_NAME,
                path.name,
                ai_probability,
            )

            return ai_probability

        except Exception as exc:
            logger.warning(
                "ML_PREDICTION_FAILED | model=%s | file=%s | error=%s",
                self.MODEL_NAME,
                path.name,
                exc,
            )
            return None