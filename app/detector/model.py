from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from app.core.logging_config import get_logger

logger = get_logger("detector")


class AIDetector:
    """
    Hugging Face image classifier for AI-generated vs human-created images.

    The model is loaded lazily on the first prediction rather than during
    application startup. This keeps FastAPI startup fast and allows the
    application to continue running even if the ML dependencies/model
    download are unavailable.
    """

    MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"

    AI_LABELS = {
        "ai-generated",
        "ai generated",
        "ai-generated image",
        "artificial",
        "synthetic",
        "fake",
        "ai",
    }

    HUMAN_LABELS = {
        "human",
        "real",
        "real image",
        "human-generated",
        "human generated",
    }

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_attempted = False
        self._available = False
        self._lock = Lock()

    @property
    def available(self) -> bool:
        """Return whether the ML detector has successfully loaded."""
        return self._available

    def _load_model(self) -> bool:
        """
        Load the Hugging Face pipeline once.

        Returns True if loading succeeds, otherwise False.
        """
        if self._load_attempted:
            return self._available

        with self._lock:
            if self._load_attempted:
                return self._available

            self._load_attempted = True

            try:
                # Imported lazily so Stage 2 can still start if ML packages
                # have not been installed yet.
                from transformers import pipeline

                logger.info(
                    "ML_MODEL_LOADING | model=%s | device=CPU",
                    self.MODEL_NAME,
                )

                self._pipeline = pipeline(
                    task="image-classification",
                    model=self.MODEL_NAME,
                    device=-1,  # CPU
                )

                self._available = True

                logger.info(
                    "ML_MODEL_READY | model=%s | device=CPU",
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

    @staticmethod
    def _normalise_label(label: str) -> str:
        return " ".join(label.strip().lower().replace("_", " ").split())

    def _extract_ai_probability(
        self,
        predictions: list[dict[str, Any]],
    ) -> float | None:
        """
        Convert Hugging Face classification results into one AI probability.

        The selected model uses:
            human
            AI-generated

        We also support a few equivalent labels so the detector is easier
        to extend to another compatible model later.
        """
        ai_probability = 0.0
        found_ai_label = False
        found_human_label = False

        for prediction in predictions:
            label = self._normalise_label(str(prediction.get("label", "")))

            try:
                score = float(prediction["score"])
            except (KeyError, TypeError, ValueError):
                continue

            if not 0.0 <= score <= 1.0:
                continue

            if label in self.AI_LABELS:
                ai_probability += score
                found_ai_label = True

            elif label in self.HUMAN_LABELS:
                found_human_label = True

        if found_ai_label:
            return min(max(ai_probability, 0.0), 1.0)

        if found_human_label:
            # Binary classifier returned a human probability but no
            # recognised AI label.
            human_probability = sum(
                float(item["score"])
                for item in predictions
                if self._normalise_label(str(item.get("label", "")))
                in self.HUMAN_LABELS
                and isinstance(item.get("score"), (int, float))
            )

            return min(max(1.0 - human_probability, 0.0), 1.0)

        logger.warning(
            "ML_PREDICTION_UNKNOWN_LABELS | predictions=%s",
            predictions,
        )
        return None

    def predict(self, image_path: str | Path) -> float | None:
        """
        Predict AI-generation probability.

        Returns:
            float: AI probability from 0.0 to 1.0
            None: detector unavailable or prediction failed
        """
        path = Path(image_path)

        if not path.is_file():
            logger.warning(
                "ML_PREDICTION_FAILED | file=%s | reason=file_not_found",
                path.name,
            )
            return None

        if not self._load_model():
            return None

        try:
            # top_k=2 is enough for this binary classifier and avoids
            # unnecessary output.
            predictions = self._pipeline(
                str(path),
                top_k=2,
            )

            if not isinstance(predictions, list):
                logger.warning(
                    "ML_PREDICTION_FAILED | file=%s | reason=unexpected_output",
                    path.name,
                )
                return None

            probability = self._extract_ai_probability(predictions)

            if probability is None:
                return None

            logger.info(
                "ML_PREDICTION_SUCCESS | file=%s | ai_probability=%.4f",
                path.name,
                probability,
            )

            return probability

        except Exception as exc:
            logger.warning(
                "ML_PREDICTION_FAILED | file=%s | error=%s",
                path.name,
                exc,
            )
            return None