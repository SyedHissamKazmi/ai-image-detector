from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDetector(ABC):
    """Common interface for all AI-image detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name for this detector."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the detector has successfully loaded."""

    @abstractmethod
    def predict(self, image_path: str | Path) -> float | None:
        """
        Return AI probability (0.0 to 1.0) or None if unavailable/failed.
        """