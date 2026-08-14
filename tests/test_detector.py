import pytest
from pathlib import Path

from app.detector.model import AIDetector


class MockDetector:
    def __init__(self, prob):
        self.prob = prob

    def predict(self, path):
        return self.prob


def test_ensemble_uses_only_available_models(tmp_path,monkeypatch):
    # Create a dummy file so path.is_file() passes.
    dummy = tmp_path / "img.jpg"
    dummy.write_bytes(b"fake")

    """If one model returns None, the other should be used."""
    detector = AIDetector()

    # Replace both sub-detectors with mocks.
    detector._ateeq = MockDetector(0.8)
    detector._wkaandemir = MockDetector(None)  # wkaandemir unavailable

    result = detector.predict("dummy_path")
    assert result == 0.8


def test_ensemble_weighted_average(tmp_path):
    dummy = tmp_path / "img.jpg"
    dummy.write_bytes(b"fake")

    detector = AIDetector()
    detector._ateeq = MockDetector(0.9)
    detector._wkaandemir = MockDetector(0.7)

    result = detector.predict("dummy_path")
    # Weighted average => (0.9 * 0.47 + 0.7 * 0.53) / (0.47 + 0.53) = 0.794
    assert result == 0.794


def test_ensemble_none_when_all_unavailable(tmp_path):
    dummy = tmp_path / "img.jpg"
    dummy.write_bytes(b"fake")

    detector = AIDetector()
    detector._ateeq = MockDetector(None)
    detector._wkaandemir = MockDetector(None)

    result = detector.predict("dummy_path")
    assert result is None