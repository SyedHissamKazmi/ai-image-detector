from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import analyzer


def _make_test_image() -> BytesIO:
    """Create a tiny in‑memory JPEG image."""
    img = Image.new("RGB", (100, 100), color=(120, 30, 200))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_root_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_analyze_with_mock_detector(client, monkeypatch):
    """
    Test the /analyze endpoint with a stubbed detector.
    This avoids model loading and runs quickly.
    """

    # Create a fake detector that always returns 0.5 AI probability.
    class MockDetector:
        async def predict_detailed_async(self, image_path):
            return {"ensemble": 0.5, "models": {"mock": 0.5}}

    # Replace the real detector in the analyzer module with our mock.
    monkeypatch.setattr(analyzer, "detector", MockDetector())

    image_file = _make_test_image()
    response = client.post(
        "/analyze",
        files={"file": ("test.jpg", image_file, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["ai_probability"] == 0.5
    assert data["human_probability"] == 0.5
    assert data["confidence"] == "LOW"

    assert "ML ensemble: uncertain" in data["signals"]
    assert "ML ensemble: strong AI signal" not in data["signals"]

    assert data["dominant_colors"]