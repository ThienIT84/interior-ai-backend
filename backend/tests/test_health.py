import asyncio

from app.api.v1.endpoints.health import health_check
from app.config import settings
from app.dependencies import get_model_manager


def test_health_endpoint_returns_expected_schema(monkeypatch):
    monkeypatch.setattr(settings, "SEGMENTATION_BACKEND", "sam3_replicate")

    data = asyncio.run(health_check(get_model_manager()))

    assert data["status"] == "healthy"
    assert "device" in data
    assert "segmentation_backend" in data
    assert "segmentation_model" in data
    assert "models" in data
    assert "message" in data
