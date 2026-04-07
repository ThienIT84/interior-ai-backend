import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.v1.endpoints.segmentation import segment_with_points_json
from app.config import settings
from app.dependencies import get_model_manager
from app.models.segmentation import SegmentationRequest


def test_segment_points_request_requires_points_field():
    with pytest.raises(ValidationError):
        SegmentationRequest(image_id="missing-points")


def test_segment_points_with_valid_schema_returns_not_found_for_missing_image(monkeypatch):
    monkeypatch.setattr(settings, "SEGMENTATION_BACKEND", "sam3_replicate")

    request = SegmentationRequest(
        image_id="non-existent-image-id",
        points=[{"x": 100, "y": 120, "label": 1}],
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(segment_with_points_json(request, get_model_manager()))

    assert exc.value.status_code == 404
    assert "not found" in str(exc.value.detail).lower()
