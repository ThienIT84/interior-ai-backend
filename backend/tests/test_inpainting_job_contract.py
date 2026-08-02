import asyncio

from fastapi import BackgroundTasks

from app.api.v1.endpoints.inpainting import InpaintRequest, remove_object_async
from tests.fakes import FakeJobService


def test_remove_object_async_returns_job_contract():
    fake_job_service = FakeJobService()
    request = InpaintRequest(image_id="demo-image", mask_id="demo-mask")

    response = asyncio.run(
        remove_object_async(
            request=request,
            background_tasks=BackgroundTasks(),
            job_service=fake_job_service,
        )
    )

    assert response.job_id.startswith("job-")
    assert response.status == "pending"
    assert "submitted" in response.message.lower()
