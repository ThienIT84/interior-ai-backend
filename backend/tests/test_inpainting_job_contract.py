import asyncio

from fastapi import BackgroundTasks

from app.api.v1.endpoints.inpainting import InpaintRequest, remove_object_async


class FakeJobService:
    def __init__(self):
        self.jobs = {}
        self._counter = 0

    def create_job(self, job_type, payload, user_id=None):
        self._counter += 1
        job_id = f"job-{self._counter}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "progress": 0.0,
            "payload": payload,
            "user_id": user_id or "anonymous",
        }
        return job_id

    def update_job(self, job_id, **kwargs):
        self.jobs.setdefault(job_id, {"job_id": job_id})
        self.jobs[job_id].update(kwargs)
        return True

    def get_job(self, job_id):
        return self.jobs.get(job_id)


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
