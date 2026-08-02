import asyncio
from types import SimpleNamespace

from PIL import Image

from app.api.v1.endpoints import generation
from app.config import settings


class FakeGenerationService:
    def __init__(self):
        self.jobs = {}

    def submit_job(self, image, style, model_id="controlnet", guidance_scale=None, steps=None, seed=None):
        job = SimpleNamespace(
            job_id="job-generation",
            status="pending",
            style=style,
            progress=0.0,
            result_url=None,
            result_id=None,
            processing_time=None,
            error=None,
            metadata={"model_id": model_id},
        )
        self.jobs[job.job_id] = job
        return job

    def get_job(self, job_id):
        return self.jobs.get(job_id)


def test_generate_design_submit_and_status_contract(monkeypatch, tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    monkeypatch.setattr(settings, "INPUTS_DIR", inputs_dir)
    monkeypatch.setattr(settings, "OUTPUTS_DIR", tmp_path / "outputs")

    image_id = "demo-image"
    Image.new("RGB", (16, 16), "white").save(inputs_dir / f"demo_{image_id}.jpg")

    fake_service = FakeGenerationService()
    monkeypatch.setattr(
        "app.core.controlnet_generation.get_controlnet_service",
        lambda: fake_service,
    )

    request = generation.GenerateDesignRequest(
        image_id=image_id,
        style="modern",
        model_id="controlnet",
    )
    response = asyncio.run(generation.generate_design(request))

    assert response.job_id == "job-generation"
    assert response.status == "pending"

    status = asyncio.run(generation.get_job_status(response.job_id))
    assert status.job_id == "job-generation"
    assert status.status == "pending"
    assert status.progress == 0.0
