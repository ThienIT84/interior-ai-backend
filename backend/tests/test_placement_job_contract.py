import asyncio

from PIL import Image

from app.api.v1.endpoints import generation
from app.config import settings
from tests.fakes import FakeJobService


class NoopThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        return None


def test_place_furniture_submit_and_status_contract(monkeypatch, tmp_path):
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    monkeypatch.setattr(settings, "INPUTS_DIR", inputs_dir)
    monkeypatch.setattr(settings, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(generation.threading, "Thread", NoopThread)

    image_id = "demo-room"
    Image.new("RGB", (16, 16), "white").save(inputs_dir / f"demo_{image_id}.jpg")

    fake_job_service = FakeJobService()
    request = generation.PlaceFurnitureRequest(
        image_id=image_id,
        bbox_x=0.1,
        bbox_y=0.1,
        bbox_w=0.5,
        bbox_h=0.5,
        furniture_description="a modern chair",
    )

    response = asyncio.run(
        generation.place_furniture(
            request=request,
            job_service=fake_job_service,
        )
    )

    assert response.job_id.startswith("job-")
    assert response.status == "pending"

    status = asyncio.run(
        generation.get_placement_job_status(
            job_id=response.job_id,
            job_service=fake_job_service,
        )
    )

    assert status.job_id == response.job_id
    assert status.status == "pending"
    assert status.furniture_description == "a modern chair"
