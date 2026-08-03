import asyncio
import json

import pytest
from redis.exceptions import ConnectionError

from app.main import job_store_unavailable_handler
from app.services import job_service as job_service_module
from app.services.job_service import (
    JobService,
    JobStoreUnavailableError,
    is_redis_available,
)


class StubRedis:
    def __init__(self, fail_on=None):
        self.fail_on = set(fail_on or [])
        self.hashes = {}
        self.expiries = {}

    def _fail(self, operation):
        if operation in self.fail_on:
            raise ConnectionError(f"simulated {operation} failure")

    def ping(self):
        self._fail("ping")
        return True

    def hset(self, key, mapping):
        self._fail("hset")
        self.hashes.setdefault(key, {}).update(
            {name: str(value) for name, value in mapping.items()}
        )
        return len(mapping)

    def expire(self, key, seconds):
        self._fail("expire")
        self.expiries[key] = seconds
        return True

    def hgetall(self, key):
        self._fail("hgetall")
        return dict(self.hashes.get(key, {}))

    def delete(self, key):
        self._fail("delete")
        existed = key in self.hashes
        self.hashes.pop(key, None)
        self.expiries.pop(key, None)
        return int(existed)

    def keys(self, pattern):
        self._fail("keys")
        prefix = pattern.removesuffix("*")
        return [key for key in self.hashes if key.startswith(prefix)]

    def close(self):
        return None


def test_connection_failure_uses_stable_job_store_error():
    with pytest.raises(JobStoreUnavailableError):
        JobService(redis_client=StubRedis(fail_on={"ping"}))


@pytest.mark.parametrize("operation", ["hset", "expire"])
def test_write_failure_uses_stable_job_store_error(operation):
    service = JobService(redis_client=StubRedis(fail_on={operation}))

    with pytest.raises(JobStoreUnavailableError):
        service.create_job("inpainting", {"image_id": "image-1"})


def test_read_failure_uses_stable_job_store_error():
    service = JobService(redis_client=StubRedis(fail_on={"hgetall"}))

    with pytest.raises(JobStoreUnavailableError):
        service.get_job("job-1")


def test_health_probe_reports_unavailable_without_raising(monkeypatch):
    client = StubRedis(fail_on={"ping"})
    monkeypatch.setattr(
        job_service_module.redis,
        "from_url",
        lambda *args, **kwargs: client,
    )

    assert is_redis_available() is False


def test_failed_initialization_is_not_cached(monkeypatch):
    sentinel = object()
    calls = 0

    def recovering_factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise JobStoreUnavailableError()
        return sentinel

    monkeypatch.setattr(job_service_module, "_job_service", None)
    monkeypatch.setattr(job_service_module, "JobService", recovering_factory)

    with pytest.raises(JobStoreUnavailableError):
        job_service_module.get_job_service()

    assert job_service_module.get_job_service() is sentinel
    assert calls == 2


def test_job_round_trip_sets_ttl():
    client = StubRedis()
    service = JobService(redis_client=client)

    job_id = service.create_job("inpainting", {"image_id": "image-1"})
    job = service.get_job(job_id)

    assert job["payload"] == {"image_id": "image-1"}
    assert job["status"] == "pending"
    assert client.expiries[f"interior_job:{job_id}"] == 86400


def test_unavailable_handler_returns_public_503_schema():
    response = asyncio.run(
        job_store_unavailable_handler(None, JobStoreUnavailableError())
    )
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload == {
        "detail": {
            "code": "redis_unavailable",
            "message": "Background job service is temporarily unavailable.",
        }
    }


@pytest.mark.redis
def test_real_redis_create_read_update_and_ttl():
    try:
        service = JobService()
    except JobStoreUnavailableError:
        pytest.skip("Redis integration service is unavailable")

    job_id = service.create_job("integration_test", {"value": 1})
    try:
        service.update_job(job_id, progress=0.5)
        job = service.get_job(job_id)
        ttl = service.redis_client.ttl(f"interior_job:{job_id}")

        assert job["payload"] == {"value": 1}
        assert float(job["progress"]) == 0.5
        assert 0 < ttl <= 86400
    finally:
        service.delete_job(job_id)
