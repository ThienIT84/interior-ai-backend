from typing import Any, Dict, Optional


class FakeJobService:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def create_job(self, job_type, payload, user_id=None):
        self._counter += 1
        job_id = f"job-{self._counter}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "user_id": user_id or "anonymous",
            "status": "pending",
            "progress": 0.0,
            "payload": payload,
            "metadata": {},
        }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        result_id: Optional[str] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        job = self.jobs.setdefault(job_id, {"job_id": job_id, "metadata": {}})
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if result_id is not None:
            job["result_id"] = result_id
        if result_url is not None:
            job["result_url"] = result_url
        if error is not None:
            job["error"] = error
        if metadata:
            job.setdefault("metadata", {}).update(metadata)
        return True

    def get_job(self, job_id):
        return self.jobs.get(job_id)
