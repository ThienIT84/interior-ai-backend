"""
Job service for managing async job persistence using Redis
"""
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enums"""
    PENDING = "pending"
    PROCESSING = "processing"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class JobService:
    """
    Service for managing async jobs with Redis persistence
    """

    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis connected: {settings.REDIS_URL}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise

    def create_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """
        Create a new job

        Args:
            job_type: Type of job (inpainting, generation, segmentation)
            payload: Job payload data
            user_id: Optional user identifier

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        
        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "user_id": user_id or "anonymous",
            "status": JobStatus.PENDING.value,
            "progress": 0.0,
            "payload": json.dumps(payload),
            "metadata": json.dumps({
                "retries_used": 0,
                "pass2_applied": False,
                "quality_score": None,
            }),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        redis_key = f"{settings.REDIS_KEY_PREFIX}{job_id}"
        self.redis_client.hset(redis_key, mapping=job_data)
        self.redis_client.expire(redis_key, settings.JOB_EXPIRY_SECONDS)
        
        logger.info(f"✅ Job created: {job_id} (type: {job_type})")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job by ID

        Args:
            job_id: Job identifier

        Returns:
            Job data dict or None if not found
        """
        redis_key = f"{settings.REDIS_KEY_PREFIX}{job_id}"
        job_data = self.redis_client.hgetall(redis_key)
        
        if not job_data:
            logger.warning(f"⚠️ Job not found: {job_id}")
            return None

        # Parse JSON fields
        try:
            job_data["payload"] = json.loads(job_data.get("payload", "{}"))
            job_data["metadata"] = json.loads(job_data.get("metadata", "{}"))
        except json.JSONDecodeError:
            logger.warning(f"⚠️ Failed to parse JSON fields for job {job_id}")

        return job_data

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        result_url: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update job status and fields

        Args:
            job_id: Job identifier
            status: New job status
            progress: Progress percentage (0.0-1.0)
            result_url: Result image URL or path
            error: Error message if failed
            metadata: Update metadata dict

        Returns:
            Success flag
        """
        redis_key = f"{settings.REDIS_KEY_PREFIX}{job_id}"
        
        update_data = {
            "updated_at": datetime.utcnow().isoformat(),
        }

        if status:
            update_data["status"] = status
            if status == JobStatus.COMPLETED.value:
                update_data["completed_at"] = datetime.utcnow().isoformat()

        if progress is not None:
            update_data["progress"] = min(max(progress, 0.0), 1.0)

        if result_url is not None:
            update_data["result_url"] = result_url

        if error is not None:
            update_data["error"] = error

        if metadata:
            # Merge with existing metadata
            current_meta = self.get_job(job_id)
            if current_meta:
                merged_meta = current_meta.get("metadata", {})
                merged_meta.update(metadata)
                update_data["metadata"] = json.dumps(merged_meta)

        try:
            self.redis_client.hset(redis_key, mapping=update_data)
            self.redis_client.expire(redis_key, settings.JOB_EXPIRY_SECONDS)
            logger.debug(f"✅ Job updated: {job_id} → {update_data}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update job {job_id}: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job from Redis

        Args:
            job_id: Job identifier

        Returns:
            Success flag
        """
        redis_key = f"{settings.REDIS_KEY_PREFIX}{job_id}"
        deleted = self.redis_client.delete(redis_key)
        if deleted:
            logger.info(f"✅ Job deleted: {job_id}")
        return deleted > 0

    def list_jobs(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering

        Args:
            status: Filter by status (pending, processing, completed, failed)
            user_id: Filter by user ID
            limit: Maximum number of jobs to return

        Returns:
            List of job dicts
        """
        pattern = f"{settings.REDIS_KEY_PREFIX}*"
        keys = self.redis_client.keys(pattern)

        jobs = []
        for key in keys[:limit]:
            job_data = self.redis_client.hgetall(key)
            
            # Apply filters
            if status and job_data.get("status") != status:
                continue
            if user_id and job_data.get("user_id") != user_id:
                continue

            # Parse JSON fields
            try:
                job_data["payload"] = json.loads(job_data.get("payload", "{}"))
                job_data["metadata"] = json.loads(job_data.get("metadata", "{}"))
            except json.JSONDecodeError:
                pass

            jobs.append(job_data)

        logger.info(f"📋 Listed {len(jobs)} jobs (pattern: {pattern}, limit: {limit})")
        return jobs

    def increment_retries(self, job_id: str) -> int:
        """
        Increment retry count for a job

        Args:
            job_id: Job identifier

        Returns:
            New retry count
        """
        job = self.get_job(job_id)
        if not job:
            return 0

        current_retries = job.get("metadata", {}).get("retries_used", 0)
        new_retries = current_retries + 1

        self.update_job(
            job_id,
            metadata={"retries_used": new_retries}
        )

        return new_retries

    def cleanup_old_jobs(self, older_than_seconds: int = 86400) -> int:
        """
        Clean up old completed/failed jobs

        Args:
            older_than_seconds: Delete jobs older than this (default: 24h)

        Returns:
            Number of jobs deleted
        """
        pattern = f"{settings.REDIS_KEY_PREFIX}*"
        keys = self.redis_client.keys(pattern)

        deleted_count = 0
        now = datetime.utcnow()

        for key in keys:
            job_data = self.redis_client.hgetall(key)
            created_at_str = job_data.get("created_at")
            
            if not created_at_str:
                continue

            created_at = datetime.fromisoformat(created_at_str)
            age_seconds = (now - created_at).total_seconds()

            # Only delete completed or failed jobs that are old enough
            status = job_data.get("status")
            if status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                if age_seconds > older_than_seconds:
                    self.delete_job(job_data.get("job_id"))
                    deleted_count += 1

        logger.info(f"🧹 Cleaned up {deleted_count} old jobs (older than {older_than_seconds}s)")
        return deleted_count


# Global job service instance
_job_service: Optional[JobService] = None


def get_job_service() -> JobService:
    """FastAPI dependency to get job service"""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service
