"""
Job executor for handling timeouts, retries, and error recovery
"""
import asyncio
import time
from typing import Callable, Any, Optional, Coroutine
from app.config import settings
from app.services.job_service import JobService, JobStatus
import logging

logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Handles job execution with timeout enforcement and retry logic
    """

    def __init__(self, job_service: JobService):
        """Initialize executor with job service"""
        self.job_service = job_service

    async def execute_with_timeout(
        self,
        job_id: str,
        task_name: str,
        async_func: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Execute async function with timeout and retry logic

        Args:
            job_id: Job identifier for tracking
            task_name: Human-readable task name (e.g., "remove_object_pass1")
            async_func: Async function to execute
            *args: Positional arguments for async_func
            max_retries: Max retry attempts (default: settings.MAX_RETRIES)
            timeout_seconds: Timeout for execution (default: settings.REQUEST_TIMEOUT_SECONDS)
            **kwargs: Keyword arguments for async_func

        Returns:
            Result from async_func or None if failed after retries
        """
        max_retries = max_retries or settings.MAX_RETRIES
        timeout_seconds = timeout_seconds or settings.REQUEST_TIMEOUT_SECONDS

        attempt = 0
        last_error = None

        while attempt <= max_retries:
            attempt += 1
            job = self.job_service.get_job(job_id)
            
            if not job:
                logger.error(f"❌ Job not found: {job_id}")
                return None

            current_retries = job.get("metadata", {}).get("retries_used", 0)
            
            try:
                # Update job status to processing/retrying
                status = JobStatus.RETRYING.value if attempt > 1 else JobStatus.PROCESSING.value
                self.job_service.update_job(
                    job_id,
                    status=status,
                    metadata={"retries_used": current_retries + (1 if attempt > 1 else 0)}
                )

                logger.info(f"🔄 Executing: {task_name} (job: {job_id}, attempt: {attempt}/{max_retries + 1})")

                # Execute with timeout
                result = await asyncio.wait_for(
                    async_func(*args, **kwargs),
                    timeout=timeout_seconds
                )

                logger.info(f"✅ Task completed: {task_name} (job: {job_id}, attempt: {attempt})")
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_seconds}s"
                logger.warning(f"⏱️ {last_error}: {task_name} (job: {job_id}, attempt: {attempt})")

                if attempt <= max_retries:
                    delay = settings.RETRY_DELAY_SECONDS
                    logger.info(f"⏳ Retrying in {delay}s... ({attempt}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

            except Exception as e:
                last_error = str(e)
                logger.warning(f"❌ Task failed: {task_name} (job: {job_id}, attempt: {attempt}): {e}")

                if attempt <= max_retries:
                    delay = settings.RETRY_DELAY_SECONDS
                    logger.info(f"⏳ Retrying in {delay}s... ({attempt}/{max_retries + 1})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

        # Max retries exceeded
        error_msg = f"Task failed after {max_retries + 1} attempts: {last_error}"
        logger.error(f"🔴 {error_msg} (job: {job_id})")

        self.job_service.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=error_msg,
            metadata={"retries_used": max_retries + 1}
        )

        return None

    def execute_sync_with_timeout(
        self,
        job_id: str,
        task_name: str,
        sync_func: Callable[..., Any],
        *args,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Execute sync function with timeout and retry logic
        (Wrapper around execute_with_timeout for sync functions)

        Args:
            job_id: Job identifier for tracking
            task_name: Human-readable task name
            sync_func: Sync function to execute
            *args: Positional arguments
            max_retries: Max retry attempts
            timeout_seconds: Timeout for execution
            **kwargs: Keyword arguments

        Returns:
            Result from sync_func or None if failed
        """
        max_retries = max_retries or settings.MAX_RETRIES
        timeout_seconds = timeout_seconds or settings.REQUEST_TIMEOUT_SECONDS

        attempt = 0
        last_error = None

        while attempt <= max_retries:
            attempt += 1
            job = self.job_service.get_job(job_id)
            
            if not job:
                logger.error(f"❌ Job not found: {job_id}")
                return None

            current_retries = job.get("metadata", {}).get("retries_used", 0)

            try:
                # Update job status
                status = JobStatus.RETRYING.value if attempt > 1 else JobStatus.PROCESSING.value
                self.job_service.update_job(
                    job_id,
                    status=status,
                    metadata={"retries_used": current_retries + (1 if attempt > 1 else 0)}
                )

                logger.info(f"🔄 Executing: {task_name} (job: {job_id}, attempt: {attempt}/{max_retries + 1})")

                # Execute with timeout using threading
                start_time = time.time()
                result = sync_func(*args, **kwargs)
                elapsed = time.time() - start_time

                if elapsed > timeout_seconds:
                    raise TimeoutError(f"Execution exceeded {timeout_seconds}s ({elapsed:.1f}s)")

                logger.info(f"✅ Task completed: {task_name} (job: {job_id}, attempt: {attempt})")
                return result

            except TimeoutError as e:
                last_error = str(e)
                logger.warning(f"⏱️ {last_error}: {task_name} (job: {job_id}, attempt: {attempt})")

                if attempt <= max_retries:
                    delay = settings.RETRY_DELAY_SECONDS
                    logger.info(f"⏳ Retrying in {delay}s... ({attempt}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    break

            except Exception as e:
                last_error = str(e)
                logger.warning(f"❌ Task failed: {task_name} (job: {job_id}, attempt: {attempt}): {e}")

                if attempt <= max_retries:
                    delay = settings.RETRY_DELAY_SECONDS
                    logger.info(f"⏳ Retrying in {delay}s... ({attempt}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    break

        # Max retries exceeded
        error_msg = f"Task failed after {max_retries + 1} attempts: {last_error}"
        logger.error(f"🔴 {error_msg} (job: {job_id})")

        self.job_service.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=error_msg,
            metadata={"retries_used": max_retries + 1}
        )

        return None


def get_job_executor(job_service: JobService) -> JobExecutor:
    """Factory function to create job executor"""
    return JobExecutor(job_service)
