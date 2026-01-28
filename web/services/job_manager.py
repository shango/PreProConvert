"""
Job Manager Service
In-memory job state management with thread-safe progress updates.
"""

from typing import Dict, Optional, List
from queue import Queue
from threading import Lock
from datetime import datetime, timedelta
import logging

from web.models.job import Job, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Manages job state and progress queues"""

    def __init__(self, cleanup_hours: int = 3):
        self._jobs: Dict[str, Job] = {}
        self._progress_queues: Dict[str, Queue] = {}
        self._lock = Lock()
        self.cleanup_hours = cleanup_hours

    def create_job(self, filename: str, format: str, input_path: str, output_dir: str) -> Job:
        """Create a new job"""
        job = Job(
            filename=filename,
            format=format,
            input_path=input_path,
            output_dir=output_dir
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._progress_queues[job.job_id] = Queue()
        logger.info(f"Created job {job.job_id} for {filename}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Get all jobs"""
        return list(self._jobs.values())

    def update_status(self, job_id: str, status: JobStatus, error: Optional[str] = None):
        """Update job status"""
        job = self._jobs.get(job_id)
        if job:
            job.status = status
            if status == JobStatus.CONVERTING:
                job.started_at = datetime.now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now()
            if error:
                job.error = error
            logger.info(f"Job {job_id} status: {status.value}")

    def add_progress(self, job_id: str, message: str, progress: int = None):
        """Add progress message with optional percentage (thread-safe)

        Args:
            job_id: Job ID
            message: Progress message
            progress: Optional progress percentage (0-100)
        """
        job = self._jobs.get(job_id)
        if job:
            job.progress_messages.append(message)
            if progress is not None:
                job.progress_percent = progress

        queue = self._progress_queues.get(job_id)
        if queue:
            data = {"message": message}
            if progress is not None:
                data["progress"] = progress
            queue.put(data)

    def get_progress_queue(self, job_id: str) -> Optional[Queue]:
        """Get progress queue for SSE streaming"""
        return self._progress_queues.get(job_id)

    def signal_complete(self, job_id: str):
        """Signal that conversion is complete (for SSE)"""
        queue = self._progress_queues.get(job_id)
        if queue:
            queue.put({"complete": True})

    def set_zip_path(self, job_id: str, zip_path: str):
        """Set the ZIP file path for completed job"""
        job = self._jobs.get(job_id)
        if job:
            job.zip_path = zip_path

    def set_detected_info(self, job_id: str, frames: Optional[int], fps: Optional[float]):
        """Set detected frame count and FPS"""
        job = self._jobs.get(job_id)
        if job:
            job.detected_frames = frames
            job.detected_fps = fps

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its progress queue"""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                if job_id in self._progress_queues:
                    del self._progress_queues[job_id]
                logger.info(f"Deleted job {job_id}")
                return True
        return False

    def cleanup_old_jobs(self) -> List[str]:
        """Remove jobs older than cleanup_hours, return list of removed job IDs"""
        cutoff = datetime.now() - timedelta(hours=self.cleanup_hours)
        to_remove = []

        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.created_at < cutoff:
                    to_remove.append(job_id)

            for job_id in to_remove:
                del self._jobs[job_id]
                if job_id in self._progress_queues:
                    del self._progress_queues[job_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")

        return to_remove


# Global instance
job_manager = JobManager()
