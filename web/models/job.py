"""
Job Models
Pydantic models for job management and API requests/responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime
import uuid


class JobStatus(str, Enum):
    """Job status states"""
    PENDING = "pending"           # File uploaded, waiting for conversion
    CONVERTING = "converting"     # Conversion in progress
    COMPLETED = "completed"       # Conversion successful
    FAILED = "failed"             # Conversion failed


class UploadResponse(BaseModel):
    """Response after file upload"""
    job_id: str
    filename: str
    format: str                   # "alembic", "usd", "maya"
    file_size: int
    detected_frames: Optional[int] = None
    detected_fps: Optional[float] = None


class ConvertRequest(BaseModel):
    """Request to start conversion"""
    shot_name: str = Field(default="shot_001", min_length=1, max_length=100)
    fps: float = Field(default=24.0, gt=0, le=120)
    frame_count: Optional[int] = Field(default=None, gt=0)
    export_ae: bool = True
    export_usd: bool = True
    export_maya_ma: bool = True
    export_fbx: bool = True


class JobStatusResponse(BaseModel):
    """Response for job status query"""
    job_id: str
    status: JobStatus
    filename: str
    format: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_messages: List[str] = []
    error: Optional[str] = None
    download_ready: bool = False


class ProgressEvent(BaseModel):
    """SSE progress event data"""
    message: str
    percent: Optional[int] = None


class Job:
    """Internal job representation"""

    def __init__(self, filename: str, format: str, input_path: str, output_dir: str):
        self.job_id = str(uuid.uuid4())[:8]  # Short ID for simplicity
        self.filename = filename
        self.format = format
        self.input_path = input_path
        self.output_dir = output_dir
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress_messages: List[str] = []
        self.progress_percent: int = 0
        self.error: Optional[str] = None
        self.detected_frames: Optional[int] = None
        self.detected_fps: Optional[float] = None
        self.zip_path: Optional[str] = None

    def to_status_response(self) -> JobStatusResponse:
        """Convert to API response"""
        return JobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            filename=self.filename,
            format=self.format,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            progress_messages=self.progress_messages.copy(),
            error=self.error,
            download_ready=self.zip_path is not None
        )
