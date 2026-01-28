"""
FastAPI Application
Main application entry point with all routes.
"""

import asyncio
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.models.job import (
    UploadResponse, ConvertRequest, JobStatusResponse, JobStatus
)
from web.services.job_manager import job_manager
from web.services.file_manager import file_manager
from web.services.conversion import conversion_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Max upload size (20GB)
MAX_UPLOAD_SIZE = 20 * 1024 * 1024 * 1024


async def cleanup_task():
    """Background task to clean up old jobs and files every 2 minutes"""
    while True:
        await asyncio.sleep(120)  # Check every 2 minutes
        try:
            old_job_ids = job_manager.cleanup_old_jobs()
            if old_job_ids:
                file_manager.cleanup_old_files(old_job_ids)
                logger.info(f"Cleanup removed {len(old_job_ids)} jobs older than {job_manager.cleanup_minutes} minutes")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("PreProConvert starting...")
    logger.info("Checking VFX library imports...")
    try:
        import alembic
        logger.info(f"PyAlembic loaded: {alembic.__file__}")
    except ImportError as e:
        logger.error(f"PyAlembic not available: {e}")
    try:
        import imath
        logger.info(f"imath loaded: {imath.__file__}")
    except ImportError as e:
        logger.error(f"imath not available: {e}")

    # Start background cleanup task
    cleanup = asyncio.create_task(cleanup_task())
    logger.info("Started background cleanup task (downloads expire 15 min after conversion)")

    yield

    # Cancel cleanup task on shutdown
    cleanup.cancel()
    logger.info("PreProConvert shutting down...")


app = FastAPI(
    title="PreProConvert",
    description="Convert Alembic, USD, and Maya files to multiple formats",
    version="1.0.0",
    lifespan=lifespan
)


# === UPLOAD ENDPOINT ===

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a scene file for conversion"""

    # Validate file extension
    if not file_manager.is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: .abc, .usd, .usda, .usdc, .ma"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    # Get format
    format_name = file_manager.get_format(file.filename)

    # Create job
    job = job_manager.create_job(
        filename=file.filename,
        format=format_name,
        input_path="",  # Will be set after save
        output_dir=""   # Will be set after save
    )

    # Save file
    input_path, file_size = await file_manager.save_upload(
        job.job_id, file.filename, content
    )
    job.input_path = input_path
    job.output_dir = str(file_manager.get_job_output_dir(job.job_id))

    # Detect scene info
    scene_info = conversion_service.detect_scene_info(input_path)
    job_manager.set_detected_info(job.job_id, scene_info.get('frames'), scene_info.get('fps'))

    return UploadResponse(
        job_id=job.job_id,
        filename=file.filename,
        format=format_name,
        file_size=file_size,
        detected_frames=scene_info.get('frames'),
        detected_fps=scene_info.get('fps')
    )


# === JOB ENDPOINTS ===

@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_status_response()


@app.post("/api/jobs/{job_id}/convert", response_model=JobStatusResponse)
async def start_conversion(job_id: str, options: ConvertRequest, background_tasks: BackgroundTasks):
    """Start conversion for a job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is already {job.status.value}"
        )

    # Start conversion in background
    background_tasks.add_task(conversion_service.convert, job_id, options)

    # Update status immediately
    job_manager.update_status(job_id, JobStatus.CONVERTING)

    return job.to_status_response()


@app.get("/api/jobs/{job_id}/progress")
async def get_progress_stream(job_id: str):
    """Server-Sent Events stream for progress updates"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        queue = job_manager.get_progress_queue(job_id)
        if not queue:
            return

        while True:
            try:
                # Non-blocking check
                if not queue.empty():
                    data = queue.get_nowait()

                    if data.get('complete'):
                        # Send final status
                        final_job = job_manager.get_job(job_id)
                        yield f"event: complete\ndata: {json.dumps(final_job.to_status_response().model_dump(), default=str)}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps(data)}\n\n"

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"SSE error: {e}")
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    file_manager.delete_job_files(job_id)

    # Delete job from manager
    job_manager.delete_job(job_id)

    return {"message": "Job deleted"}


# === DOWNLOAD ENDPOINT ===

@app.get("/api/jobs/{job_id}/download")
async def download_results(job_id: str):
    """Download conversion results as ZIP"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete (status: {job.status.value})"
        )

    if not job.zip_path or not Path(job.zip_path).exists():
        raise HTTPException(status_code=404, detail="Results file not found")

    return FileResponse(
        path=job.zip_path,
        filename=Path(job.zip_path).name,
        media_type="application/zip"
    )


# === STATIC FILES ===

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return HTMLResponse(content="<h1>PreProConvert</h1><p>Frontend not found</p>")


# === HEALTH CHECK ===

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}


# === INFO ENDPOINT ===

@app.get("/api/formats")
async def get_supported_formats():
    """Get supported input and output formats"""
    return {
        "input_formats": [
            {"extension": ".abc", "name": "Alembic"},
            {"extension": ".usd", "name": "USD"},
            {"extension": ".usda", "name": "USD ASCII"},
            {"extension": ".usdc", "name": "USD Binary"},
            {"extension": ".ma", "name": "Maya ASCII"}
        ],
        "output_formats": [
            {"id": "ae", "name": "After Effects", "extensions": [".jsx", ".obj"]},
            {"id": "usd", "name": "USD", "extensions": [".usdc"]},
            {"id": "maya_ma", "name": "Maya MA", "extensions": [".ma"]},
            {"id": "fbx", "name": "FBX", "extensions": [".fbx"]}
        ]
    }
