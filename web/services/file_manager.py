"""
File Manager Service
Handles file uploads, output directories, and ZIP creation.
"""

import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.abc', '.usd', '.usda', '.usdc', '.ma'}

# Format mapping
EXTENSION_TO_FORMAT = {
    '.abc': 'alembic',
    '.usd': 'usd',
    '.usda': 'usd',
    '.usdc': 'usd',
    '.ma': 'maya'
}


class FileManager:
    """Manages file storage for uploads and outputs"""

    def __init__(self, work_dir: str = "./multiconverter_work"):
        self.work_dir = Path(work_dir)
        self.uploads_dir = self.work_dir / "uploads"
        self.outputs_dir = self.work_dir / "outputs"

        # Create directories
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FileManager initialized: {self.work_dir}")

    def is_supported_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        ext = Path(filename).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    def get_format(self, filename: str) -> Optional[str]:
        """Get format name from filename"""
        ext = Path(filename).suffix.lower()
        return EXTENSION_TO_FORMAT.get(ext)

    def get_job_upload_dir(self, job_id: str) -> Path:
        """Get upload directory for a job"""
        path = self.uploads_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get output directory for a job"""
        path = self.outputs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_upload(self, job_id: str, filename: str, file_content: bytes) -> Tuple[str, int]:
        """Save uploaded file to job's upload directory

        Returns:
            Tuple of (file_path, file_size)
        """
        upload_dir = self.get_job_upload_dir(job_id)
        file_path = upload_dir / filename

        with open(file_path, 'wb') as f:
            f.write(file_content)

        file_size = len(file_content)
        logger.info(f"Saved upload: {file_path} ({file_size} bytes)")

        return str(file_path), file_size

    def create_results_zip(self, job_id: str, shot_name: str) -> Optional[str]:
        """Create ZIP file of conversion results

        Returns:
            Path to ZIP file, or None if no results
        """
        output_dir = self.get_job_output_dir(job_id)

        # Find all output subdirectories
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if not subdirs:
            logger.warning(f"No output directories found for job {job_id}")
            return None

        zip_path = output_dir / f"{shot_name}_results.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for subdir in subdirs:
                for file_path in subdir.rglob('*'):
                    if file_path.is_file():
                        # Archive name relative to output_dir
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)

        logger.info(f"Created ZIP: {zip_path}")
        return str(zip_path)

    def delete_upload_files(self, job_id: str):
        """Delete only uploaded source files for a job (keeps outputs)"""
        upload_dir = self.uploads_dir / job_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            logger.info(f"Deleted upload dir: {upload_dir}")

    def delete_output_files(self, job_id: str):
        """Delete only output/download files for a job"""
        output_dir = self.outputs_dir / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"Deleted output dir: {output_dir}")

    def delete_job_files(self, job_id: str):
        """Delete all files for a job"""
        self.delete_upload_files(job_id)
        self.delete_output_files(job_id)

    def cleanup_old_files(self, job_ids: list):
        """Delete files for multiple jobs"""
        for job_id in job_ids:
            self.delete_job_files(job_id)


# Global instance
file_manager = FileManager()
