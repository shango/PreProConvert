"""
Conversion Service
Wraps AlembicToJSXConverter for async execution with progress streaming.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from preproconverter import PreProConverter
from web.models.job import ConvertRequest, JobStatus
from web.services.job_manager import job_manager
from web.services.file_manager import file_manager

logger = logging.getLogger(__name__)

# Thread pool for running blocking conversions
executor = ThreadPoolExecutor(max_workers=2)


class ConversionService:
    """Handles scene file conversion with progress callbacks"""

    async def convert(self, job_id: str, options: ConvertRequest) -> bool:
        """Run conversion asynchronously

        Args:
            job_id: Job ID to convert
            options: Conversion options

        Returns:
            True if successful, False otherwise
        """
        job = job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        # Update status to converting
        job_manager.update_status(job_id, JobStatus.CONVERTING)

        def progress_callback(message: str, progress: int = None):
            """Thread-safe progress callback with optional percentage"""
            job_manager.add_progress(job_id, message, progress)

        try:
            # Run conversion in thread pool (blocking I/O)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                executor,
                self._run_conversion,
                job,
                options,
                progress_callback
            )

            if results.get('success', False):
                # Create ZIP of results
                zip_path = file_manager.create_results_zip(job_id, options.shot_name)
                if zip_path:
                    job_manager.set_zip_path(job_id, zip_path)

                job_manager.update_status(job_id, JobStatus.COMPLETED)
                job_manager.add_progress(job_id, "Conversion complete!", progress=100)
                job_manager.signal_complete(job_id)
                return True
            else:
                error_msg = results.get('error', 'Unknown error')
                job_manager.update_status(job_id, JobStatus.FAILED, error=error_msg)
                job_manager.add_progress(job_id, f"Error: {error_msg}")
                job_manager.signal_complete(job_id)
                return False

        except Exception as e:
            logger.exception(f"Conversion failed for job {job_id}")
            job_manager.update_status(job_id, JobStatus.FAILED, error=str(e))
            job_manager.add_progress(job_id, f"Error: {str(e)}")
            job_manager.signal_complete(job_id)
            return False

    def _run_conversion(self, job, options: ConvertRequest, progress_callback) -> dict:
        """Run conversion synchronously (called in thread pool)"""
        try:
            converter = PreProConverter(progress_callback=progress_callback)

            results = converter.convert_multi_format(
                input_file=job.input_path,
                output_dir=job.output_dir,
                shot_name=options.shot_name,
                fps=options.fps,
                frame_count=options.frame_count,
                export_ae=options.export_ae,
                export_usd=options.export_usd,
                export_maya_ma=options.export_maya_ma,
                export_fbx=options.export_fbx
            )

            return {'success': True, 'results': results}

        except Exception as e:
            logger.exception("Conversion error")
            return {'success': False, 'error': str(e)}

    def detect_scene_info(self, input_path: str, fps: float = 24.0) -> dict:
        """Detect frame count and FPS from input file

        Args:
            input_path: Path to input file
            fps: Default FPS to use

        Returns:
            Dict with 'frames' and 'fps' keys
        """
        try:
            from readers import create_reader

            reader = create_reader(input_path)
            detected_frames = reader.detect_frame_count(fps=int(fps))

            return {
                'frames': detected_frames,
                'fps': fps
            }
        except Exception as e:
            logger.warning(f"Could not detect scene info: {e}")
            return {
                'frames': None,
                'fps': fps
            }


# Global instance
conversion_service = ConversionService()
