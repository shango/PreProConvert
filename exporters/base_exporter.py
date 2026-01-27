#!/usr/bin/env python3
"""
Base Exporter Module
Abstract base class ensuring consistent interface across all exporters

v2.5.0 - Exporters now receive SceneData instead of reader objects.
This decouples exporters from input format implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.scene_data import SceneData


class BaseExporter(ABC):
    """Abstract base class for all format exporters

    Provides consistent interface and common utilities for all exporters.
    Each format exporter (AE, USD, etc.) inherits from this class.

    Key principles:
    - Single Responsibility: Each exporter handles ONE format
    - Consistent Interface: All exporters implement the same methods
    - Shared Utilities: Common functionality (logging, path validation) provided here
    - Format Agnostic: Works with SceneData, not reader objects (v2.5.0+)
    """

    def __init__(self, progress_callback=None):
        """Initialize exporter

        Args:
            progress_callback: Optional function to call for progress updates
                              Signature: callback(message: str) -> None
        """
        self.progress_callback = progress_callback

    def log(self, message):
        """Send progress/status message

        Args:
            message: Message to log
        """
        if self.progress_callback:
            self.progress_callback(message)
        print(message)

    @abstractmethod
    def export(self, scene_data: 'SceneData', output_path, shot_name):
        """Export scene data to specific format

        This is the main export method that each format must implement.
        v2.5.0+: Takes SceneData instead of reader for format-agnostic export.

        Args:
            scene_data: SceneData instance with all pre-extracted animation and geometry.
                       Contains cameras, meshes, transforms with keyframes, and metadata.
            output_path: Output directory path (Path object or string)
            shot_name: Shot name for naming files

        Returns:
            dict: Export results with format-specific keys
                  Should include at least:
                  - 'success': bool
                  - 'files': list of created file paths
                  - 'message': str status message
        """
        pass

    @abstractmethod
    def get_format_name(self):
        """Return human-readable format name

        Returns:
            str: Format name (e.g., "After Effects JSX", "USD", "Maya USD")
        """
        pass

    @abstractmethod
    def get_file_extension(self):
        """Return primary file extension for this format

        Returns:
            str: File extension without dot (e.g., "jsx", "usdc", "ma")
        """
        pass

    def validate_output_path(self, output_path):
        """Validate and create output directory if needed

        Args:
            output_path: Directory path to validate

        Returns:
            Path: Validated Path object

        Raises:
            ValueError: If path is invalid
        """
        path = Path(output_path)

        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory {path}: {e}")

        # Verify we can write to the directory
        if not path.is_dir():
            raise ValueError(f"Output path is not a directory: {path}")

        return path

    def get_export_summary(self, result):
        """Generate human-readable summary of export results

        Args:
            result: Export result dict from export() method

        Returns:
            str: Formatted summary text
        """
        lines = []
        lines.append(f"✓ {self.get_format_name()} Export Complete")

        if 'files' in result:
            files = result['files']
            if isinstance(files, list):
                lines.append(f"  Files created: {len(files)}")
                for file_path in files:
                    lines.append(f"    - {Path(file_path).name}")
            else:
                lines.append(f"    - {Path(files).name}")

        if 'message' in result:
            lines.append(f"  {result['message']}")

        return "\n".join(lines)
