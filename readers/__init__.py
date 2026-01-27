#!/usr/bin/env python3
"""
Readers Module
Scene file readers for different 3D formats (Alembic, USD, Maya)
"""

from pathlib import Path

from .base_reader import BaseReader
from .alembic_reader import AlembicReader

# Supported file extensions
ALEMBIC_EXTENSIONS = {'.abc'}
USD_EXTENSIONS = {'.usd', '.usda', '.usdc'}
MAYA_EXTENSIONS = {'.ma'}
SUPPORTED_EXTENSIONS = ALEMBIC_EXTENSIONS | USD_EXTENSIONS | MAYA_EXTENSIONS


def create_reader(input_file):
    """Factory function to create appropriate reader based on file extension

    Args:
        input_file: Path to input scene file

    Returns:
        BaseReader: AlembicReader, USDReader, or MayaReader instance

    Raises:
        ValueError: If file extension is not supported
    """
    path = Path(input_file)
    ext = path.suffix.lower()

    if ext in ALEMBIC_EXTENSIONS:
        return AlembicReader(input_file)
    elif ext in USD_EXTENSIONS:
        # Lazy import to avoid requiring USD when only using Alembic
        from .usd_reader import USDReader
        return USDReader(input_file)
    elif ext in MAYA_EXTENSIONS:
        # Lazy import to avoid loading parser when not needed
        from .maya_reader import MayaReader
        return MayaReader(input_file)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def get_file_type(input_file):
    """Get the file type string for a given file

    Args:
        input_file: Path to input scene file

    Returns:
        str: 'alembic', 'usd', 'maya', or 'unknown'
    """
    ext = Path(input_file).suffix.lower()
    if ext in ALEMBIC_EXTENSIONS:
        return 'alembic'
    elif ext in USD_EXTENSIONS:
        return 'usd'
    elif ext in MAYA_EXTENSIONS:
        return 'maya'
    return 'unknown'


def is_supported_format(input_file):
    """Check if a file has a supported format

    Args:
        input_file: Path to input scene file

    Returns:
        bool: True if format is supported
    """
    ext = Path(input_file).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


__all__ = [
    'BaseReader',
    'AlembicReader',
    'create_reader',
    'get_file_type',
    'is_supported_format',
    'ALEMBIC_EXTENSIONS',
    'USD_EXTENSIONS',
    'MAYA_EXTENSIONS',
    'SUPPORTED_EXTENSIONS',
]
