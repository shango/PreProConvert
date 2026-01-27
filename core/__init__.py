#!/usr/bin/env python3
"""
Core Module
Utility classes for scene analysis and format-agnostic data structures.

v2.5.0 - Added SceneData classes for decoupling readers from exporters.
"""

from .animation_detector import AnimationDetector
from .scene_data import (
    SceneData,
    SceneMetadata,
    CameraData,
    CameraProperties,
    MeshData,
    MeshGeometry,
    TransformData,
    Keyframe,
    AnimationCategories,
    AnimationType,
)

__all__ = [
    'AnimationDetector',
    'SceneData',
    'SceneMetadata',
    'CameraData',
    'CameraProperties',
    'MeshData',
    'MeshGeometry',
    'TransformData',
    'Keyframe',
    'AnimationCategories',
    'AnimationType',
]
