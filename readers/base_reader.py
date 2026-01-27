#!/usr/bin/env python3
"""
Base Reader Module
Abstract interface for reading 3D scene files (Alembic, USD, etc.)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional


class BaseReader(ABC):
    """Abstract base class for scene file readers

    Provides a consistent interface for reading different 3D file formats.
    All format-specific readers (AlembicReader, USDReader) must implement these methods.
    """

    def __init__(self, file_path: str):
        """Initialize reader with file path

        Args:
            file_path: Path to the scene file
        """
        self.file_path = Path(file_path)
        self._objects_cache = None
        self._parent_map_cache = None

    @abstractmethod
    def get_format_name(self) -> str:
        """Return human-readable format name (e.g., 'Alembic', 'USD')"""
        pass

    @abstractmethod
    def get_all_objects(self) -> List[Any]:
        """Get all objects in the scene hierarchy (cached)

        Returns:
            list: All scene objects
        """
        pass

    @abstractmethod
    def get_cameras(self) -> List[Any]:
        """Get all camera objects in the scene

        Returns:
            list: Camera objects
        """
        pass

    @abstractmethod
    def get_meshes(self) -> List[Any]:
        """Get all mesh objects in the scene

        Returns:
            list: Mesh objects
        """
        pass

    @abstractmethod
    def get_transforms(self) -> List[Any]:
        """Get all transform objects in the scene

        Returns:
            list: Transform objects
        """
        pass

    @abstractmethod
    def get_parent_map(self) -> Dict[str, Any]:
        """Build parent-child relationship map (cached)

        Returns:
            dict: Mapping of child name -> parent object
        """
        pass

    @abstractmethod
    def detect_frame_count(self, fps: int = 24) -> int:
        """Auto-detect frame count from file time sampling

        Args:
            fps: Frames per second (used for calculation)

        Returns:
            int: Number of frames in the animation
        """
        pass

    @abstractmethod
    def get_transform_at_time(self, obj: Any, time_seconds: float,
                              maya_compat: bool = False) -> Tuple[List[float], List[float], List[float]]:
        """Get transform data (position, rotation, scale) at a specific time

        Args:
            obj: Scene object
            time_seconds: Time in seconds to sample
            maya_compat: If True, use Maya-compatible rotation decomposition

        Returns:
            tuple: (translation, rotation, scale) where:
                - translation: [x, y, z]
                - rotation: [rx, ry, rz] in degrees (XYZ Euler)
                - scale: [sx, sy, sz]
        """
        pass

    def get_transform_with_matrix_at_time(self, obj: Any, time_seconds: float) -> Optional[Tuple[List[float], Tuple[float, ...], List[float]]]:
        """Get transform data with rotation matrix for lazy decomposition

        Optional method - if implemented, enables lazy rotation decomposition
        in Keyframe. If not implemented (returns None), falls back to calling
        get_transform_at_time twice.

        Args:
            obj: Scene object
            time_seconds: Time in seconds to sample

        Returns:
            tuple: (translation, rotation_matrix, scale) where:
                - translation: [x, y, z]
                - rotation_matrix: 9 floats (3x3 row-major normalized rotation matrix)
                - scale: [sx, sy, sz]
            Or None if not implemented
        """
        return None  # Default: not implemented, use fallback

    @abstractmethod
    def get_mesh_data_at_time(self, mesh_obj: Any, time_seconds: float) -> Dict[str, Any]:
        """Get mesh geometry data at a specific time

        Args:
            mesh_obj: Mesh object
            time_seconds: Time in seconds to sample

        Returns:
            dict: Mesh data with keys:
                - 'positions': Vertex positions as list of [x, y, z]
                - 'indices': Face vertex indices
                - 'counts': Face vertex counts
        """
        pass

    @abstractmethod
    def get_camera_properties(self, cam_obj: Any, time_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get camera properties at a specific time

        Args:
            cam_obj: Camera object
            time_seconds: Time in seconds (None for first sample)

        Returns:
            dict: Camera properties with keys:
                - 'focal_length': Focal length in mm
                - 'h_aperture': Horizontal aperture in cm
                - 'v_aperture': Vertical aperture in cm
        """
        pass

    def extract_footage_path(self) -> Optional[str]:
        """Extract footage file path from scene metadata

        Returns:
            str: Footage file path, or None if not found
        """
        return None  # Default implementation - override if supported

    def extract_render_resolution(self) -> Tuple[int, int]:
        """Extract render resolution from scene metadata

        Returns:
            tuple: (width, height) in pixels, or (1920, 1080) as fallback
        """
        return (1920, 1080)  # Default implementation

    @abstractmethod
    def _get_full_path(self, obj: Any) -> str:
        """Get full hierarchy path for an object

        Args:
            obj: Scene object

        Returns:
            str: Full path like "/World/Camera/CameraShape"
        """
        pass

    def _is_organizational_group(self, obj: Any) -> bool:
        """Check if transform is just an organizational container

        Override in subclasses if needed.

        Args:
            obj: Scene object to check

        Returns:
            bool: True if object is organizational only
        """
        return False

    def extract_scene_data(self, fps: int, frame_count: int) -> 'SceneData':
        """Extract complete scene data with all animation pre-sampled

        This is the main extraction method that creates a format-agnostic
        SceneData structure. All animation is sampled for all frames with
        both AE and Maya rotation decompositions.

        Args:
            fps: Frames per second for time calculation
            frame_count: Total number of frames to extract

        Returns:
            SceneData: Complete scene data with all animation
        """
        from core.scene_data import (
            SceneData, SceneMetadata, CameraData, MeshData, TransformData,
            Keyframe, CameraProperties, MeshGeometry, AnimationType, AnimationCategories,
            HierarchyTree
        )
        from core.animation_detector import AnimationDetector

        # Step 1: Analyze animation types
        detector = AnimationDetector()
        animation_analysis = detector.analyze_scene(self, frame_count, fps)

        # Step 2: Build parent map once
        parent_map = self.get_parent_map()

        # Step 3: Extract metadata
        width, height = self.extract_render_resolution()
        metadata = SceneMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            footage_path=self.extract_footage_path(),
            source_file_path=str(self.file_path.resolve()),
            source_format_name=self.get_format_name()
        )

        # Step 4: Extract cameras with animation
        cameras = []
        for cam_obj in self.get_cameras():
            cam_name = cam_obj.getName()
            parent = parent_map.get(cam_name)

            # Determine parent_name for display purposes
            # Only use parent_name if the camera follows Alembic convention (name ends with "Shape")
            if cam_name.endswith('Shape') and parent:
                parent_name = parent.getName()
            else:
                parent_name = None

            # Always use the camera object itself for transform extraction
            # This ensures we get the correct world transform regardless of hierarchy
            transform_obj = cam_obj

            # Get camera properties (first frame)
            props = self.get_camera_properties(cam_obj, 1.0 / fps)
            cam_props = CameraProperties(
                focal_length=props['focal_length'],
                h_aperture=props['h_aperture'],
                v_aperture=props['v_aperture']
            )

            # Extract keyframes for all frames (both rotation modes)
            keyframes = self._extract_keyframes(transform_obj, fps, frame_count)

            cameras.append(CameraData(
                name=cam_name,
                parent_name=parent_name,
                full_path=self._get_full_path(cam_obj),
                properties=cam_props,
                keyframes=keyframes
            ))

        # Step 5: Extract meshes with animation
        meshes = []
        vertex_animated_set = set(animation_analysis['vertex_animated'])
        transform_only_set = set(animation_analysis['transform_only'])

        for mesh_obj in self.get_meshes():
            mesh_name = mesh_obj.getName()
            parent = parent_map.get(mesh_name)

            # Determine parent_name for display purposes
            # Only use parent_name if the mesh follows Alembic convention (name ends with "Shape")
            # For USD meshes, the mesh name itself is the proper display name
            if mesh_name.endswith('Shape') and parent:
                parent_name = parent.getName()
            else:
                parent_name = None

            # Always use the mesh object itself for transform extraction
            # USD stores transforms on mesh prims; Alembic uses ComputeLocalToWorldTransform
            # which already includes parent transforms
            transform_obj = mesh_obj

            # Check for blend shapes (Maya reader only)
            blend_shapes = None
            if hasattr(self, 'get_blend_shape_for_mesh'):
                blend_shapes = self.get_blend_shape_for_mesh(mesh_name)

            # Determine animation type
            if blend_shapes is not None:
                anim_type = AnimationType.BLEND_SHAPE
            elif mesh_name in vertex_animated_set:
                anim_type = AnimationType.VERTEX_ANIMATED
            elif mesh_name in transform_only_set:
                anim_type = AnimationType.TRANSFORM_ONLY
            else:
                anim_type = AnimationType.STATIC

            # Get first frame geometry
            mesh_data = self.get_mesh_data_at_time(mesh_obj, 1.0 / fps)
            geometry = MeshGeometry(
                positions=[(p[0], p[1], p[2]) for p in mesh_data['positions']],
                indices=list(mesh_data['indices']),
                counts=list(mesh_data['counts'])
            )

            # Extract transform keyframes
            keyframes = self._extract_keyframes(transform_obj, fps, frame_count)

            # Extract vertex positions per frame if vertex-animated (raw, not blend shape)
            vertex_positions = None
            if anim_type == AnimationType.VERTEX_ANIMATED:
                vertex_positions = {}
                for frame in range(1, frame_count + 1):
                    time_seconds = frame / fps
                    frame_mesh_data = self.get_mesh_data_at_time(mesh_obj, time_seconds)
                    vertex_positions[frame] = [
                        (p[0], p[1], p[2]) for p in frame_mesh_data['positions']
                    ]

            meshes.append(MeshData(
                name=mesh_name,
                parent_name=parent_name,
                full_path=self._get_full_path(mesh_obj),
                animation_type=anim_type,
                keyframes=keyframes,
                geometry=geometry,
                vertex_positions_per_frame=vertex_positions,
                blend_shapes=blend_shapes
            ))

        # Step 6: Extract pure transforms (locators - no camera/mesh children)
        transforms = []
        processed = set(c.name for c in cameras) | set(m.name for m in meshes)
        processed.update(c.parent_name for c in cameras if c.parent_name)
        processed.update(m.parent_name for m in meshes if m.parent_name)

        # Also exclude ALL ancestors of cameras/meshes (grandparents, etc.)
        # This prevents exporting intermediate transforms that are part of camera/mesh hierarchies
        all_items = list(cameras) + list(meshes)
        for item in all_items:
            parts = [p for p in item.full_path.split('/') if p]
            # Add all path components except the last (the item itself)
            for part in parts[:-1]:
                processed.add(part)

        for xform_obj in self.get_transforms():
            xform_name = xform_obj.getName()
            if xform_name in processed:
                continue
            if self._is_organizational_group(xform_obj):
                continue

            parent = parent_map.get(xform_name)
            parent_name = parent.getName() if parent else None

            keyframes = self._extract_keyframes(xform_obj, fps, frame_count)
            transforms.append(TransformData(
                name=xform_name,
                parent_name=parent_name,
                full_path=self._get_full_path(xform_obj),
                keyframes=keyframes
            ))

        # Step 7: Build animation categories (accounting for blend shapes)
        blend_shape_names = [m.name for m in meshes if m.animation_type == AnimationType.BLEND_SHAPE]
        vertex_animated_names = [m.name for m in meshes if m.animation_type == AnimationType.VERTEX_ANIMATED]
        transform_only_names = [m.name for m in meshes if m.animation_type == AnimationType.TRANSFORM_ONLY]
        static_names = [m.name for m in meshes if m.animation_type == AnimationType.STATIC]

        categories = AnimationCategories(
            vertex_animated=vertex_animated_names,
            blend_shape=blend_shape_names,
            transform_only=transform_only_names,
            static=static_names
        )

        # Step 8: Build hierarchy tree (centralized, used by all exporters)
        hierarchy = HierarchyTree.build_from_scene_items(cameras, meshes, transforms)

        return SceneData(
            metadata=metadata,
            cameras=cameras,
            meshes=meshes,
            transforms=transforms,
            animation_categories=categories,
            hierarchy=hierarchy
        )

    def _extract_keyframes(self, obj: Any, fps: int, frame_count: int) -> List['Keyframe']:
        """Extract keyframes with rotation data

        Uses matrix-based approach if available (single decomposition call with
        lazy rotation computation), otherwise falls back to calling decompose twice.

        Args:
            obj: Scene object to sample
            fps: Frames per second
            frame_count: Total number of frames

        Returns:
            List[Keyframe]: Animation keyframes for all frames
        """
        from core.scene_data import Keyframe

        keyframes = []

        # Check if matrix-based extraction is available (more efficient)
        use_matrix = self.get_transform_with_matrix_at_time(obj, 1.0 / fps) is not None

        for frame in range(1, frame_count + 1):
            time_seconds = frame / fps

            if use_matrix:
                # New efficient path: single call, lazy rotation decomposition
                result = self.get_transform_with_matrix_at_time(obj, time_seconds)
                pos, rot_matrix, scale = result
                keyframes.append(Keyframe.create(
                    frame=frame,
                    position=tuple(pos),
                    scale=tuple(scale),
                    rotation_matrix=tuple(rot_matrix)
                ))
            else:
                # Fallback: two decomposition calls (original behavior)
                pos_ae, rot_ae, scale = self.get_transform_at_time(obj, time_seconds, maya_compat=False)
                _, rot_maya, _ = self.get_transform_at_time(obj, time_seconds, maya_compat=True)
                keyframes.append(Keyframe.create(
                    frame=frame,
                    position=tuple(pos_ae),
                    scale=tuple(scale),
                    rotation_ae=tuple(rot_ae),
                    rotation_maya=tuple(rot_maya)
                ))

        return keyframes
