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

    def get_time_range(self) -> Optional[Tuple[float, float, int, float]]:
        """Get the time range of animation in the file

        Optional method - returns the actual time range from the file's
        time sampling. Used to support files that don't start at time 0.

        Returns:
            tuple: (start_time, end_time, num_samples, time_per_sample) or None
                - start_time: Time in seconds of first sample
                - end_time: Time in seconds of last sample
                - num_samples: Total number of time samples
                - time_per_sample: Time between samples (1/fps)
        """
        return None  # Default: not implemented, use fps-based calculation

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

    def _uses_shape_transform_pattern(self) -> bool:
        """Whether this format uses separate shape/transform nodes.

        In Alembic and Maya, cameras and meshes are 'shape' nodes that live
        under parent transform nodes. The transform's name is the meaningful
        display name (e.g., Camera1 > CameraShape, or Camera1 > object).

        In USD, prims are self-contained — no shape/transform split.

        Returns:
            bool: True if cameras/meshes should use parent name for display
        """
        return False

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
        import logging
        _logger = logging.getLogger(__name__)

        from core.scene_data import (
            SceneData, SceneMetadata, CameraData, MeshData, TransformData,
            Keyframe, CameraProperties, MeshGeometry, AnimationType, AnimationCategories,
            HierarchyTree
        )
        from core.animation_detector import AnimationDetector

        # Get actual time range from file (if available)
        time_range = self.get_time_range()
        if time_range:
            start_time, end_time, num_samples, time_per_sample = time_range
            # Use actual fps from file if significantly different
            detected_fps = 1.0 / time_per_sample if time_per_sample > 0 else fps
            # Calculate start frame from start time (e.g., 41.666s at 24fps = frame 1001)
            start_frame = round(start_time * fps) + 1
        else:
            # Fallback: assume animation starts at time 0, frame 1
            start_time = 0.0
            time_per_sample = 1.0 / fps
            start_frame = 1

        # Step 1: Analyze animation types (pass start_time)
        detector = AnimationDetector()
        animation_analysis = detector.analyze_scene(self, frame_count, fps, start_time)

        # Step 2: Build parent map once
        parent_map = self.get_parent_map()

        # Step 3: Extract metadata
        width, height = self.extract_render_resolution()
        metadata = SceneMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            start_frame=start_frame,
            footage_path=self.extract_footage_path(),
            source_file_path=str(self.file_path.resolve()),
            source_format_name=self.get_format_name()
        )

        # Step 4: Extract cameras with animation
        cameras = []
        cam_objects = self.get_cameras()
        _logger.info(f"Found {len(cam_objects)} cameras")
        for cam_obj in cam_objects:
            cam_name = cam_obj.getName()
            parent = parent_map.get(cam_name)

            # Determine parent_name for display purposes
            # Alembic/Maya: cameras are shape nodes under transforms — use parent name
            # USD: camera prim IS the named object — no parent name needed
            if self._uses_shape_transform_pattern() and parent:
                parent_name = parent.getName()
            else:
                parent_name = None

            display_name = parent_name if parent_name else cam_name
            _logger.info(f"  Camera: {cam_name} -> display: {display_name} (path: {self._get_full_path(cam_obj)})")

            # Always use the camera object itself for transform extraction
            # This ensures we get the correct world transform regardless of hierarchy
            transform_obj = cam_obj

            # Get camera properties (first frame)
            props = self.get_camera_properties(cam_obj, start_time)
            cam_props = CameraProperties(
                focal_length=props['focal_length'],
                h_aperture=props['h_aperture'],
                v_aperture=props['v_aperture']
            )

            # Extract keyframes for all frames (both rotation modes)
            keyframes = self._extract_keyframes(transform_obj, fps, frame_count, start_time)

            cameras.append(CameraData(
                name=cam_name,
                parent_name=parent_name,
                full_path=self._get_full_path(cam_obj),
                properties=cam_props,
                keyframes=keyframes
            ))

        # Step 5: Extract meshes with animation
        meshes = []
        mesh_objects = list(self.get_meshes())
        vertex_animated_set = set(animation_analysis['vertex_animated'])
        transform_only_set = set(animation_analysis['transform_only'])
        _logger.info(f"Found {len(mesh_objects)} meshes (vertex_anim: {len(vertex_animated_set)}, transform: {len(transform_only_set)}, static: {len(mesh_objects) - len(vertex_animated_set) - len(transform_only_set)})")

        for mesh_obj in mesh_objects:
            mesh_name = mesh_obj.getName()
            parent = parent_map.get(mesh_name)

            # Determine parent_name for display purposes
            # Alembic/Maya: meshes are shape nodes under transforms — use parent name
            # USD: mesh prim IS the named object — no parent name needed
            if self._uses_shape_transform_pattern() and parent:
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

            display_name = parent_name if parent_name else mesh_name
            _logger.info(f"  Mesh: {mesh_name} -> display: {display_name}, type: {anim_type.value} (path: {self._get_full_path(mesh_obj)})")

            # Get first frame geometry
            # Both Alembic and USD store mesh vertices in local (object) space
            # SceneData always stores vertices in local space (normalized)
            mesh_data = self.get_mesh_data_at_time(mesh_obj, start_time)
            geometry = MeshGeometry(
                positions=[(p[0], p[1], p[2]) for p in mesh_data['positions']],
                indices=list(mesh_data['indices']),
                counts=list(mesh_data['counts'])
            )
            _logger.info(f"    Geometry: {len(geometry.positions)} verts, {len(geometry.counts)} faces")

            # Extract transform keyframes
            keyframes = self._extract_keyframes(transform_obj, fps, frame_count, start_time)

            # Extract vertex positions per frame if vertex-animated (raw, not blend shape)
            vertex_positions = None
            if anim_type == AnimationType.VERTEX_ANIMATED:
                _logger.info(f"Extracting vertex data for {mesh_name}: {frame_count} frames")
                vertex_positions = {}
                for frame in range(1, frame_count + 1):
                    # Use start_time offset: Frame 1 = start_time, Frame 2 = start_time + 1/fps
                    time_seconds = start_time + (frame - 1) / fps
                    frame_mesh_data = self.get_mesh_data_at_time(mesh_obj, time_seconds)
                    vertex_positions[frame] = [
                        (p[0], p[1], p[2]) for p in frame_mesh_data['positions']
                    ]
                    # Log memory every 25 frames
                    if frame % 25 == 0 or frame == frame_count:
                        try:
                            with open('/proc/self/status', 'r') as _f:
                                for _line in _f:
                                    if _line.startswith('VmRSS:'):
                                        _mem_mb = int(_line.split()[1]) // 1024
                                        _logger.info(f"  Vertex extraction {mesh_name}: frame {frame}/{frame_count} - Memory: {_mem_mb}MB")
                                        break
                        except (FileNotFoundError, ValueError):
                            pass

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

            keyframes = self._extract_keyframes(xform_obj, fps, frame_count, start_time)
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

    def _extract_keyframes(self, obj: Any, fps: int, frame_count: int, start_time: float = 0.0) -> List['Keyframe']:
        """Extract keyframes with rotation data

        Uses matrix-based approach if available (single decomposition call with
        lazy rotation computation), otherwise falls back to calling decompose twice.

        Args:
            obj: Scene object to sample
            fps: Frames per second
            frame_count: Total number of frames
            start_time: Time offset for first frame (from file's time sampling)

        Returns:
            List[Keyframe]: Animation keyframes for all frames
        """
        from core.scene_data import Keyframe

        keyframes = []

        # Check if matrix-based extraction is available (more efficient)
        # Test at start_time (frame 1's time)
        use_matrix = self.get_transform_with_matrix_at_time(obj, start_time) is not None

        for frame in range(1, frame_count + 1):
            # Frame 1 = start_time, Frame 2 = start_time + 1/fps, etc.
            time_seconds = start_time + (frame - 1) / fps

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
