#!/usr/bin/env python3
"""
Animation Detector Module
Detects animation types (transform vs vertex) to determine export strategy

Works with any BaseReader implementation (Alembic, USD, etc.)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class AnimationDetector:
    """Analyzes scene to detect different types of animation

    Distinguishes between:
    - Transform animation: Position, rotation, scale changes
    - Vertex animation: Individual vertex position changes (deformation)
    - Static: No animation

    This information is critical for format-specific export strategies:
    - After Effects: Can only handle transform animation (skip vertex-animated meshes)
    - USD/Maya: Can handle both transform and vertex animation
    """

    def __init__(self, tolerance=0.0001):
        """Initialize animation detector

        Args:
            tolerance: Threshold for detecting vertex position changes
        """
        self.tolerance = tolerance

    def detect_vertex_animation(self, reader, mesh_obj, frame_count, fps, start_time=0.0):
        """Detect if a mesh has vertex animation (deformation)

        Samples vertex positions across multiple frames to detect changes.
        Uses numpy for fast vectorized comparison.

        Args:
            reader: BaseReader instance (AlembicReader or USDReader)
            mesh_obj: Mesh object to analyze
            frame_count: Total number of frames in the animation
            fps: Frames per second
            start_time: Time offset for first frame (from file's time sampling)

        Returns:
            bool: True if vertex animation detected, False otherwise
        """
        try:
            # Get first frame positions as baseline
            first_data = reader.get_mesh_data_at_time(mesh_obj, start_time)
            first_positions = first_data['positions']
            num_verts = len(first_positions)

            # Early exit if no vertices
            if num_verts == 0:
                return False

            # Convert first frame to numpy array once
            first_arr = np.array(first_positions, dtype=np.float64)

            # Sample every 5th frame for efficiency, or at least 5 frames
            sample_interval = max(5, frame_count // 20)

            # Check sampled frames for vertex position changes
            max_delta_seen = 0.0
            for frame in range(2, frame_count + 1, sample_interval):
                time_seconds = start_time + (frame - 1) / fps
                mesh_data = reader.get_mesh_data_at_time(mesh_obj, time_seconds)
                positions = mesh_data['positions']

                # Vectorized comparison with numpy
                current_arr = np.array(positions, dtype=np.float64)
                delta = np.abs(current_arr - first_arr)
                frame_max_delta = np.max(delta)
                max_delta_seen = max(max_delta_seen, frame_max_delta)

                if frame_max_delta > self.tolerance:
                    mesh_name = mesh_obj.getName()
                    logger.info(f"    Vertex animation detected: max delta={frame_max_delta:.6f} at frame {frame} (tolerance={self.tolerance})")
                    return True

            return False

        except Exception:
            # If we can't read the mesh, assume no vertex animation
            return False

    def detect_transform_animation(self, reader, obj, frame_count, fps, start_time=0.0):
        """Detect if an object has transform animation

        Args:
            reader: BaseReader instance (AlembicReader or USDReader)
            obj: Object to analyze (transform/xform)
            frame_count: Total number of frames
            fps: Frames per second
            start_time: Time offset for first frame (from file's time sampling)

        Returns:
            bool: True if transform animation detected, False otherwise
        """
        try:
            # Get first and last frame transforms
            last_time = start_time + (frame_count - 1) / fps

            first_pos, first_rot, first_scale = reader.get_transform_at_time(obj, start_time)
            last_pos, last_rot, last_scale = reader.get_transform_at_time(obj, last_time)

            # Check if position changed
            for i in range(3):
                if abs(first_pos[i] - last_pos[i]) > self.tolerance:
                    return True

            # Check if rotation changed
            for i in range(3):
                if abs(first_rot[i] - last_rot[i]) > self.tolerance:
                    return True

            # Check if scale changed
            for i in range(3):
                if abs(first_scale[i] - last_scale[i]) > self.tolerance:
                    return True

            return False

        except Exception:
            return False

    def analyze_scene(self, reader, frame_count, fps, start_time=0.0):
        """Analyze entire scene and categorize all meshes by animation type

        Args:
            reader: BaseReader instance (AlembicReader or USDReader)
            frame_count: Total number of frames
            fps: Frames per second
            start_time: Time offset for first frame (from file's time sampling)

        Returns:
            dict: Animation analysis with keys:
                - 'vertex_animated': List of mesh names with vertex animation
                - 'transform_only': List of mesh names with only transform animation
                - 'static': List of mesh names with no animation
        """
        result = {
            'vertex_animated': [],
            'transform_only': [],
            'static': []
        }

        meshes = list(reader.get_meshes())
        total = len(meshes)

        for idx, mesh_obj in enumerate(meshes, 1):
            mesh_name = mesh_obj.getName()
            logger.info(f"  Analyzing mesh {idx}/{total}: {mesh_name}")

            # Check for vertex animation first (most important for AE)
            has_vertex_anim = self.detect_vertex_animation(reader, mesh_obj, frame_count, fps, start_time)

            if has_vertex_anim:
                logger.info(f"    -> vertex animated")
                result['vertex_animated'].append(mesh_name)
                continue

            # Check for transform animation on parent
            # Use get_parent_of for direct parent lookup (avoids name collision
            # when multiple meshes share the same name)
            parent = reader.get_parent_of(mesh_obj)
            has_transform_anim = False

            if parent:
                has_transform_anim = self.detect_transform_animation(reader, parent, frame_count, fps, start_time)

            if has_transform_anim:
                logger.info(f"    -> transform only")
                result['transform_only'].append(mesh_name)
            else:
                logger.info(f"    -> static")
                result['static'].append(mesh_name)

        return result

    def get_animation_summary(self, animation_data):
        """Generate human-readable summary of animation analysis

        Args:
            animation_data: Result from analyze_scene()

        Returns:
            str: Formatted summary text
        """
        lines = []
        lines.append("Animation Analysis:")
        lines.append(f"  - Vertex Animated: {len(animation_data['vertex_animated'])} meshes")
        lines.append(f"  - Transform Only: {len(animation_data['transform_only'])} meshes")
        lines.append(f"  - Static: {len(animation_data['static'])} meshes")

        if animation_data['vertex_animated']:
            lines.append("\n  Vertex Animated Meshes:")
            for name in animation_data['vertex_animated']:
                lines.append(f"    - {name}")

        return "\n".join(lines)
