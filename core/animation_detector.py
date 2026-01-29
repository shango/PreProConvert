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
            frames_checked = 0
            for frame in range(2, frame_count + 1, sample_interval):
                time_seconds = start_time + (frame - 1) / fps
                mesh_data = reader.get_mesh_data_at_time(mesh_obj, time_seconds)
                positions = mesh_data['positions']
                frames_checked += 1

                # Vectorized comparison with numpy
                current_arr = np.array(positions, dtype=np.float64)
                delta = np.abs(current_arr - first_arr)
                frame_max_delta = np.max(delta)
                max_delta_seen = max(max_delta_seen, frame_max_delta)

                if frame_max_delta > self.tolerance:
                    mesh_name = mesh_obj.getName()
                    logger.info(f"    Vertex animation detected: max delta={frame_max_delta:.6f} at frame {frame} (tolerance={self.tolerance})")
                    # Return the delta value along with True
                    return True, frame_max_delta

            # Log diagnostic info for meshes with many vertices that weren't detected as animated
            if num_verts > 1000:
                mesh_name = mesh_obj.getName()
                # Check if reader has sample count method (Alembic-specific)
                sample_count = reader.get_mesh_sample_count(mesh_obj) if hasattr(reader, 'get_mesh_sample_count') else 'unknown'
                logger.info(f"    Large mesh ({num_verts} verts) NOT vertex animated: max_delta={max_delta_seen:.8f}, checked {frames_checked} frames, start_time={start_time:.3f}, position_samples={sample_count}")

            return False, max_delta_seen

        except Exception:
            # If we can't read the mesh, assume no vertex animation
            return False, 0.0

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
            'vertex_animated_deltas': {},  # Track delta values for diagnosis
            'transform_only': [],
            'static': []
        }

        meshes = list(reader.get_meshes())
        total = len(meshes)

        for idx, mesh_obj in enumerate(meshes, 1):
            shape_name = mesh_obj.getName()

            # Get parent transform for display name (shape names may all be "mesh" in Nuke exports)
            parent = reader.get_parent_of(mesh_obj)
            if parent and reader._uses_shape_transform_pattern():
                display_name = parent.getName()
            else:
                display_name = shape_name

            # Get full path for unique identification
            full_path = reader._get_full_path(mesh_obj) if hasattr(reader, '_get_full_path') else display_name

            logger.info(f"  Analyzing mesh {idx}/{total}: {display_name} (shape: {shape_name}, path: {full_path})")

            # Check for vertex animation first (most important for AE)
            has_vertex_anim, vertex_delta = self.detect_vertex_animation(reader, mesh_obj, frame_count, fps, start_time)

            if has_vertex_anim:
                logger.info(f"    -> vertex animated (delta={vertex_delta:.6f})")
                result['vertex_animated'].append(display_name)
                result['vertex_animated_deltas'][display_name] = vertex_delta
                continue

            # Check for transform animation on parent
            # Use get_parent_of for direct parent lookup (avoids name collision
            # when multiple meshes share the same name)
            has_transform_anim = False

            if parent:
                has_transform_anim = self.detect_transform_animation(reader, parent, frame_count, fps, start_time)

            if has_transform_anim:
                logger.info(f"    -> transform only")
                result['transform_only'].append(display_name)
            else:
                logger.info(f"    -> static")
                result['static'].append(display_name)

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
            deltas = animation_data.get('vertex_animated_deltas', {})

            # Categorize by delta magnitude
            tiny_delta = []   # < 0.001 (likely floating point noise)
            small_delta = []  # 0.001 - 0.1 (small movement)
            large_delta = []  # > 0.1 (clear animation)

            for name in animation_data['vertex_animated']:
                delta = deltas.get(name, 0)
                if delta < 0.001:
                    tiny_delta.append((name, delta))
                elif delta < 0.1:
                    small_delta.append((name, delta))
                else:
                    large_delta.append((name, delta))

            lines.append("\n  Vertex Animated Meshes:")

            if large_delta:
                lines.append(f"    Clear animation (delta > 0.1): {len(large_delta)}")
                for name, delta in large_delta[:5]:  # Show first 5
                    lines.append(f"      - {name} (delta={delta:.4f})")
                if len(large_delta) > 5:
                    lines.append(f"      ... and {len(large_delta) - 5} more")

            if small_delta:
                lines.append(f"    Small movement (0.001 < delta < 0.1): {len(small_delta)}")
                for name, delta in small_delta[:3]:
                    lines.append(f"      - {name} (delta={delta:.6f})")
                if len(small_delta) > 3:
                    lines.append(f"      ... and {len(small_delta) - 3} more")

            if tiny_delta:
                lines.append(f"    Tiny delta (< 0.001, possible FP noise): {len(tiny_delta)}")
                for name, delta in tiny_delta[:3]:
                    lines.append(f"      - {name} (delta={delta:.8f})")
                if len(tiny_delta) > 3:
                    lines.append(f"      ... and {len(tiny_delta) - 3} more")

        return "\n".join(lines)
