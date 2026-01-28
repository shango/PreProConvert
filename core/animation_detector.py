#!/usr/bin/env python3
"""
Animation Detector Module
Detects animation types (transform vs vertex) to determine export strategy

Works with any BaseReader implementation (Alembic, USD, etc.)
"""


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
        Uses a sampling interval for efficiency on large frame ranges.

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

            # Sample every 5th frame for efficiency, or at least 5 frames
            # For 120 frame animation: sample every 6 frames (120/20 = 6)
            # For 30 frame animation: sample every 5 frames (min interval)
            sample_interval = max(5, frame_count // 20)

            # Check sampled frames for vertex position changes
            for frame in range(2, frame_count + 1, sample_interval):
                # Frame 1 = start_time, Frame 2 = start_time + 1/fps, etc.
                time_seconds = start_time + (frame - 1) / fps
                mesh_data = reader.get_mesh_data_at_time(mesh_obj, time_seconds)
                positions = mesh_data['positions']

                # Compare each vertex position to first frame
                for i in range(num_verts):
                    dx = abs(positions[i][0] - first_positions[i][0])
                    dy = abs(positions[i][1] - first_positions[i][1])
                    dz = abs(positions[i][2] - first_positions[i][2])

                    # If any vertex moved beyond tolerance, vertex animation detected
                    if dx > self.tolerance or dy > self.tolerance or dz > self.tolerance:
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

        parent_map = reader.get_parent_map()

        for mesh_obj in reader.get_meshes():
            mesh_name = mesh_obj.getName()

            # Check for vertex animation first (most important for AE)
            has_vertex_anim = self.detect_vertex_animation(reader, mesh_obj, frame_count, fps, start_time)

            if has_vertex_anim:
                result['vertex_animated'].append(mesh_name)
                continue

            # Check for transform animation on parent
            parent = parent_map.get(mesh_name)
            has_transform_anim = False

            if parent:
                has_transform_anim = self.detect_transform_animation(reader, parent, frame_count, fps, start_time)

            if has_transform_anim:
                result['transform_only'].append(mesh_name)
            else:
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
