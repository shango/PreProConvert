#!/usr/bin/env python3
"""
USD Exporter Module
Exports SceneData to USD (.usdc binary format) with vertex animation support

v2.5.0 - Refactored to use SceneData instead of reader objects.
         Now format-agnostic - works with any input format.
"""

from pathlib import Path

from .base_exporter import BaseExporter
from core.scene_data import SceneData, AnimationType


class USDExporter(BaseExporter):
    """USD exporter with full animation support

    Exports to USD (.usdc binary "crate" format) with:
    - Animated cameras (transform + focal length)
    - Meshes with transform-only animation
    - Meshes with vertex animation (time-sampled point positions)
    - Scene hierarchy preservation (v2.6.2)

    USD uses Y-up coordinate system (same as Alembic), so no conversion needed!

    v2.5.0: Now works with SceneData instead of reader objects - format-agnostic.
    v2.6.2: Added scene hierarchy preservation from full_path data.
    """

    def __init__(self, progress_callback=None):
        super().__init__(progress_callback)
        self.created_prims = set()  # Track created prim paths for hierarchy

        # Lazy import USD - only import when actually creating exporter instance
        # This avoids module-level import issues
        try:
            from pxr import Usd, UsdGeom, Gf, Vt, Sdf
            # Store in instance for use throughout the class
            self.Usd = Usd
            self.UsdGeom = UsdGeom
            self.Gf = Gf
            self.Vt = Vt
            self.Sdf = Sdf
        except ImportError as e:
            raise ImportError(
                f"USD Python library (pxr) not found: {e}\n"
                "Install with: pip install usd-core or download from NVIDIA (https://developer.nvidia.com/usd)"
            )

    def get_format_name(self):
        return "USD"

    def get_file_extension(self):
        return "usdc"

    def _extract_float3(self, value):
        """Extract three floats from a value, handling nested lists

        Args:
            value: A tuple/list like (x, y, z), [x, y, z], or [[x, y, z]]

        Returns:
            tuple: (float, float, float)
        """
        # Handle nested list edge case: [[x, y, z]] -> [x, y, z]
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (list, tuple)):
            value = value[0]
        return (float(value[0]), float(value[1]), float(value[2]))

    def _make_vec3d(self, value):
        """Create a Gf.Vec3d from potentially nested data"""
        x, y, z = self._extract_float3(value)
        return self.Gf.Vec3d(x, y, z)

    def _make_vec3f(self, value):
        """Create a Gf.Vec3f from potentially nested data"""
        x, y, z = self._extract_float3(value)
        return self.Gf.Vec3f(x, y, z)

    def export(self, scene_data: SceneData, output_path, shot_name):
        """Export to USD format

        Args:
            scene_data: SceneData instance with pre-extracted animation and geometry
            output_path: Output directory path
            shot_name: Shot name for file naming

        Returns:
            dict: Export results with keys:
                - 'success': bool
                - 'usd_file': Path to created USD file
                - 'vertex_animated_count': Number of meshes with vertex animation
                - 'message': Status message
        """
        try:
            # Reset state for this export
            self.created_prims = set()

            # Extract info from SceneData
            fps = scene_data.metadata.fps
            frame_count = scene_data.metadata.frame_count

            output_dir = self.validate_output_path(output_path)
            usd_file = output_dir / f"{shot_name}.usdc"

            self.log(f"Creating USD stage: {usd_file}")

            # Create USD stage (.usdc = binary crate format)
            stage = self.Usd.Stage.CreateNew(str(usd_file))

            # Set stage metadata
            stage.SetStartTimeCode(1)
            stage.SetEndTimeCode(frame_count)
            stage.SetTimeCodesPerSecond(fps)
            stage.SetFramesPerSecond(fps)

            # Set Y-up axis (same as Alembic!)
            self.UsdGeom.SetStageUpAxis(stage, self.UsdGeom.Tokens.y)

            self.log(f"Stage setup: {frame_count} frames @ {fps} fps, Y-up axis")

            # Create root transform
            root_xform = self.UsdGeom.Xform.Define(stage, "/World")
            self.created_prims.add("/World")

            # Process cameras with hierarchy preservation
            for camera in scene_data.cameras:
                cam_name = camera.parent_name if camera.parent_name else camera.name
                # Get hierarchical USD path from full_path
                usd_path = self._get_usd_path_from_full_path(camera.full_path, cam_name)
                self._ensure_hierarchy_exists(stage, usd_path)
                self.log(f"Exporting camera: {cam_name} -> {usd_path}")
                self._export_camera(stage, camera, usd_path, frame_count)
                self.created_prims.add(usd_path)

            # Track used paths to avoid conflicts
            used_paths = set(self.created_prims)

            # Process meshes with hierarchy preservation
            vertex_animated_count = 0
            for mesh in scene_data.meshes:
                mesh_name = mesh.parent_name if mesh.parent_name else mesh.name
                # Get hierarchical USD path from full_path
                usd_path = self._get_usd_path_from_full_path(mesh.full_path, mesh_name)
                self._ensure_hierarchy_exists(stage, usd_path)

                # Log geometry info for debugging
                geom = mesh.geometry
                verts = len(geom.positions) if geom and geom.positions else 0
                faces = len(geom.counts) if geom and geom.counts else 0
                if verts == 0:
                    self.log(f"WARNING: Mesh {mesh_name} has 0 vertices - will be invisible")

                if mesh.animation_type == AnimationType.VERTEX_ANIMATED:
                    self.log(f"Exporting mesh with vertex animation: {mesh_name} ({verts} verts, {faces} faces) -> {usd_path}")
                    self._export_mesh_with_vertex_anim(stage, mesh, usd_path, frame_count)
                    vertex_animated_count += 1
                else:
                    self.log(f"Exporting mesh (transform only): {mesh_name} ({verts} verts, {faces} faces) -> {usd_path}")
                    self._export_mesh_transform_only(stage, mesh, usd_path, frame_count)

                self.created_prims.add(usd_path)
                used_paths.add(usd_path)

            # Process transforms (locators/trackers) with hierarchy preservation
            locator_count = 0
            for transform in scene_data.transforms:
                xform_name = transform.name  # Always use locator's own name
                # Get hierarchical USD path from full_path
                usd_path = self._get_usd_path_from_full_path(transform.full_path, xform_name)

                # Skip if path conflicts with existing camera/mesh
                if usd_path in used_paths:
                    self.log(f"Skipping locator (path conflict): {xform_name}")
                    continue

                # Skip if no keyframes
                if not transform.keyframes:
                    self.log(f"Skipping locator (no keyframes): {xform_name}")
                    continue

                try:
                    self._ensure_hierarchy_exists(stage, usd_path)
                    self.log(f"Exporting locator: {xform_name} -> {usd_path}")
                    self._export_locator(stage, transform, usd_path, frame_count)
                    self.created_prims.add(usd_path)
                    used_paths.add(usd_path)
                    locator_count += 1
                except Exception as e:
                    self.log(f"Warning: Failed to export locator {xform_name}: {e}")

            # Save stage
            stage.Save()
            self.log(f"\n✓ USD file saved: {usd_file}")
            self.log(f"✓ Exported {len(scene_data.cameras)} cameras, {len(scene_data.meshes)} meshes, {locator_count} locators")
            self.log(f"✓ Vertex-animated meshes: {vertex_animated_count}")

            return {
                'success': True,
                'usd_file': str(usd_file),
                'vertex_animated_count': vertex_animated_count,
                'message': f"Exported {len(scene_data.cameras)} cameras, {len(scene_data.meshes)} meshes",
                'files': [str(usd_file)]
            }

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return {
                'success': False,
                'message': f"Export failed: {str(e)}",
                'files': []
            }

    def _export_camera(self, stage, camera, usd_path, frame_count):
        """Export animated camera to USD

        Args:
            stage: USD stage
            camera: CameraData instance from SceneData
            usd_path: Full USD prim path (e.g., "/World/Group/Camera")
            frame_count: Total frames
        """
        # Define camera at the given path
        usd_camera = self.UsdGeom.Camera.Define(stage, usd_path)

        # Set camera properties from CameraData.properties
        focal_length = camera.properties.focal_length
        h_aperture = camera.properties.h_aperture * 10  # cm to mm
        v_aperture = camera.properties.v_aperture * 10  # cm to mm

        usd_camera.GetFocalLengthAttr().Set(focal_length)
        usd_camera.GetHorizontalApertureAttr().Set(h_aperture)
        usd_camera.GetVerticalApertureAttr().Set(v_aperture)

        # Animate camera transform
        xformable = self.UsdGeom.Xformable(usd_camera)

        # Create transform ops with EXPLICIT precision for Maya compatibility
        translate_op = xformable.AddTranslateOp(self.UsdGeom.XformOp.PrecisionDouble)
        rotate_op = xformable.AddRotateXYZOp(self.UsdGeom.XformOp.PrecisionFloat)
        scale_op = xformable.AddScaleOp(self.UsdGeom.XformOp.PrecisionFloat)

        # Set default values FIRST to establish attributes (required for animation)
        if camera.keyframes:
            kf = camera.keyframes[0]
            translate_op.Set(self._make_vec3d(kf.position))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya))
            scale_op.Set(self._make_vec3f(kf.scale))

            # Log first frame values for debugging
            cam_name = usd_path.split('/')[-1]
            self.log(f"  Camera {cam_name} frame 1: pos={kf.position}, rot={kf.rotation_maya}")

        # THEN set time-sampled animation from pre-extracted keyframes
        first_kf, last_kf = None, None
        for kf in camera.keyframes:
            if kf.frame == 1:
                first_kf = kf
            if kf.frame == frame_count:
                last_kf = kf

            # USD uses same Y-up coordinate system - direct copy!
            # Use float for time code (matches USD convention)
            translate_op.Set(self._make_vec3d(kf.position), time=float(kf.frame))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya), time=float(kf.frame))
            scale_op.Set(self._make_vec3f(kf.scale), time=float(kf.frame))

        # Log animation range to verify data changes
        if first_kf and last_kf:
            cam_name = usd_path.split('/')[-1]
            self.log(f"  Camera {cam_name} animation check:")
            self.log(f"    Frame 1: pos={first_kf.position}, rot={first_kf.rotation_maya}")
            self.log(f"    Frame {frame_count}: pos={last_kf.position}, rot={last_kf.rotation_maya}")
            pos_changed = first_kf.position != last_kf.position
            rot_changed = first_kf.rotation_maya != last_kf.rotation_maya
            self.log(f"    Position animated: {pos_changed}, Rotation animated: {rot_changed}")

    def _export_mesh_transform_only(self, stage, mesh, usd_path, frame_count):
        """Export mesh with static geometry and animated transform

        Args:
            stage: USD stage
            mesh: MeshData instance from SceneData
            usd_path: Full USD prim path (e.g., "/World/Group/Mesh")
            frame_count: Total frames
        """
        # Define mesh at the given path
        usd_mesh = self.UsdGeom.Mesh.Define(stage, usd_path)

        # Set orientation to handle winding order difference from Alembic
        usd_mesh.GetOrientationAttr().Set("leftHanded")

        # Get mesh data from pre-extracted geometry
        geometry = mesh.geometry

        # Set static topology
        # Convert positions to USD format
        points = self.Vt.Vec3fArray([self._make_vec3f(p) for p in geometry.positions])
        usd_mesh.GetPointsAttr().Set(points)

        # Set face vertex indices
        indices = self.Vt.IntArray([int(i) for i in geometry.indices])
        usd_mesh.GetFaceVertexIndicesAttr().Set(indices)

        # Set face vertex counts
        counts = self.Vt.IntArray([int(c) for c in geometry.counts])
        usd_mesh.GetFaceVertexCountsAttr().Set(counts)

        # Animate transform
        xformable = self.UsdGeom.Xformable(usd_mesh)

        # Create transform ops with EXPLICIT precision for Maya compatibility
        translate_op = xformable.AddTranslateOp(self.UsdGeom.XformOp.PrecisionDouble)
        rotate_op = xformable.AddRotateXYZOp(self.UsdGeom.XformOp.PrecisionFloat)
        scale_op = xformable.AddScaleOp(self.UsdGeom.XformOp.PrecisionFloat)

        # Set default values FIRST to establish attributes (required for animation)
        if mesh.keyframes:
            kf = mesh.keyframes[0]
            translate_op.Set(self._make_vec3d(kf.position))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya))
            scale_op.Set(self._make_vec3f(kf.scale))

        # THEN set time-sampled animation from pre-extracted keyframes
        for kf in mesh.keyframes:
            # Y-up coordinate system - direct copy from source
            # Use float for time code (matches USD convention)
            translate_op.Set(self._make_vec3d(kf.position), time=float(kf.frame))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya), time=float(kf.frame))
            scale_op.Set(self._make_vec3f(kf.scale), time=float(kf.frame))

    def _export_mesh_with_vertex_anim(self, stage, mesh, usd_path, frame_count):
        """Export mesh with vertex animation (time-sampled point positions)

        Args:
            stage: USD stage
            mesh: MeshData instance from SceneData
            usd_path: Full USD prim path (e.g., "/World/Group/Mesh")
            frame_count: Total frames
        """
        # Define mesh at the given path
        usd_mesh = self.UsdGeom.Mesh.Define(stage, usd_path)

        # Set orientation to handle winding order difference from Alembic
        usd_mesh.GetOrientationAttr().Set("leftHanded")

        # Get static topology from pre-extracted geometry
        geometry = mesh.geometry

        # Set static topology (indices and counts don't change)
        indices = self.Vt.IntArray([int(i) for i in geometry.indices])
        usd_mesh.GetFaceVertexIndicesAttr().Set(indices)

        counts = self.Vt.IntArray([int(c) for c in geometry.counts])
        usd_mesh.GetFaceVertexCountsAttr().Set(counts)

        # Get points attribute for time-sampled animation
        points_attr = usd_mesh.GetPointsAttr()

        # Sample vertex positions from pre-extracted per-frame data
        if mesh.vertex_positions_per_frame:
            for frame, positions in mesh.vertex_positions_per_frame.items():
                # Convert positions to USD format
                points = self.Vt.Vec3fArray([self._make_vec3f(p) for p in positions])

                # Set time-sampled point positions (use float for time code)
                points_attr.Set(points, time=float(frame))
        else:
            # Fallback to static geometry if vertex_positions_per_frame not available
            points = self.Vt.Vec3fArray([self._make_vec3f(p) for p in geometry.positions])
            points_attr.Set(points)

        # Animate transform (if transform is also animated)
        xformable = self.UsdGeom.Xformable(usd_mesh)

        # Create transform ops with EXPLICIT precision for Maya compatibility
        translate_op = xformable.AddTranslateOp(self.UsdGeom.XformOp.PrecisionDouble)
        rotate_op = xformable.AddRotateXYZOp(self.UsdGeom.XformOp.PrecisionFloat)
        scale_op = xformable.AddScaleOp(self.UsdGeom.XformOp.PrecisionFloat)

        # Set default values FIRST to establish attributes (required for animation)
        if mesh.keyframes:
            kf = mesh.keyframes[0]
            translate_op.Set(self._make_vec3d(kf.position))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya))
            scale_op.Set(self._make_vec3f(kf.scale))

        # THEN set time-sampled animation from pre-extracted keyframes
        for kf in mesh.keyframes:
            # Y-up coordinate system - direct copy from source
            # Use float for time code (matches USD convention)
            translate_op.Set(self._make_vec3d(kf.position), time=float(kf.frame))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya), time=float(kf.frame))
            scale_op.Set(self._make_vec3f(kf.scale), time=float(kf.frame))

    def _export_locator(self, stage, transform, usd_path, frame_count):
        """Export animated locator/tracker to USD as pure Xform

        Creates a UsdGeom.Xform with animated transforms and no geometry.
        Importing applications (Maya, Houdini, etc.) will recognize this as
        a locator/null and display it with their native representation.

        Args:
            stage: USD stage
            transform: TransformData instance from SceneData
            usd_path: Full USD prim path (e.g., "/World/Group/Locator")
            frame_count: Total frames
        """
        # Define Xform only - no geometry child
        # DCCs will display this with their native locator/null representation
        usd_xform = self.UsdGeom.Xform.Define(stage, usd_path)

        # Get xformable for adding transform ops
        xformable = self.UsdGeom.Xformable(usd_xform)

        # Create transform ops with EXPLICIT precision for Maya compatibility
        translate_op = xformable.AddTranslateOp(self.UsdGeom.XformOp.PrecisionDouble)
        rotate_op = xformable.AddRotateXYZOp(self.UsdGeom.XformOp.PrecisionFloat)
        scale_op = xformable.AddScaleOp(self.UsdGeom.XformOp.PrecisionFloat)

        # Set default values FIRST to establish attributes
        if transform.keyframes:
            kf = transform.keyframes[0]
            translate_op.Set(self._make_vec3d(kf.position))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya))
            scale_op.Set(self._make_vec3f(kf.scale))

        # Set time-sampled animation
        for kf in transform.keyframes:
            translate_op.Set(self._make_vec3d(kf.position), time=float(kf.frame))
            rotate_op.Set(self._make_vec3f(kf.rotation_maya), time=float(kf.frame))
            scale_op.Set(self._make_vec3f(kf.scale), time=float(kf.frame))

    def _sanitize_name(self, name):
        """Sanitize name for USD prim path

        Args:
            name: Original name

        Returns:
            str: Sanitized name safe for USD paths
        """
        # Replace spaces and special characters
        sanitized = name.replace(' ', '_').replace('-', '_')
        # Remove other problematic characters
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'mesh_' + sanitized
        return sanitized or 'mesh'

    # === HIERARCHY UTILITIES ===

    def _get_usd_path_from_full_path(self, full_path, display_name):
        """Convert a full_path to a USD prim path with hierarchy

        Parses the source full_path (e.g., "/Group/SubGroup/ObjectShape")
        and constructs a USD path preserving hierarchy.

        For shapes (ending in "Shape"), uses the display_name (parent transform name)
        as the object name, with grandparent as the hierarchy parent.

        Args:
            full_path: Full hierarchy path from source (e.g., "/CameraRig/Camera/CameraShape")
            display_name: Display name for the object (e.g., "Camera" for a camera shape)

        Returns:
            str: USD prim path (e.g., "/World/CameraRig/Camera")
        """
        parts = [p for p in full_path.split('/') if p]
        if not parts:
            return f"/World/{self._sanitize_name(display_name)}"

        # Build path components
        sanitized_name = self._sanitize_name(display_name)

        # Determine hierarchy path
        # For shapes: /Group/Transform/Shape -> /World/Group/Transform
        # For transforms: /Group/Transform -> /World/Group/Transform
        # Detect shape nodes by comparing last path component to display_name
        obj_name = parts[-1]
        is_shape = (self._sanitize_name(obj_name) != sanitized_name) and len(parts) >= 2
        if is_shape:
            # Shape node - use parent path elements (excluding shape itself)
            hierarchy_parts = parts[:-1]  # Everything except the shape
        else:
            # Transform node - use all path elements
            hierarchy_parts = parts

        # If the last element of hierarchy matches display_name, use full hierarchy
        # Otherwise, replace last element with display_name
        if hierarchy_parts and self._sanitize_name(hierarchy_parts[-1]) == sanitized_name:
            path_parts = [self._sanitize_name(p) for p in hierarchy_parts]
        else:
            path_parts = [self._sanitize_name(p) for p in hierarchy_parts[:-1]] + [sanitized_name]

        return "/World/" + "/".join(path_parts) if path_parts else f"/World/{sanitized_name}"

    def _ensure_hierarchy_exists(self, stage, usd_path):
        """Ensure all parent Xforms exist for a given USD path

        Creates intermediate Xform prims for any missing parent paths.

        Args:
            stage: USD stage
            usd_path: Target USD prim path (e.g., "/World/Group/SubGroup/Object")
        """
        parts = usd_path.split('/')
        # Build paths incrementally, skipping empty parts and the final object
        current_path = ""
        for part in parts[1:-1]:  # Skip empty first part and final object
            current_path += "/" + part
            if current_path not in self.created_prims:
                # Create Xform for hierarchy group
                self.UsdGeom.Xform.Define(stage, current_path)
                self.created_prims.add(current_path)
                self.log(f"  Creating hierarchy group: {current_path}")
