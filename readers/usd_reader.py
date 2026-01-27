#!/usr/bin/env python3
"""
USD Reader Module
Centralized USD reading utilities implementing the BaseReader interface
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from .base_reader import BaseReader


class USDPrimWrapper:
    """Wrapper for USD prims to provide consistent interface with Alembic objects

    Provides getName(), getFullName(), getParent()-like interface for compatibility
    with existing code that expects Alembic-style object access.
    """

    def __init__(self, prim):
        """Initialize wrapper with USD prim

        Args:
            prim: USD Prim object
        """
        self.prim = prim
        self._children = None
        self._UsdGeom = None

    def _get_usd_geom(self):
        """Lazy import of UsdGeom"""
        if self._UsdGeom is None:
            from pxr import UsdGeom
            self._UsdGeom = UsdGeom
        return self._UsdGeom

    def getName(self) -> str:
        """Get the prim's name (last component of path)"""
        return self.prim.GetName()

    def getFullName(self) -> str:
        """Get the full prim path"""
        return str(self.prim.GetPath())

    def getParent(self):
        """Get the parent wrapper, or None if at root"""
        from pxr import Sdf
        parent_prim = self.prim.GetParent()
        if parent_prim and parent_prim.GetPath() != Sdf.Path.absoluteRootPath:
            return USDPrimWrapper(parent_prim)
        return None

    def getHeader(self):
        """Return self for type checking compatibility"""
        return self

    @property
    def children(self):
        """Get child prim wrappers"""
        if self._children is None:
            self._children = [USDPrimWrapper(child) for child in self.prim.GetChildren()]
        return self._children

    def IsCamera(self) -> bool:
        """Check if this prim is a camera"""
        UsdGeom = self._get_usd_geom()
        return self.prim.IsA(UsdGeom.Camera)

    def IsMesh(self) -> bool:
        """Check if this prim is a mesh"""
        UsdGeom = self._get_usd_geom()
        return self.prim.IsA(UsdGeom.Mesh)

    def IsXform(self) -> bool:
        """Check if this prim is a transform (Xform or Xformable but not Camera/Mesh)"""
        UsdGeom = self._get_usd_geom()
        return self.prim.IsA(UsdGeom.Xformable) and not self.IsCamera() and not self.IsMesh()


class USDReader(BaseReader):
    """USD file reader implementing the BaseReader interface

    Reads USD files (.usd, .usda, .usdc) and provides the same interface
    as AlembicReader for seamless integration with existing exporters.
    """

    def __init__(self, usd_file: str):
        """Open USD stage and initialize

        Args:
            usd_file: Path to USD file (.usd, .usda, .usdc)
        """
        super().__init__(usd_file)

        # Import USD libraries
        try:
            from pxr import Usd, UsdGeom, Gf
            self.Usd = Usd
            self.UsdGeom = UsdGeom
            self.Gf = Gf
        except ImportError as e:
            raise ImportError(
                f"USD Python library (pxr) not found: {e}\n"
                "Install with: pip install usd-core"
            )

        self.stage = Usd.Stage.Open(str(self.file_path))
        if not self.stage:
            raise ValueError(f"Failed to open USD file: {usd_file}")

        # Get stage timing info
        self._fps = self.stage.GetTimeCodesPerSecond() or 24.0
        self._start_time = self.stage.GetStartTimeCode()
        self._end_time = self.stage.GetEndTimeCode()

        # Handle default time codes
        if self._start_time is None or self._start_time == float('inf'):
            self._start_time = 1.0
        if self._end_time is None or self._end_time == float('-inf'):
            self._end_time = 1.0

    def get_format_name(self) -> str:
        """Return human-readable format name"""
        return "USD"

    def get_all_objects(self) -> List[USDPrimWrapper]:
        """Get all objects in the USD hierarchy (cached)

        Returns:
            list: All USD prim wrappers in the scene
        """
        if self._objects_cache is None:
            self._objects_cache = []
            for prim in self.stage.Traverse():
                self._objects_cache.append(USDPrimWrapper(prim))
        return self._objects_cache

    def get_cameras(self) -> List[USDPrimWrapper]:
        """Get all camera objects in the scene

        Returns:
            list: Camera prim wrappers
        """
        cameras = []
        for obj in self.get_all_objects():
            if obj.IsCamera():
                cameras.append(obj)
        return cameras

    def get_meshes(self) -> List[USDPrimWrapper]:
        """Get all mesh objects in the scene

        Returns:
            list: Mesh prim wrappers
        """
        meshes = []
        for obj in self.get_all_objects():
            if obj.IsMesh():
                meshes.append(obj)
        return meshes

    def get_transforms(self) -> List[USDPrimWrapper]:
        """Get all transform objects in the scene

        Returns:
            list: Transform prim wrappers
        """
        transforms = []
        for obj in self.get_all_objects():
            if obj.IsXform():
                transforms.append(obj)
        return transforms

    def get_parent_map(self) -> Dict[str, USDPrimWrapper]:
        """Build parent-child relationship map (cached)

        Returns:
            dict: Mapping of child name -> parent prim wrapper
        """
        if self._parent_map_cache is None:
            self._parent_map_cache = {}
            for obj in self.get_all_objects():
                for child in obj.children:
                    self._parent_map_cache[child.getName()] = obj
        return self._parent_map_cache

    def detect_frame_count(self, fps: int = 24) -> int:
        """Auto-detect frame count from USD time sampling

        Args:
            fps: Frames per second (used for calculation if needed)

        Returns:
            int: Number of frames in the animation
        """
        try:
            start = self._start_time
            end = self._end_time

            if end > start:
                return int(end - start + 1)

            return 120  # Fallback
        except Exception:
            return 120

    def _time_seconds_to_time_code(self, time_seconds: float):
        """Convert time in seconds to USD TimeCode

        Args:
            time_seconds: Time in seconds

        Returns:
            USD TimeCode
        """
        frame = time_seconds * self._fps
        return self.Usd.TimeCode(frame)

    def get_transform_at_time(self, obj: USDPrimWrapper, time_seconds: float,
                              maya_compat: bool = False) -> Tuple[List[float], List[float], List[float]]:
        """Get transform data (position, rotation, scale) at a specific time

        Args:
            obj: USDPrimWrapper object
            time_seconds: Time in seconds to sample
            maya_compat: If True, use Maya-compatible rotation decomposition

        Returns:
            tuple: (translation, rotation, scale) where:
                - translation: [x, y, z]
                - rotation: [rx, ry, rz] in degrees (XYZ Euler)
                - scale: [sx, sy, sz]
        """
        time_code = self._time_seconds_to_time_code(time_seconds)

        xformable = self.UsdGeom.Xformable(obj.prim)

        # Get local transformation matrix for scale extraction
        local_matrix = xformable.GetLocalTransformation(time_code)
        local_scale = self._extract_scale_from_matrix(local_matrix)

        # Get world transform for position and rotation
        world_matrix = xformable.ComputeLocalToWorldTransform(time_code)
        pos, rot, _ = self._decompose_matrix(world_matrix, maya_compat=maya_compat)

        return pos, rot, local_scale

    def _extract_scale_from_matrix(self, matrix) -> List[float]:
        """Extract scale from transformation matrix

        Args:
            matrix: USD Gf.Matrix4d

        Returns:
            list: [sx, sy, sz] scale values
        """
        m = np.array(matrix)
        sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
        sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
        sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
        return [sx, sy, sz]

    def _decompose_matrix(self, matrix, maya_compat: bool = False) -> Tuple[List[float], List[float], List[float]]:
        """Decompose a 4x4 matrix into translation, rotation, scale

        Uses the same decomposition logic as AlembicReader for consistency.

        Args:
            matrix: USD Gf.Matrix4d
            maya_compat: If True, use Maya-compatible rotation decomposition

        Returns:
            tuple: (translation, rotation, scale)
        """
        m = np.array(matrix)

        # Extract translation (row 3 in row-major format)
        translation = [m[3][0], m[3][1], m[3][2]]

        # Extract scale
        sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
        sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
        sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
        scale = [sx, sy, sz]

        # Build normalized rotation matrix
        rot = np.zeros((3, 3))
        rot[0] = [m[0][i] / sx if sx > 0 else m[0][i] for i in range(3)]
        rot[1] = [m[1][i] / sy if sy > 0 else m[1][i] for i in range(3)]
        rot[2] = [m[2][i] / sz if sz > 0 else m[2][i] for i in range(3)]

        # Extract XYZ Euler angles
        if maya_compat:
            # Row-major decomposition for Maya/USD compatibility
            cy = np.sqrt(rot[0][0]**2 + rot[0][1]**2)
            if cy > 1e-6:
                x = np.arctan2(rot[1][2], rot[2][2])
                y = np.arctan2(-rot[0][2], cy)
                z = np.arctan2(rot[0][1], rot[0][0])
            else:
                x = np.arctan2(-rot[2][1], rot[1][1])
                y = np.arctan2(-rot[0][2], cy)
                z = 0
        else:
            # Column-major decomposition for After Effects compatibility
            sy_test = np.sqrt(rot[0][0]**2 + rot[1][0]**2)
            if sy_test > 1e-6:
                x = np.arctan2(rot[2][1], rot[2][2])
                y = np.arctan2(-rot[2][0], sy_test)
                z = np.arctan2(rot[1][0], rot[0][0])
            else:
                x = np.arctan2(-rot[1][2], rot[1][1])
                y = np.arctan2(-rot[2][0], sy_test)
                z = 0

        rotation = [np.degrees(x), np.degrees(y), np.degrees(z)]

        return translation, rotation, scale

    def get_mesh_data_at_time(self, mesh_obj: USDPrimWrapper, time_seconds: float) -> Dict[str, Any]:
        """Get mesh geometry data at a specific time

        Args:
            mesh_obj: USDPrimWrapper for mesh
            time_seconds: Time in seconds to sample

        Returns:
            dict: Mesh data with keys:
                - 'positions': Vertex positions as list of [x, y, z]
                - 'indices': Face vertex indices
                - 'counts': Face vertex counts
        """
        time_code = self._time_seconds_to_time_code(time_seconds)

        mesh = self.UsdGeom.Mesh(mesh_obj.prim)

        # Get points (vertices)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get(time_code)

        # Get face vertex indices
        indices_attr = mesh.GetFaceVertexIndicesAttr()
        indices = indices_attr.Get(time_code)

        # Get face vertex counts
        counts_attr = mesh.GetFaceVertexCountsAttr()
        counts = counts_attr.Get(time_code)

        # Convert to lists for consistency with AlembicReader
        return {
            'positions': list(points) if points else [],
            'indices': list(indices) if indices else [],
            'counts': list(counts) if counts else []
        }

    def get_camera_properties(self, cam_obj: USDPrimWrapper,
                              time_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get camera properties at a specific time

        Args:
            cam_obj: USDPrimWrapper for camera
            time_seconds: Time in seconds (None for default)

        Returns:
            dict: Camera properties with keys:
                - 'focal_length': Focal length in mm
                - 'h_aperture': Horizontal aperture in cm
                - 'v_aperture': Vertical aperture in cm
        """
        if time_seconds is not None:
            time_code = self._time_seconds_to_time_code(time_seconds)
        else:
            time_code = self.Usd.TimeCode.Default()

        camera = self.UsdGeom.Camera(cam_obj.prim)

        # Get camera attributes
        focal_length = camera.GetFocalLengthAttr().Get(time_code)
        h_aperture_mm = camera.GetHorizontalApertureAttr().Get(time_code)
        v_aperture_mm = camera.GetVerticalApertureAttr().Get(time_code)

        # Provide defaults if not set
        if focal_length is None:
            focal_length = 35.0
        if h_aperture_mm is None:
            h_aperture_mm = 36.0
        if v_aperture_mm is None:
            v_aperture_mm = 24.0

        # USD stores aperture in mm, convert to cm for consistency with Alembic
        return {
            'focal_length': focal_length,
            'h_aperture': h_aperture_mm / 10.0,  # mm to cm
            'v_aperture': v_aperture_mm / 10.0   # mm to cm
        }

    def extract_render_resolution(self) -> Tuple[int, int]:
        """Extract render resolution from USD metadata or camera

        Returns:
            tuple: (width, height) in pixels, or (1920, 1080) as fallback
        """
        # Try to get resolution from render settings or camera
        # Default to HD resolution
        return (1920, 1080)

    def _get_full_path(self, obj: USDPrimWrapper) -> str:
        """Get full hierarchy path for a USD object

        Args:
            obj: USDPrimWrapper object

        Returns:
            str: Full path like "/World/Camera/CameraShape"
        """
        return obj.getFullName()

    def _is_organizational_group(self, obj: USDPrimWrapper) -> bool:
        """Check if transform is just an organizational container

        Args:
            obj: USDPrimWrapper object

        Returns:
            bool: True if object is organizational only
        """
        if not obj.IsXform():
            return False

        name = obj.getName()

        # Root container names should be treated as organizational
        # These are common USD root group names
        if name in ('World', 'Root', 'Scene', 'root', 'world', 'scene'):
            return True

        # Check if it has direct shape children
        has_direct_shape = False
        has_children = False
        for child in obj.children:
            has_children = True
            if child.IsCamera() or child.IsMesh():
                has_direct_shape = True
                break

        # If has children but no direct shapes, it's organizational
        return has_children and not has_direct_shape
