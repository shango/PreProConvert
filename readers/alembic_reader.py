#!/usr/bin/env python3
"""
Alembic Reader Module
Centralized Alembic reading utilities implementing the BaseReader interface
"""

import numpy as np
from pathlib import Path

# Import Alembic libraries
from alembic.Abc import IArchive, ISampleSelector, WrapExistingFlag
from alembic.AbcGeom import IXform, ICamera, IPolyMesh
import imath

from .base_reader import BaseReader


class AlembicReader(BaseReader):
    """Centralized Alembic file reading and data extraction

    This class handles all Alembic file I/O operations and provides a clean
    interface for exporters to access scene data without duplicating code.

    Key principle: Read the Alembic file ONCE, then share data across all exporters.
    """

    def __init__(self, abc_file):
        """Open Alembic archive and initialize

        Args:
            abc_file: Path to Alembic (.abc) file
        """
        super().__init__(abc_file)
        self.archive = IArchive(str(self.file_path))
        self.top = self.archive.getTop()

    @property
    def abc_file(self):
        """Backward compatibility property"""
        return self.file_path

    def get_format_name(self):
        """Return human-readable format name"""
        return "Alembic"

    def get_archive(self):
        """Get the Alembic IArchive object"""
        return self.archive

    def get_top(self):
        """Get the top-level Alembic object"""
        return self.top

    def get_all_objects(self):
        """Get all objects in the Alembic hierarchy (cached)

        Returns:
            list: All Alembic objects in the scene
        """
        if self._objects_cache is None:
            self._objects_cache = []
            self._collect_objects_recursive(self.top, self._objects_cache)
        return self._objects_cache

    def _collect_objects_recursive(self, obj, objects_list):
        """Recursively collect all objects in hierarchy"""
        objects_list.append(obj)
        for child in obj.children:
            self._collect_objects_recursive(child, objects_list)

    def get_cameras(self):
        """Get all camera objects in the scene

        Returns:
            list: ICamera objects
        """
        cameras = []
        for obj in self.get_all_objects():
            if ICamera.matches(obj.getHeader()):
                cameras.append(obj)
        return cameras

    def get_meshes(self):
        """Get all mesh objects in the scene

        Returns:
            list: IPolyMesh objects
        """
        meshes = []
        for obj in self.get_all_objects():
            if IPolyMesh.matches(obj.getHeader()):
                meshes.append(obj)
        return meshes

    def get_transforms(self):
        """Get all transform objects in the scene

        Returns:
            list: IXform objects
        """
        transforms = []
        for obj in self.get_all_objects():
            if IXform.matches(obj.getHeader()):
                transforms.append(obj)
        return transforms

    def get_parent_map(self):
        """Build parent-child relationship map (cached)

        Returns:
            dict: Mapping of child name -> parent object
        """
        if self._parent_map_cache is None:
            self._parent_map_cache = {}
            for obj in self.get_all_objects():
                for child in obj.children:
                    self._parent_map_cache[child.getName()] = obj
        return self._parent_map_cache

    def detect_frame_count(self, fps=24):
        """Auto-detect frame count from Alembic time sampling

        Args:
            fps: Frames per second (used for fallback calculation)

        Returns:
            int: Number of frames in the animation
        """
        try:
            # Get time sampling
            num_time_samplings = self.archive.getNumTimeSamplings()
            if num_time_samplings > 1:
                # Use the first non-uniform time sampling (index 1)
                time_sampling = self.archive.getTimeSampling(1)
                num_samples = self.archive.getMaxNumSamplesForTimeSamplingIndex(1)

                if num_samples > 0:
                    return num_samples

            # Fallback: assume 120 frames if we can't detect
            return 120
        except Exception:
            return 120

    def get_transform_at_time(self, obj, time_seconds, maya_compat=False):
        """Get transform data (position, rotation, scale) at a specific time

        This extracts scale from the LOCAL transform matrix, then gets world position/rotation.
        SynthEyes stores per-object scale in the local matrix.

        Args:
            obj: Alembic object (should be IXform or have parent IXform)
            time_seconds: Time in seconds to sample
            maya_compat: If True, use Maya-compatible rotation decomposition (row-major)

        Returns:
            tuple: (translation, rotation, scale) where:
                - translation: [x, y, z] in Alembic units (cm)
                - rotation: [rx, ry, rz] in degrees (XYZ Euler)
                - scale: [sx, sy, sz] as multipliers (NOT multiplied by 100)
        """
        # First, get the LOCAL matrix of this object to extract scale
        local_scale = None
        if IXform.matches(obj.getHeader()):
            xform = IXform(obj, WrapExistingFlag.kWrapExisting)
            schema = xform.getSchema()

            sample_sel = ISampleSelector(time_seconds)
            xf_sample = schema.getValue(sample_sel)
            local_matrix = xf_sample.getMatrix()

            # Extract scale from LOCAL matrix (this is where SynthEyes stores it)
            m = np.array(local_matrix)
            sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
            sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
            sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
            local_scale = [sx, sy, sz]

        # Now accumulate world matrix for position and rotation
        matrices = []
        current = obj

        while current:
            if IXform.matches(current.getHeader()):
                xform = IXform(current, WrapExistingFlag.kWrapExisting)
                schema = xform.getSchema()

                sample_sel = ISampleSelector(time_seconds)
                xf_sample = schema.getValue(sample_sel)
                matrices.append(xf_sample.getMatrix())

            parent = current.getParent()
            if parent and parent.getName() != "ABC":
                current = parent
            else:
                break

        # Combine transforms for world matrix
        world_matrix = imath.M44d()
        world_matrix.makeIdentity()

        for mat in reversed(matrices):
            world_matrix = world_matrix * mat

        # Decompose world matrix for position and rotation
        pos, rot, world_scale = self._decompose_matrix(world_matrix, maya_compat=maya_compat)

        # Use local scale (from the object's own matrix), not world scale
        # This prevents parent scales from affecting the object's scale
        final_scale = local_scale if local_scale is not None else world_scale

        return pos, rot, final_scale

    def get_transform_with_matrix_at_time(self, obj, time_seconds):
        """Get transform data with rotation matrix for lazy decomposition

        Returns the normalized 3x3 rotation matrix instead of decomposed angles,
        enabling lazy rotation decomposition in Keyframe.

        Args:
            obj: Alembic object
            time_seconds: Time in seconds to sample

        Returns:
            tuple: (translation, rotation_matrix, scale) where:
                - translation: [x, y, z]
                - rotation_matrix: 9 floats (3x3 row-major normalized rotation matrix)
                - scale: [sx, sy, sz]
        """
        # First, get the LOCAL matrix of this object to extract scale
        local_scale = None
        if IXform.matches(obj.getHeader()):
            xform = IXform(obj, WrapExistingFlag.kWrapExisting)
            schema = xform.getSchema()

            sample_sel = ISampleSelector(time_seconds)
            xf_sample = schema.getValue(sample_sel)
            local_matrix = xf_sample.getMatrix()

            # Extract scale from LOCAL matrix
            m = np.array(local_matrix)
            sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
            sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
            sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
            local_scale = [sx, sy, sz]

        # Accumulate world matrix for position and rotation
        matrices = []
        current = obj

        while current:
            if IXform.matches(current.getHeader()):
                xform = IXform(current, WrapExistingFlag.kWrapExisting)
                schema = xform.getSchema()

                sample_sel = ISampleSelector(time_seconds)
                xf_sample = schema.getValue(sample_sel)
                matrices.append(xf_sample.getMatrix())

            parent = current.getParent()
            if parent and parent.getName() != "ABC":
                current = parent
            else:
                break

        # Combine transforms for world matrix
        world_matrix = imath.M44d()
        world_matrix.makeIdentity()

        for mat in reversed(matrices):
            world_matrix = world_matrix * mat

        # Decompose matrix - extract position, rotation matrix, and scale
        m = np.array(world_matrix)

        # Extract translation
        translation = [m[3][0], m[3][1], m[3][2]]

        # Extract scale from row lengths
        sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
        sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
        sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
        world_scale = [sx, sy, sz]

        # Build normalized rotation matrix (3x3 as 9 floats, row-major)
        rotation_matrix = (
            m[0][0] / sx if sx > 0 else m[0][0],
            m[0][1] / sx if sx > 0 else m[0][1],
            m[0][2] / sx if sx > 0 else m[0][2],
            m[1][0] / sy if sy > 0 else m[1][0],
            m[1][1] / sy if sy > 0 else m[1][1],
            m[1][2] / sy if sy > 0 else m[1][2],
            m[2][0] / sz if sz > 0 else m[2][0],
            m[2][1] / sz if sz > 0 else m[2][1],
            m[2][2] / sz if sz > 0 else m[2][2],
        )

        # Use local scale if available
        final_scale = local_scale if local_scale is not None else world_scale

        return translation, rotation_matrix, final_scale

    def get_mesh_data_at_time(self, mesh_obj, time_seconds):
        """Get mesh geometry data at a specific time

        Args:
            mesh_obj: IPolyMesh object
            time_seconds: Time in seconds to sample

        Returns:
            dict: Mesh data with keys:
                - 'positions': Vertex positions as list of [x, y, z]
                - 'indices': Face vertex indices
                - 'counts': Face vertex counts
        """
        poly = IPolyMesh(mesh_obj, WrapExistingFlag.kWrapExisting)
        schema = poly.getSchema()

        sample_sel = ISampleSelector(time_seconds)
        sample = schema.getValue(sample_sel)

        positions = sample.getPositions()
        indices = sample.getFaceIndices()
        counts = sample.getFaceCounts()

        return {
            'positions': positions,
            'indices': indices,
            'counts': counts
        }

    def get_camera_properties(self, cam_obj, time_seconds=None):
        """Get camera properties at a specific time

        Args:
            cam_obj: ICamera object
            time_seconds: Time in seconds (None for first sample)

        Returns:
            dict: Camera properties with keys:
                - 'focal_length': Focal length in mm
                - 'h_aperture': Horizontal aperture in cm
                - 'v_aperture': Vertical aperture in cm
        """
        camera = ICamera(cam_obj, WrapExistingFlag.kWrapExisting)
        schema = camera.getSchema()

        if time_seconds is not None:
            sample_sel = ISampleSelector(time_seconds)
            sample = schema.getValue(sample_sel)
        else:
            sample = schema.getValue()

        return {
            'focal_length': sample.getFocalLength(),
            'h_aperture': sample.getHorizontalAperture(),
            'v_aperture': sample.getVerticalAperture()
        }

    def extract_footage_path(self):
        """Extract footage file path from Alembic camera metadata

        Returns:
            str: Footage file path, or None if not found
        """
        try:
            # Search for camera objects to extract footage path from metadata
            def find_camera(obj):
                if ICamera.matches(obj.getHeader()):
                    return obj
                for child in obj.children:
                    cam = find_camera(child)
                    if cam:
                        return cam
                return None

            camera_obj = find_camera(self.top)
            if camera_obj:
                camera = ICamera(camera_obj, WrapExistingFlag.kWrapExisting)
                schema = camera.getSchema()

                # Check for arbGeomParams (custom properties)
                if schema.getArbGeomParams():
                    arb_params = schema.getArbGeomParams()
                    # Try common property names for footage file paths
                    for prop_name in ['footagePath', 'footage', 'sourceFile', 'imagePath',
                                     'videoFile', 'mediaPath', 'sourceImage', 'backgroundImage']:
                        try:
                            if arb_params.getPropertyHeader(prop_name):
                                prop = arb_params.getProperty(prop_name)
                                if prop.valid():
                                    val = prop.getValue()
                                    if val and isinstance(val, str):
                                        return val
                        except:
                            pass

                # Check user properties
                if schema.getUserProperties():
                    user_props = schema.getUserProperties()
                    for prop_name in ['footagePath', 'footage', 'sourceFile', 'imagePath',
                                     'videoFile', 'mediaPath', 'sourceImage', 'backgroundImage']:
                        try:
                            if user_props.getPropertyHeader(prop_name):
                                prop = user_props.getProperty(prop_name)
                                if prop.valid():
                                    val = prop.getValue()
                                    if val and isinstance(val, str):
                                        return val
                        except:
                            pass

        except Exception:
            pass

        return None

    def extract_render_resolution(self):
        """Extract render resolution from Alembic camera metadata

        Returns:
            tuple: (width, height) in pixels, or (1920, 1080) as fallback
        """
        try:
            # Search for camera to extract resolution
            def find_camera(obj):
                if ICamera.matches(obj.getHeader()):
                    return obj
                for child in obj.children:
                    cam = find_camera(child)
                    if cam:
                        return cam
                return None

            camera_obj = find_camera(self.top)
            if camera_obj:
                camera = ICamera(camera_obj, WrapExistingFlag.kWrapExisting)
                schema = camera.getSchema()
                cam_sample = schema.getValue()

                # Get aperture (film back dimensions)
                h_aperture_cm = cam_sample.getHorizontalAperture()
                v_aperture_cm = cam_sample.getVerticalAperture()

                # Use standard HD resolution as default
                return (1920, 1080)

        except Exception:
            pass

        # Fallback to HD resolution
        return (1920, 1080)

    def _decompose_matrix(self, matrix, maya_compat=False):
        """Decompose a 4x4 matrix into translation, rotation (XYZ Euler), and scale

        Alembic uses row-major matrices with post-multiplication (v' = v * M).
        Two decomposition modes are available:
        - Default (maya_compat=False): Column-major Euler extraction for After Effects
        - Maya mode (maya_compat=True): Row-major Euler extraction for Maya/USD

        Args:
            matrix: 4x4 transformation matrix (row-major)
            maya_compat: If True, use Maya-compatible row-major rotation extraction

        Returns:
            tuple: (translation, rotation, scale) where:
                - translation: [x, y, z]
                - rotation: [rx, ry, rz] in degrees (XYZ Euler)
                - scale: [sx, sy, sz]
        """
        m = np.array(matrix)

        # Extract translation (row 3 contains translation in row-major format)
        translation = [m[3][0], m[3][1], m[3][2]]

        # Extract scale from row lengths (row-major: rows are transformed basis vectors)
        sx = np.linalg.norm([m[0][0], m[0][1], m[0][2]])
        sy = np.linalg.norm([m[1][0], m[1][1], m[1][2]])
        sz = np.linalg.norm([m[2][0], m[2][1], m[2][2]])
        scale = [sx, sy, sz]

        # Build normalized rotation matrix
        rot = np.zeros((3, 3))
        rot[0][0] = m[0][0] / sx if sx > 0 else m[0][0]
        rot[0][1] = m[0][1] / sx if sx > 0 else m[0][1]
        rot[0][2] = m[0][2] / sx if sx > 0 else m[0][2]
        rot[1][0] = m[1][0] / sy if sy > 0 else m[1][0]
        rot[1][1] = m[1][1] / sy if sy > 0 else m[1][1]
        rot[1][2] = m[1][2] / sy if sy > 0 else m[1][2]
        rot[2][0] = m[2][0] / sz if sz > 0 else m[2][0]
        rot[2][1] = m[2][1] / sz if sz > 0 else m[2][1]
        rot[2][2] = m[2][2] / sz if sz > 0 else m[2][2]

        # Extract XYZ Euler angles from rotation matrix
        if maya_compat:
            # Row-major decomposition for Maya/USD compatibility
            cy = np.sqrt(rot[0][0]**2 + rot[0][1]**2)

            if cy > 1e-6:
                # Normal case
                x = np.arctan2(rot[1][2], rot[2][2])
                y = np.arctan2(-rot[0][2], cy)
                z = np.arctan2(-rot[0][1], rot[0][0])  # Negated for correct sign
            else:
                # Gimbal lock case
                x = np.arctan2(-rot[2][1], rot[1][1])
                y = np.arctan2(-rot[0][2], cy)
                z = 0
        else:
            # Column-major decomposition for After Effects compatibility
            sy_test = np.sqrt(rot[0][0]**2 + rot[1][0]**2)

            if sy_test > 1e-6:
                # Normal case
                x = np.arctan2(rot[2][1], rot[2][2])
                y = np.arctan2(-rot[2][0], sy_test)
                z = np.arctan2(rot[1][0], rot[0][0])
            else:
                # Gimbal lock case
                x = np.arctan2(-rot[1][2], rot[1][1])
                y = np.arctan2(-rot[2][0], sy_test)
                z = 0

        rotation = [np.degrees(x), np.degrees(y), np.degrees(z)]

        return translation, rotation, scale

    def _get_full_path(self, obj):
        """Get full hierarchy path for an Alembic object

        Args:
            obj: Alembic object

        Returns:
            str: Full path like "/World/Camera/CameraShape"
        """
        return obj.getFullName()

    def _is_organizational_group(self, obj):
        """Check if transform is just an organizational container

        Detects groups that only contain other transforms but no direct shapes.

        Args:
            obj: Alembic object

        Returns:
            bool: True if object is organizational only
        """
        if not IXform.matches(obj.getHeader()):
            return False

        xform = IXform(obj, WrapExistingFlag.kWrapExisting)
        schema = xform.getSchema()

        num_samples = schema.getNumSamples()
        if num_samples <= 1:
            has_direct_shape = False
            has_children = False
            for child in obj.children:
                has_children = True
                if ICamera.matches(child.getHeader()) or IPolyMesh.matches(child.getHeader()):
                    has_direct_shape = True
                    break

            if has_children and not has_direct_shape:
                return True

        return False
