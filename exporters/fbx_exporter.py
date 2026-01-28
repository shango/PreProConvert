"""
FBX ASCII exporter for Unreal Engine

v2.6.2 - Pure Python FBX ASCII writer, no external SDK required.

Export Strategy:
- Cameras: Full camera nodes with transform animation
- Meshes: Static geometry with transform animation (skip vertex-animated)
- Locators: Null nodes with animation
- Coordinate system: Y-up to Z-up conversion for Unreal Engine

FBX ASCII format based on FBX 7.4 specification.
"""

import re
from pathlib import Path
from datetime import datetime
from exporters.base_exporter import BaseExporter
from core.scene_data import SceneData, AnimationType


class FBXExporter(BaseExporter):
    """FBX ASCII file exporter for Unreal Engine

    v2.6.2: Added scene hierarchy preservation from full_path data.
    """

    def __init__(self, progress_callback=None):
        super().__init__(progress_callback)
        self.shot_name = ""
        self.fps = 24.0
        self.frame_count = 1

        # Object ID tracking (FBX uses unique 64-bit IDs)
        # Start at 1000000001 to reserve 1000000000 for Document
        self._next_id = 1000000001
        self._object_ids = {}  # name -> id mapping
        self._connections = []  # List of (child_id, parent_id, property) tuples
        self._created_groups = set()  # Track created hierarchy group names

    def _get_id(self, name):
        """Get or create unique ID for an object"""
        if name not in self._object_ids:
            self._object_ids[name] = self._next_id
            self._next_id += 1
        return self._object_ids[name]

    def get_format_name(self):
        return "FBX"

    def get_file_extension(self):
        return "fbx"

    def export(self, scene_data: SceneData, output_path, shot_name):
        """Main export method using SceneData

        Args:
            scene_data: SceneData instance with all pre-extracted animation
            output_path: Output directory
            shot_name: Shot/scene name
        """
        try:
            self.shot_name = shot_name
            self.fps = scene_data.metadata.fps
            self.frame_count = scene_data.metadata.frame_count
            self._object_ids = {}
            self._connections = []
            # Start at 1000000001 to reserve 1000000000 for Document
            self._next_id = 1000000001
            self._created_groups = set()

            self.log(f"Exporting FBX format for Unreal Engine...")

            output_dir = Path(output_path)
            self.validate_output_path(output_dir)
            fbx_file = output_dir / f"{shot_name}.fbx"

            lines = []

            # === FBX HEADER ===
            lines.extend(self._write_header())

            # === GLOBAL SETTINGS (Z-up for UE) ===
            lines.extend(self._write_global_settings())

            # === DOCUMENTS ===
            lines.extend(self._write_documents())

            # === REFERENCES ===
            lines.extend(self._write_references())

            # === HIERARCHY SETUP ===
            # Use centralized hierarchy from SceneData if available (new architecture)
            # Fall back to local methods for backward compatibility
            if scene_data.hierarchy:
                hierarchy_groups = scene_data.hierarchy.get_intermediate_groups()
            else:
                hierarchy_groups = self._get_hierarchy_groups(scene_data)

            # === DEFINITIONS ===
            # Count objects for definitions
            num_cameras = len(scene_data.cameras)
            num_meshes = sum(1 for m in scene_data.meshes
                           if m.animation_type != AnimationType.VERTEX_ANIMATED)
            num_locators = len(scene_data.transforms)
            num_groups = len(hierarchy_groups)

            # Count blend shape objects
            num_blend_shape_deformers = 0
            num_blend_shape_channels = 0
            num_shape_geometries = 0
            for mesh in scene_data.meshes:
                if mesh.animation_type == AnimationType.BLEND_SHAPE and mesh.blend_shapes:
                    num_blend_shape_deformers += 1
                    for channel in mesh.blend_shapes.channels:
                        num_blend_shape_channels += 1
                        num_shape_geometries += len(channel.targets)

            # Count animation curve nodes and curves
            num_anim_curve_nodes, num_anim_curves = self._count_animation_curves(scene_data)

            lines.extend(self._write_definitions(
                num_cameras, num_meshes, num_locators, num_groups,
                num_blend_shape_deformers, num_blend_shape_channels, num_shape_geometries,
                num_anim_curve_nodes, num_anim_curves
            ))

            # === PRE-REGISTER ALL MODEL IDS ===
            # This ensures parent checks work regardless of object write order
            # (e.g., cameras written before locators can still find locator parents)
            for group_name, _ in hierarchy_groups:
                self._get_id(f"Model::{group_name}")
            for cam in scene_data.cameras:
                display_name = cam.parent_name if cam.parent_name else cam.name
                self._get_id(f"Model::{self._sanitize_name(display_name)}")
            for mesh in scene_data.meshes:
                if mesh.animation_type != AnimationType.VERTEX_ANIMATED:
                    display_name = mesh.parent_name if mesh.parent_name else mesh.name
                    self._get_id(f"Model::{self._sanitize_name(display_name)}")
            for transform in scene_data.transforms:
                if transform.keyframes:
                    self._get_id(f"Model::{self._sanitize_name(transform.name)}")

            # === OBJECTS ===
            lines.append("Objects:  {")

            # Create hierarchy groups first (as Null nodes)
            if hierarchy_groups:
                for group_name, parent_name in hierarchy_groups:
                    if group_name not in self._created_groups:
                        # Ensure parent exists
                        if parent_name and parent_name not in self._created_groups:
                            lines.extend(self._write_hierarchy_group(parent_name, None))
                        lines.extend(self._write_hierarchy_group(group_name, parent_name))
                        self.log(f"  Creating hierarchy group: {group_name}")

            # Export cameras with hierarchy
            for cam in scene_data.cameras:
                display_name = cam.parent_name if cam.parent_name else cam.name
                cam_name = self._sanitize_name(display_name)
                # Use centralized hierarchy or fall back to local method
                if scene_data.hierarchy:
                    parent = scene_data.hierarchy.get_node_parent_from_path(cam.full_path, display_name)
                else:
                    parent = self._get_node_parent(cam.full_path, None, display_name)
                self.log(f"  Processing camera: {cam_name}" + (f" (parent: {parent})" if parent else ""))
                lines.extend(self._write_camera(cam, cam_name, parent))

            # Export meshes (skip raw vertex-animated, but keep blend shapes) with hierarchy
            skipped_meshes = []
            for mesh in scene_data.meshes:
                display_name = mesh.parent_name if mesh.parent_name else mesh.name
                mesh_name = self._sanitize_name(display_name)

                if mesh.animation_type == AnimationType.VERTEX_ANIMATED:
                    skipped_meshes.append(mesh_name)
                    self.log(f"  Skipping vertex-animated mesh: {mesh_name}")
                    continue

                # Use centralized hierarchy or fall back to local method
                if scene_data.hierarchy:
                    parent = scene_data.hierarchy.get_node_parent_from_path(mesh.full_path, display_name)
                else:
                    parent = self._get_node_parent(mesh.full_path, None, display_name)

                if mesh.animation_type == AnimationType.BLEND_SHAPE:
                    self.log(f"  Processing mesh with blend shapes: {mesh_name}" + (f" (parent: {parent})" if parent else ""))
                else:
                    self.log(f"  Processing mesh: {mesh_name}" + (f" (parent: {parent})" if parent else ""))

                lines.extend(self._write_mesh(mesh, mesh_name, parent))

            # Export locators/transforms with hierarchy
            for transform in scene_data.transforms:
                xform_name = self._sanitize_name(transform.name)
                if not transform.keyframes:
                    continue
                # Use centralized hierarchy or fall back to local method
                if scene_data.hierarchy:
                    parent = scene_data.hierarchy.get_node_parent_from_path(transform.full_path, xform_name)
                else:
                    parent = self._get_node_parent(transform.full_path, None, xform_name)
                self.log(f"  Processing locator: {xform_name}" + (f" (parent: {parent})" if parent else ""))
                lines.extend(self._write_locator(transform, xform_name, parent))

            # Write animation stacks
            lines.extend(self._write_animation_stack())

            lines.append("}")
            lines.append("")

            # === CONNECTIONS ===
            lines.extend(self._write_connections())

            # === TAKES ===
            lines.extend(self._write_takes())

            # Write file
            with open(fbx_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            self.log(f"FBX file created: {fbx_file.name}")

            result = {
                'success': True,
                'fbx_file': str(fbx_file),
                'files': [str(fbx_file)],
                'message': f"FBX export complete: {fbx_file.name}"
            }

            if skipped_meshes:
                result['skipped_meshes'] = skipped_meshes
                result['message'] += f" (skipped {len(skipped_meshes)} vertex-animated meshes)"

            return result

        except Exception as e:
            error_msg = f"FBX export failed: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            return {
                'success': False,
                'message': error_msg,
                'files': []
            }

    # === COORDINATE CONVERSION ===

    def _convert_position(self, pos):
        """Convert position for FBX export (pass-through, Y-up preserved)"""
        # Handle nested list edge case: [[x, y, z]] -> [x, y, z]
        if isinstance(pos, (list, tuple)) and len(pos) > 0 and isinstance(pos[0], (list, tuple)):
            pos = pos[0]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _convert_rotation(self, rot):
        """Convert rotation for FBX export (pass-through, Y-up preserved)"""
        # Handle nested list edge case: [[rx, ry, rz]] -> [rx, ry, rz]
        if isinstance(rot, (list, tuple)) and len(rot) > 0 and isinstance(rot[0], (list, tuple)):
            rot = rot[0]
        return (float(rot[0]), float(rot[1]), float(rot[2]))

    def _compute_face_normals(self, positions, indices, counts):
        """Compute flat face normals for FBX mesh

        Computes one normal per face vertex (flat shading).
        Normals are computed using cross product of face edges.

        Args:
            positions: List of vertex positions [(x,y,z), ...]
            indices: Flat list of face vertex indices
            counts: List of vertex counts per face

        Returns:
            list: Normals in polygon-vertex order [(nx,ny,nz), ...]
        """
        import math

        normals = []
        idx_offset = 0

        for face_vert_count in counts:
            if face_vert_count < 3:
                # Degenerate face, use up vector
                for _ in range(face_vert_count):
                    normals.append((0.0, 0.0, 1.0))
                idx_offset += face_vert_count
                continue

            # Get first three vertices of the face
            v0_idx = indices[idx_offset]
            v1_idx = indices[idx_offset + 1]
            v2_idx = indices[idx_offset + 2]

            v0 = positions[v0_idx] if v0_idx < len(positions) else (0, 0, 0)
            v1 = positions[v1_idx] if v1_idx < len(positions) else (0, 0, 0)
            v2 = positions[v2_idx] if v2_idx < len(positions) else (0, 0, 0)

            # Compute two edges
            edge1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            edge2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])

            # Cross product for normal (edge2 x edge1 for correct winding)
            nx = edge2[1] * edge1[2] - edge2[2] * edge1[1]
            ny = edge2[2] * edge1[0] - edge2[0] * edge1[2]
            nz = edge2[0] * edge1[1] - edge2[1] * edge1[0]

            # Normalize
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 1e-10:
                nx, ny, nz = nx / length, ny / length, nz / length
            else:
                # Fallback for degenerate faces
                nx, ny, nz = 0.0, 0.0, 1.0

            # Repeat normal for each vertex in the face (flat shading)
            for _ in range(face_vert_count):
                normals.append((nx, ny, nz))

            idx_offset += face_vert_count

        return normals

    def _count_animation_curves(self, scene_data):
        """Pre-calculate the number of animation curve nodes and curves

        This is needed for the Definitions section which must declare
        object types before they are created.

        Returns:
            tuple: (num_anim_curve_nodes, num_anim_curves)
        """
        def is_animated(vals):
            return len(set(round(v, 4) for v in vals)) > 1

        total_curve_nodes = 0
        total_curves = 0

        # Helper to count for a list of keyframes
        def count_for_keyframes(keyframes):
            if not keyframes or len(keyframes) < 2:
                return 0, 0

            nodes = 0
            curves = 0

            # Extract position and rotation values
            positions = [self._convert_position(kf.position) for kf in keyframes]
            rotations = [self._convert_rotation(kf.rotation_maya) for kf in keyframes]

            tx = [p[0] for p in positions]
            ty = [p[1] for p in positions]
            tz = [p[2] for p in positions]
            rx = [r[0] for r in rotations]
            ry = [r[1] for r in rotations]
            rz = [r[2] for r in rotations]

            # Check translation
            trans_animated = [is_animated(tx), is_animated(ty), is_animated(tz)]
            if any(trans_animated):
                nodes += 1
                curves += sum(trans_animated)

            # Check rotation
            rot_animated = [is_animated(rx), is_animated(ry), is_animated(rz)]
            if any(rot_animated):
                nodes += 1
                curves += sum(rot_animated)

            return nodes, curves

        # Count for cameras
        for cam in scene_data.cameras:
            n, c = count_for_keyframes(cam.keyframes)
            total_curve_nodes += n
            total_curves += c

        # Count for meshes (only transform-only animation)
        for mesh in scene_data.meshes:
            if mesh.animation_type == AnimationType.TRANSFORM_ONLY:
                n, c = count_for_keyframes(mesh.keyframes)
                total_curve_nodes += n
                total_curves += c

        # Count for locators/transforms
        for transform in scene_data.transforms:
            if transform.keyframes:
                n, c = count_for_keyframes(transform.keyframes)
                total_curve_nodes += n
                total_curves += c

        # Count blend shape weight animation curves
        for mesh in scene_data.meshes:
            if mesh.animation_type == AnimationType.BLEND_SHAPE and mesh.blend_shapes:
                for channel in mesh.blend_shapes.channels:
                    if channel.weight_animation:
                        total_curve_nodes += 1
                        total_curves += 1

        return total_curve_nodes, total_curves

    # === FBX STRUCTURE WRITERS ===

    def _write_header(self):
        """Write FBX ASCII header"""
        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S.000")
        return [
            "; FBX 7.4.0 project file",
            "; Created by MultiConverter",
            "; ----------------------------------------------------",
            "",
            "FBXHeaderExtension:  {",
            "    FBXHeaderVersion: 1003",
            "    FBXVersion: 7400",
            "    CreationTimeStamp:  {",
            f'        Version: 1000',
            f'        Year: {datetime.now().year}',
            f'        Month: {datetime.now().month}',
            f'        Day: {datetime.now().day}',
            f'        Hour: {datetime.now().hour}',
            f'        Minute: {datetime.now().minute}',
            f'        Second: {datetime.now().second}',
            f'        Millisecond: 0',
            "    }",
            '    Creator: "MultiConverter v2.6.2"',
            "    SceneInfo: \"SceneInfo::GlobalInfo\", \"UserData\" {",
            '        Type: "UserData"',
            "        Version: 100",
            "        MetaData:  {",
            "            Version: 100",
            '            Title: ""',
            '            Subject: ""',
            '            Author: ""',
            '            Keywords: ""',
            '            Revision: ""',
            '            Comment: ""',
            "        }",
            "    }",
            "}",
            "",
        ]

    def _write_global_settings(self):
        """Write global settings with Y-up axis (Maya/Alembic native)"""
        return [
            "GlobalSettings:  {",
            "    Version: 1000",
            "    Properties70:  {",
            '        P: "UpAxis", "int", "Integer", "",1',
            '        P: "UpAxisSign", "int", "Integer", "",1',
            '        P: "FrontAxis", "int", "Integer", "",2',
            '        P: "FrontAxisSign", "int", "Integer", "",1',
            '        P: "CoordAxis", "int", "Integer", "",0',
            '        P: "CoordAxisSign", "int", "Integer", "",1',
            '        P: "OriginalUpAxis", "int", "Integer", "",1',
            '        P: "OriginalUpAxisSign", "int", "Integer", "",1',
            f'        P: "UnitScaleFactor", "double", "Number", "",1',
            f'        P: "OriginalUnitScaleFactor", "double", "Number", "",1',
            f'        P: "TimeSpanStart", "KTime", "Time", "",0',
            f'        P: "TimeSpanStop", "KTime", "Time", "",{int(self.frame_count * 46186158000 / self.fps)}',
            f'        P: "CustomFrameRate", "double", "Number", "",{self.fps}',
            "    }",
            "}",
            "",
        ]

    def _write_documents(self):
        """Write documents section"""
        return [
            "Documents:  {",
            "    Count: 1",
            "    Document: 1000000000, \"\", \"Scene\" {",
            "        Properties70:  {",
            '            P: "SourceObject", "object", "", ""',
            f'            P: "ActiveAnimStackName", "KString", "", "", "Take 001"',
            "        }",
            "        RootNode: 0",
            "    }",
            "}",
            "",
        ]

    def _write_references(self):
        """Write references section (empty for our exports)"""
        return [
            "References:  {",
            "}",
            "",
        ]

    def _write_definitions(self, num_cameras, num_meshes, num_locators, num_groups=0,
                           num_bs_deformers=0, num_bs_channels=0, num_shape_geoms=0,
                           num_anim_curve_nodes=0, num_anim_curves=0):
        """Write object type definitions

        Args:
            num_cameras: Number of camera nodes
            num_meshes: Number of mesh nodes
            num_locators: Number of locator/tracking marker nodes (use NodeAttribute Null)
            num_groups: Number of hierarchy group nodes (use NodeAttribute Null)
            num_bs_deformers: Number of BlendShape deformer nodes
            num_bs_channels: Number of BlendShapeChannel nodes
            num_shape_geoms: Number of Shape geometry nodes
            num_anim_curve_nodes: Number of AnimationCurveNode objects
            num_anim_curves: Number of AnimationCurve objects
        """
        # Count total objects: Models + Geometry + NodeAttributes + AnimStack + AnimLayer + Deformers + AnimCurves
        total_models = num_cameras + num_meshes + num_locators + num_groups
        total_geometry = num_meshes + num_shape_geoms  # Mesh geometries + Shape geometries
        total_node_attrs = num_cameras + num_groups + num_locators  # Camera attrs + group Nulls + locator Nulls
        total_deformers = num_bs_deformers + num_bs_channels  # BlendShape + BlendShapeChannel

        # Base count: GlobalSettings(1) + Model + Geometry + NodeAttribute + AnimStack(1) + AnimLayer(1)
        total_count = 4 + total_models + total_geometry + total_node_attrs + total_deformers
        if num_anim_curve_nodes > 0:
            total_count += num_anim_curve_nodes + num_anim_curves

        lines = [
            "Definitions:  {",
            "    Version: 100",
            f"    Count: {total_count}",
            "    ObjectType: \"GlobalSettings\" {",
            "        Count: 1",
            "    }",
            f'    ObjectType: "Model" {{',
            f"        Count: {total_models}",
            "        PropertyTemplate: \"FbxNode\" {",
            "            Properties70:  {",
            '                P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0',
            '                P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0',
            '                P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1',
            "            }",
            "        }",
            "    }",
            f'    ObjectType: "Geometry" {{',
            f"        Count: {total_geometry}",
            '        PropertyTemplate: "FbxMesh" {',
            "            Properties70:  {",
            '                P: "Color", "ColorRGB", "Color", "",0.8,0.8,0.8',
            '                P: "BBoxMin", "Vector3D", "Vector", "",0,0,0',
            '                P: "BBoxMax", "Vector3D", "Vector", "",0,0,0',
            '                P: "Primary Visibility", "bool", "", "",1',
            '                P: "Casts Shadows", "bool", "", "",1',
            '                P: "Receive Shadows", "bool", "", "",1',
            "            }",
            "        }",
            "    }",
            f'    ObjectType: "NodeAttribute" {{',
            f"        Count: {total_node_attrs}",
            '        PropertyTemplate: "FbxCamera" {',
            "            Properties70:  {",
            '                P: "FocalLength", "Number", "", "A",35',
            '                P: "NearPlane", "double", "Number", "",0.1',
            '                P: "FarPlane", "double", "Number", "",10000',
            "            }",
            "        }",
            "    }",
        ]

        lines.extend([
            '    ObjectType: "AnimationStack" {',
            "        Count: 1",
            "    }",
            '    ObjectType: "AnimationLayer" {',
            "        Count: 1",
            "    }",
        ])

        # Add AnimationCurveNode definition if we have animation
        if num_anim_curve_nodes > 0:
            lines.extend([
                f'    ObjectType: "AnimationCurveNode" {{',
                f"        Count: {num_anim_curve_nodes}",
                "    }",
            ])

        # Add AnimationCurve definition if we have animation
        if num_anim_curves > 0:
            lines.extend([
                f'    ObjectType: "AnimationCurve" {{',
                f"        Count: {num_anim_curves}",
                "    }",
            ])

        # Add Deformer definition if we have blend shapes
        if total_deformers > 0:
            lines.extend([
                f'    ObjectType: "Deformer" {{',
                f"        Count: {total_deformers}",
                "    }",
            ])

        lines.extend([
            "}",
            "",
        ])

        return lines

    def _write_camera(self, cam_data, cam_name, parent_name=None):
        """Write camera node and attributes

        Args:
            cam_data: CameraData from SceneData
            cam_name: Sanitized camera name
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        model_id = self._get_id(f"Model::{cam_name}")
        cam_id = self._get_id(f"NodeAttribute::{cam_name}")

        # Get initial transform (converted to Z-up)
        if cam_data.keyframes:
            kf = cam_data.keyframes[0]
            pos = self._convert_position(kf.position)
            rot = self._convert_rotation(kf.rotation_maya)
        else:
            pos = (0, 0, 0)
            rot = (0, 0, 0)

        focal_length = cam_data.properties.focal_length
        # Convert aperture from cm (Alembic/SceneData) to inches (FBX)
        # 1 cm = 0.393701 inches
        h_aperture_cm = cam_data.properties.h_aperture
        v_aperture_cm = cam_data.properties.v_aperture
        film_width_inches = h_aperture_cm * 0.393701
        film_height_inches = v_aperture_cm * 0.393701
        film_aspect = film_width_inches / film_height_inches if film_height_inches > 0 else 1.77778

        # Camera as NodeAttribute (Maya-compatible format)
        lines.extend([
            f'    NodeAttribute: {cam_id}, "NodeAttribute::{cam_name}", "Camera" {{',
            '        Properties70:  {',
            f'            P: "FilmWidth", "Number", "", "A",{film_width_inches:.6f}',
            f'            P: "FilmHeight", "Number", "", "A",{film_height_inches:.6f}',
            f'            P: "FilmAspectRatio", "double", "Number", "",{film_aspect:.10f}',
            '            P: "ApertureMode", "enum", "", "",3',
            '            P: "GateFit", "enum", "", "",2',
            f'            P: "FocalLength", "Number", "", "A",{focal_length}',
            '            P: "NearPlane", "double", "Number", "",0.1',
            '            P: "FarPlane", "double", "Number", "",10000',
            '        }',
            '        TypeFlags: "Camera"',
            '        GeometryVersion: 124',
            '        Position: 0,0,0',
            '        Up: 0,1,0',
            '        LookAt: 0,0,-1',
            '        ShowInfoOnMoving: 1',
            '        ShowAudio: 0',
            '        AudioColor: 0,1,0',
            '        CameraOrthoZoom: 1',
            '    }',
        ])

        # Camera model
        # PostRotation -90 on Y aligns FBX camera's default orientation with Maya convention
        lines.extend([
            f'    Model: {model_id}, "Model::{cam_name}", "Camera" {{',
            '        Version: 232',
            '        Properties70:  {',
            '            P: "PostRotation", "Vector3D", "Vector", "",0,-90,0',
            '            P: "RotationActive", "bool", "", "",1',
            '            P: "InheritType", "enum", "", "",1',
            '            P: "ScalingMax", "Vector3D", "Vector", "",0,0,0',
            '            P: "DefaultAttributeIndex", "int", "Integer", "",0',
            f'            P: "Lcl Translation", "Lcl Translation", "", "A",{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}',
            f'            P: "Lcl Rotation", "Lcl Rotation", "", "A",{rot[0]:.6f},{rot[1]:.6f},{rot[2]:.6f}',
            '            P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1',
            '        }',
            '        Shading: Y',
            '        Culling: "CullingOff"',
            '    }',
        ])

        # Connect model to parent or root FIRST (before camera connection)
        # Check if parent exists and is not self (parent_name != cam_name)
        if parent_name and parent_name != cam_name and (parent_name in self._created_groups or f"Model::{parent_name}" in self._object_ids):
            parent_id = self._get_id(f"Model::{parent_name}")
            self._connections.append((model_id, parent_id, None))
        else:
            self._connections.append((model_id, 0, None))

        # Connect NodeAttribute to model AFTER model-to-parent
        self._connections.append((cam_id, model_id, None))

        # Add animation curves
        self._add_animation_curves(cam_data.keyframes, cam_name, lines)

        return lines

    def _write_mesh(self, mesh_data, mesh_name, parent_name=None):
        """Write mesh geometry and model node

        Args:
            mesh_data: MeshData from SceneData
            mesh_name: Sanitized mesh name
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        model_id = self._get_id(f"Model::{mesh_name}")
        geom_id = self._get_id(f"Geometry::{mesh_name}")

        # Get initial transform (converted to Z-up)
        if mesh_data.keyframes:
            kf = mesh_data.keyframes[0]
            pos = self._convert_position(kf.position)
            rot = self._convert_rotation(kf.rotation_maya)
            scale = kf.scale
        else:
            pos = (0, 0, 0)
            rot = (0, 0, 0)
            scale = (1, 1, 1)

        # === GEOMETRY ===
        # SceneData always stores vertices in local (object) space
        positions = mesh_data.geometry.positions
        indices = mesh_data.geometry.indices
        counts = mesh_data.geometry.counts

        # Apply any coordinate system conversion (pass-through for Y-up)
        converted_positions = [self._convert_position(p) for p in positions]

        # Flatten positions for FBX format
        pos_array = []
        for p in converted_positions:
            pos_array.extend([p[0], p[1], p[2]])

        # Build polygon vertex indices (negative marks end of polygon in FBX)
        poly_indices = []
        idx_offset = 0
        for count in counts:
            for i in range(count - 1):
                poly_indices.append(indices[idx_offset + i])
            # Last index is negative (XOR with -1)
            poly_indices.append(-indices[idx_offset + count - 1] - 1)
            idx_offset += count

        # Compute face normals (using converted Z-up positions)
        normals = self._compute_face_normals(converted_positions, indices, counts)

        # Flatten normals for FBX format
        normals_array = []
        for n in normals:
            normals_array.extend([n[0], n[1], n[2]])

        lines.extend([
            f'    Geometry: {geom_id}, "Geometry::{mesh_name}", "Mesh" {{',
            f'        Vertices: *{len(pos_array)} {{',
            f'            a: {",".join(f"{v:.6f}" for v in pos_array)}',
            '        }',
            f'        PolygonVertexIndex: *{len(poly_indices)} {{',
            f'            a: {",".join(str(i) for i in poly_indices)}',
            '        }',
            '        GeometryVersion: 124',
            '        LayerElementNormal: 0 {',
            '            Version: 102',
            '            Name: ""',
            '            MappingInformationType: "ByPolygonVertex"',
            '            ReferenceInformationType: "Direct"',
            f'            Normals: *{len(normals_array)} {{',
            f'                a: {",".join(f"{v:.6f}" for v in normals_array)}',
            '            }',
            '        }',
            '        LayerElementUV: 0 {',
            '            Version: 101',
            '            Name: "UVMap"',
            '            MappingInformationType: "ByPolygonVertex"',
            '            ReferenceInformationType: "Direct"',
            f'            UV: *{len(poly_indices) * 2} {{',
            f'                a: {",".join(["0,0"] * len(poly_indices))}',
            '            }',
            '        }',
            '        Layer: 0 {',
            '            Version: 100',
            '            LayerElement:  {',
            '                Type: "LayerElementNormal"',
            '                TypedIndex: 0',
            '            }',
            '            LayerElement:  {',
            '                Type: "LayerElementUV"',
            '                TypedIndex: 0',
            '            }',
            '        }',
            '    }',
        ])

        # === MODEL ===
        lines.extend([
            f'    Model: {model_id}, "Model::{mesh_name}", "Mesh" {{',
            '        Version: 232',
            '        Properties70:  {',
            '            P: "RotationActive", "bool", "", "",1',
            '            P: "InheritType", "enum", "", "",1',
            '            P: "ScalingMax", "Vector3D", "Vector", "",0,0,0',
            '            P: "DefaultAttributeIndex", "int", "Integer", "",0',
            f'            P: "Lcl Translation", "Lcl Translation", "", "A",{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}',
            f'            P: "Lcl Rotation", "Lcl Rotation", "", "A",{rot[0]:.6f},{rot[1]:.6f},{rot[2]:.6f}',
            f'            P: "Lcl Scaling", "Lcl Scaling", "", "A",{scale[0]:.6f},{scale[1]:.6f},{scale[2]:.6f}',
            '        }',
            '        Shading: T',
            '        Culling: "CullingOff"',
            '    }',
        ])

        # Connect model to parent or root FIRST (before geometry connection)
        # Check if parent exists and is not self (parent_name != mesh_name)
        if parent_name and parent_name != mesh_name and (parent_name in self._created_groups or f"Model::{parent_name}" in self._object_ids):
            parent_id = self._get_id(f"Model::{parent_name}")
            self._connections.append((model_id, parent_id, None))
        else:
            self._connections.append((model_id, 0, None))

        # Connect geometry to model AFTER model-to-parent
        self._connections.append((geom_id, model_id, None))

        # Add animation curves if animated
        if mesh_data.animation_type == AnimationType.TRANSFORM_ONLY:
            self._add_animation_curves(mesh_data.keyframes, mesh_name, lines)

        # Add blend shapes if present
        if mesh_data.animation_type == AnimationType.BLEND_SHAPE and mesh_data.blend_shapes:
            lines.extend(self._write_blend_shapes(mesh_data.blend_shapes, mesh_name, geom_id))

        return lines

    def _write_blend_shapes(self, blend_shapes, mesh_name, geom_id):
        """Write blend shape deformers and shape geometries

        Args:
            blend_shapes: BlendShapeDeformer from scene_data
            mesh_name: Sanitized mesh name
            geom_id: Geometry ID to connect deformer to

        Returns:
            list: FBX lines for blend shape deformer
        """
        lines = []

        # Create BlendShape deformer
        deformer_id = self._get_id(f"Deformer::{blend_shapes.name}")
        lines.extend([
            f'    Deformer: {deformer_id}, "Deformer::{blend_shapes.name}", "BlendShape" {{',
            '        Version: 100',
            '    }',
        ])

        # Connect deformer to mesh geometry
        self._connections.append((deformer_id, geom_id, None))

        # Process each channel
        for channel in blend_shapes.channels:
            channel_id = self._get_id(f"SubDeformer::{channel.name}")

            # DeformPercent is 0-100 scale
            deform_percent = channel.default_weight * 100.0

            # Build FullWeights array (one entry per target)
            full_weights = [int(t.full_weight * 100) for t in channel.targets]

            lines.extend([
                f'    Deformer: {channel_id}, "SubDeformer::{channel.name}", "BlendShapeChannel" {{',
                '        Version: 100',
                f'        DeformPercent: {deform_percent:.1f}',
                f'        FullWeights: *{len(full_weights)} {{',
                f'            a: {",".join(str(w) for w in full_weights)}',
                '        }',
                '    }',
            ])

            # Connect channel to deformer
            self._connections.append((channel_id, deformer_id, None))

            # Write shape geometries for each target
            for target in channel.targets:
                shape_id = self._get_id(f"Geometry::{mesh_name}_{target.name}")

                # Convert deltas to Z-up
                converted_deltas = [self._convert_position(d) for d in target.deltas]

                # Flatten indices and vertices
                indices_str = ",".join(str(i) for i in target.vertex_indices)
                vertices_flat = []
                for d in converted_deltas:
                    vertices_flat.extend([d[0], d[1], d[2]])
                vertices_str = ",".join(f"{v:.6f}" for v in vertices_flat)

                lines.extend([
                    f'    Geometry: {shape_id}, "Geometry::{target.name}", "Shape" {{',
                    '        Version: 100',
                    f'        Indexes: *{len(target.vertex_indices)} {{',
                    f'            a: {indices_str}',
                    '        }',
                    f'        Vertices: *{len(vertices_flat)} {{',
                    f'            a: {vertices_str}',
                    '        }',
                    '    }',
                ])

                # Connect shape to channel
                self._connections.append((shape_id, channel_id, None))

            # Add weight animation if present
            if channel.weight_animation:
                self._add_blend_shape_weight_animation(channel, lines)

        return lines

    def _add_blend_shape_weight_animation(self, channel, lines):
        """Add animation curve for blend shape weight

        Args:
            channel: BlendShapeChannel with weight_animation
            lines: List to append FBX lines to
        """
        if not channel.weight_animation:
            return

        anim_layer_id = self._get_id("AnimationLayer::BaseLayer")
        channel_id = self._get_id(f"SubDeformer::{channel.name}")

        # Time conversion: frames to FBX time (46186158000 units per second)
        time_scale = 46186158000 / self.fps

        # Build keyframe data
        times = [int(kf.frame * time_scale) for kf in channel.weight_animation]
        # Convert weights from 0-1 to 0-100 for FBX
        values = [kf.weight * 100.0 for kf in channel.weight_animation]

        # Create AnimCurveNode for DeformPercent
        curve_node_id = self._get_id(f"AnimCurveNode::{channel.name}_DeformPercent")
        lines.extend([
            f'    AnimationCurveNode: {curve_node_id}, "AnimCurveNode::DeformPercent", "" {{',
            '        Properties70:  {',
            f'            P: "d|DeformPercent", "Number", "", "A",{values[0]:.6f}',
            '        }',
            '    }',
        ])

        # Connect curve node to animation layer and channel
        self._connections.append((curve_node_id, anim_layer_id, None))
        self._connections.append((curve_node_id, channel_id, "DeformPercent"))

        # Create AnimCurve
        curve_id = self._get_id(f"AnimCurve::{channel.name}_DeformPercent")
        key_count = len(times)
        time_str = ",".join(str(t) for t in times)
        val_str = ",".join(f"{v:.6f}" for v in values)

        # AttrFlags: all linear interpolation
        attr_flags = ",".join(["24836"] * key_count)
        # AttrData: 4 zeros per key (tangent data)
        attr_data = ",".join(["0,0,0,0"] * key_count)

        lines.extend([
            f'    AnimationCurve: {curve_id}, "AnimCurve::", "" {{',
            '        Default: 0',
            '        KeyVer: 4009',
            f'        KeyTime: *{key_count} {{',
            f'            a: {time_str}',
            '        }',
            f'        KeyValueFloat: *{key_count} {{',
            f'            a: {val_str}',
            '        }',
            f'        KeyAttrFlags: *{key_count} {{',
            f'            a: {attr_flags}',
            '        }',
            f'        KeyAttrDataFloat: *{key_count * 4} {{',
            f'            a: {attr_data}',
            '        }',
            f'        KeyAttrRefCount: *{key_count} {{',
            f'            a: {",".join(["1"] * key_count)}',
            '        }',
            '    }',
        ])

        # Connect curve to curve node
        self._connections.append((curve_id, curve_node_id, "d|DeformPercent"))

    def _write_locator(self, transform_data, loc_name, parent_name=None):
        """Write locator/tracking point node using FBX NodeAttribute Null type

        FBX NodeAttribute Null is used for locators/tracking points.
        This creates a Null transform in Maya that appears in the Outliner.

        Args:
            transform_data: TransformData from SceneData
            loc_name: Sanitized locator name
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        model_id = self._get_id(f"Model::{loc_name}")
        nodeattr_id = self._get_id(f"NodeAttribute::{loc_name}")

        # Get initial transform
        if transform_data.keyframes:
            kf = transform_data.keyframes[0]
            pos = self._convert_position(kf.position)
            rot = self._convert_rotation(kf.rotation_maya)
            scale = kf.scale
        else:
            pos = (0, 0, 0)
            rot = (0, 0, 0)
            scale = (1, 1, 1)

        # NodeAttribute Null object (for locators/tracking points)
        lines.extend([
            f'    NodeAttribute: {nodeattr_id}, "NodeAttribute::{loc_name}", "Null" {{',
            '        TypeFlags: "Null"',
            '    }',
        ])

        # Model Null (transform node)
        lines.extend([
            f'    Model: {model_id}, "Model::{loc_name}", "Null" {{',
            '        Version: 232',
            '        Properties70:  {',
            f'            P: "Lcl Translation", "Lcl Translation", "", "A",{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}',
            f'            P: "Lcl Rotation", "Lcl Rotation", "", "A",{rot[0]:.6f},{rot[1]:.6f},{rot[2]:.6f}',
            f'            P: "Lcl Scaling", "Lcl Scaling", "", "A",{scale[0]:.6f},{scale[1]:.6f},{scale[2]:.6f}',
            '        }',
            '        Shading: Y',
            '        Culling: "CullingOff"',
            '    }',
        ])

        # Connect NodeAttribute to Model
        self._connections.append((nodeattr_id, model_id, None))

        # Connect model to parent or root
        # Check if parent exists and is not self (parent_name != loc_name)
        if parent_name and parent_name != loc_name and (parent_name in self._created_groups or f"Model::{parent_name}" in self._object_ids):
            parent_id = self._get_id(f"Model::{parent_name}")
            self._connections.append((model_id, parent_id, None))
        else:
            self._connections.append((model_id, 0, None))

        # Add animation curves
        self._add_animation_curves(transform_data.keyframes, loc_name, lines)

        return lines

    def _add_animation_curves(self, keyframes, obj_name, lines):
        """Add animation curve nodes for an object"""
        if not keyframes or len(keyframes) < 2:
            return

        model_id = self._get_id(f"Model::{obj_name}")
        anim_layer_id = self._get_id("AnimationLayer::BaseLayer")

        # Time conversion: frames to FBX time (46186158000 units per second)
        time_scale = 46186158000 / self.fps

        def is_animated(vals):
            return len(set(round(v, 4) for v in vals)) > 1

        # Extract and convert values
        times = [int(kf.frame * time_scale) for kf in keyframes]

        # Convert positions to Z-up
        positions = [self._convert_position(kf.position) for kf in keyframes]
        tx = [p[0] for p in positions]
        ty = [p[1] for p in positions]
        tz = [p[2] for p in positions]

        # Convert rotations to Z-up
        rotations = [self._convert_rotation(kf.rotation_maya) for kf in keyframes]
        rx = [r[0] for r in rotations]
        ry = [r[1] for r in rotations]
        rz = [r[2] for r in rotations]

        channels = [
            ('T', 'Lcl Translation', [
                ('X', tx), ('Y', ty), ('Z', tz)
            ]),
            ('R', 'Lcl Rotation', [
                ('X', rx), ('Y', ry), ('Z', rz)
            ]),
        ]

        for prefix, prop_name, axes in channels:
            # Check if any axis is animated
            if not any(is_animated(vals) for _, vals in axes):
                continue

            # AnimCurveNode
            curve_node_id = self._get_id(f"AnimCurveNode::{obj_name}_{prefix}")

            default_vals = [axes[0][1][0], axes[1][1][0], axes[2][1][0]]

            lines.extend([
                f'    AnimationCurveNode: {curve_node_id}, "AnimCurveNode::{prefix}", "" {{',
                '        Properties70:  {',
                f'            P: "d|X", "Number", "", "A",{default_vals[0]:.6f}',
                f'            P: "d|Y", "Number", "", "A",{default_vals[1]:.6f}',
                f'            P: "d|Z", "Number", "", "A",{default_vals[2]:.6f}',
                '        }',
                '    }',
            ])

            # Connect curve node to layer and model
            self._connections.append((curve_node_id, anim_layer_id, None))
            self._connections.append((curve_node_id, model_id, prop_name))

            # AnimCurves for each axis
            for axis_name, vals in axes:
                if not is_animated(vals):
                    continue

                curve_id = self._get_id(f"AnimCurve::{obj_name}_{prefix}_{axis_name}")

                # Build keyframe data
                key_count = len(times)
                time_str = ",".join(str(t) for t in times)
                val_str = ",".join(f"{v:.6f}" for v in vals)

                # AttrFlags: all linear interpolation
                attr_flags = ",".join(["24836"] * key_count)
                # AttrData: 4 zeros per key (tangent data)
                attr_data = ",".join(["0,0,0,0"] * key_count)

                lines.extend([
                    f'    AnimationCurve: {curve_id}, "AnimCurve::", "" {{',
                    '        Default: 0',
                    f'        KeyVer: 4009',
                    f'        KeyTime: *{key_count} {{',
                    f'            a: {time_str}',
                    '        }',
                    f'        KeyValueFloat: *{key_count} {{',
                    f'            a: {val_str}',
                    '        }',
                    f'        KeyAttrFlags: *{key_count} {{',
                    f'            a: {attr_flags}',
                    '        }',
                    f'        KeyAttrDataFloat: *{key_count * 4} {{',
                    f'            a: {attr_data}',
                    '        }',
                    f'        KeyAttrRefCount: *{key_count} {{',
                    f'            a: {",".join(["1"] * key_count)}',
                    '        }',
                    '    }',
                ])

                # Connect curve to curve node
                self._connections.append((curve_id, curve_node_id, f"d|{axis_name}"))

    def _write_animation_stack(self):
        """Write animation stack and layer"""
        lines = []

        stack_id = self._get_id("AnimationStack::Take001")
        layer_id = self._get_id("AnimationLayer::BaseLayer")

        # Time span
        time_scale = 46186158000 / self.fps
        end_time = int(self.frame_count * time_scale)

        lines.extend([
            f'    AnimationStack: {stack_id}, "AnimStack::Take 001", "" {{',
            '        Properties70:  {',
            f'            P: "LocalStop", "KTime", "Time", "",{end_time}',
            f'            P: "ReferenceStop", "KTime", "Time", "",{end_time}',
            '        }',
            '    }',
            f'    AnimationLayer: {layer_id}, "AnimLayer::BaseLayer", "" {{',
            '    }',
        ])

        # Connect layer to stack
        self._connections.append((layer_id, stack_id, None))

        return lines

    def _write_connections(self):
        """Write all object connections"""
        lines = [
            "Connections:  {",
        ]

        for child_id, parent_id, prop in self._connections:
            if prop:
                # Property connection
                lines.append(f'    C: "OP",{child_id},{parent_id}, "{prop}"')
            else:
                # Object-Object connection
                lines.append(f'    C: "OO",{child_id},{parent_id}')

        lines.append("}")
        lines.append("")

        return lines

    def _write_takes(self):
        """Write takes section"""
        time_scale = 46186158000 / self.fps
        end_time = int(self.frame_count * time_scale)

        return [
            "Takes:  {",
            '    Current: "Take 001"',
            '    Take: "Take 001" {',
            f'        FileName: "Take_001.tak"',
            f'        LocalTime: 0,{end_time}',
            f'        ReferenceTime: 0,{end_time}',
            '    }',
            "}",
            "",
        ]

    # === HIERARCHY UTILITIES ===

    def _build_hierarchy_map(self, scene_data: SceneData):
        """Build hierarchy map from full_path data

        Returns:
            dict: Mapping of sanitized node_name -> sanitized parent_name
        """
        hierarchy = {}

        # Collect all full paths
        all_items = list(scene_data.cameras) + list(scene_data.meshes) + list(scene_data.transforms)

        for item in all_items:
            parts = [p for p in item.full_path.split('/') if p]
            if len(parts) < 2:
                continue

            # Build relationships for all intermediate nodes
            for i in range(1, len(parts)):
                child = self._sanitize_name(parts[i])
                parent = self._sanitize_name(parts[i - 1])
                if child != parent:
                    hierarchy[child] = parent

        return hierarchy

    def _get_hierarchy_groups(self, scene_data: SceneData):
        """Get list of hierarchy groups that need to be created as Null nodes

        Returns:
            list: List of (group_name, parent_name) tuples in creation order
        """
        # Build sets of known nodes (cameras, meshes, transforms we'll create)
        known_nodes = set()

        for cam in scene_data.cameras:
            display_name = cam.parent_name if cam.parent_name else cam.name
            known_nodes.add(self._sanitize_name(display_name))

        for mesh in scene_data.meshes:
            display_name = mesh.parent_name if mesh.parent_name else mesh.name
            known_nodes.add(self._sanitize_name(display_name))

        for xform in scene_data.transforms:
            known_nodes.add(self._sanitize_name(xform.name))

        # Find all hierarchy groups from paths
        hierarchy_groups = {}  # group_name -> parent_name
        group_depths = {}  # group_name -> depth

        for item in list(scene_data.cameras) + list(scene_data.meshes) + list(scene_data.transforms):
            parts = [p for p in item.full_path.split('/') if p]

            for i, part in enumerate(parts[:-1]):
                sanitized = self._sanitize_name(part)

                if sanitized not in known_nodes:
                    parent = self._sanitize_name(parts[i - 1]) if i > 0 else None
                    hierarchy_groups[sanitized] = parent
                    group_depths[sanitized] = i

        # Sort by depth (parents first)
        sorted_groups = sorted(hierarchy_groups.items(), key=lambda x: group_depths.get(x[0], 0))
        return sorted_groups

    def _get_node_parent(self, full_path, hierarchy_map, display_name=None):
        """Get the parent node name for an object from its full_path

        Detects shape nodes by comparing the last path component to the
        display_name. If they differ, the object is a shape node.
        """
        parts = [p for p in full_path.split('/') if p]
        if len(parts) < 2:
            return None

        obj_name = parts[-1]

        # If display_name differs from last path component, it's a shape node
        is_shape = False
        if display_name and len(parts) >= 3:
            is_shape = self._sanitize_name(obj_name) != self._sanitize_name(display_name)

        if is_shape:
            return self._sanitize_name(parts[-3])
        elif len(parts) >= 2:
            return self._sanitize_name(parts[-2])

        return None

    def _write_hierarchy_group(self, group_name, parent_name=None):
        """Write a hierarchy group as a Null node

        Returns:
            list: FBX lines for this group
        """
        lines = []

        model_id = self._get_id(f"Model::{group_name}")
        attr_id = self._get_id(f"NodeAttribute::{group_name}")

        # Null attribute
        lines.extend([
            f'    NodeAttribute: {attr_id}, "NodeAttribute::{group_name}", "Null" {{',
            '        TypeFlags: "Null"',
            '    }',
        ])

        # Null model
        lines.extend([
            f'    Model: {model_id}, "Model::{group_name}", "Null" {{',
            '        Version: 232',
            '        Properties70:  {',
            '            P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0',
            '            P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0',
            '            P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1',
            '        }',
            '        Shading: Y',
            '        Culling: "CullingOff"',
            '    }',
        ])

        # Connect attribute to model
        self._connections.append((attr_id, model_id, None))

        # Connect model to parent or root
        if parent_name and parent_name in self._created_groups:
            parent_id = self._get_id(f"Model::{parent_name}")
            self._connections.append((model_id, parent_id, None))
        else:
            self._connections.append((model_id, 0, None))

        self._created_groups.add(group_name)

        return lines

    # === UTILITIES ===

    def _sanitize_name(self, name):
        """Sanitize name for FBX"""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"obj_{sanitized}"
        return sanitized or "unnamed"
