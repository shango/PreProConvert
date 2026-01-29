"""
Maya ASCII (.ma) exporter - produces native Maya geometry

v2.5.0 - Uses SceneData for format-agnostic export.

Export Strategy:
- Cameras: Full camera nodes with animCurve animation
- Meshes: Native Maya mesh geometry with proper shading
- Animation: animCurve nodes for transform animation
- Vertex-animated meshes: Reference original source via AlembicNode or USD Stage

Supports both Alembic and USD input files via SceneData.
"""

import re
from pathlib import Path
from datetime import datetime
from exporters.base_exporter import BaseExporter
from core.scene_data import SceneData, AnimationType


class MayaMAExporter(BaseExporter):
    """Maya ASCII (.ma) file exporter - produces native Maya files"""

    def __init__(self, progress_callback=None):
        super().__init__(progress_callback)
        self.maya_version = "2020"
        self.shot_name = ""
        self.mesh_shapes = []  # Track mesh shapes for shading connections
        self.created_nodes = set()  # Track nodes we've created (for hierarchy)

    def get_format_name(self):
        return "Maya MA"

    def get_file_extension(self):
        return "ma"

    def export(self, scene_data: SceneData, output_path, shot_name):
        """Main export method using SceneData

        Args:
            scene_data: SceneData instance with all pre-extracted animation
            output_path: Output directory
            shot_name: Shot/scene name
        """
        try:
            self.shot_name = shot_name
            self.mesh_shapes = []
            self.created_nodes = set()
            self.log(f"Exporting Maya MA format...")

            output_dir = Path(output_path)
            self.validate_output_path(output_dir)
            ma_file = output_dir / f"{shot_name}.ma"

            # Extract info from SceneData
            fps = scene_data.metadata.fps
            frame_count = scene_data.metadata.frame_count
            source_file_path = scene_data.metadata.source_file_path
            source_format = scene_data.metadata.source_format_name
            source_file_type = 'alembic' if source_format == 'Alembic' else 'usd'

            lines = []
            has_vertex_anim = len(scene_data.animation_categories.vertex_animated) > 0

            # === FILE HEADER ===
            lines.extend(self._generate_header())
            lines.extend(self._generate_requirements(has_vertex_anim, source_file_type))
            lines.extend(self._generate_units(fps, frame_count))
            lines.extend(self._generate_file_info(source_file_path, source_file_type))

            # === DEFAULT MAYA NODES ===
            lines.extend(self._generate_default_nodes())

            # === SCENE CONTENT ===
            lines.extend(self._generate_scene_nodes(scene_data, source_file_path, source_file_type))

            # === SHADING CONNECTIONS ===
            lines.extend(self._generate_shading_connections())

            # === DEFAULT CONNECTIONS ===
            lines.extend(self._generate_default_connections())

            # Write file
            with open(ma_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            self.log(f"✓ Maya MA file created: {ma_file.name}")

            result = {
                'success': True,
                'ma_file': str(ma_file),
                'files': [str(ma_file)],
                'message': f"Maya MA export complete: {ma_file.name}"
            }

            if has_vertex_anim:
                result['message'] += f"\n⚠ Vertex-animated meshes reference original source file"

            return result

        except Exception as e:
            error_msg = f"Maya MA export failed: {str(e)}"
            self.log(f"✗ {error_msg}")
            import traceback
            self.log(traceback.format_exc())
            return {
                'success': False,
                'message': error_msg,
                'files': []
            }

    # === HEADER GENERATION ===

    def _generate_header(self):
        """Generate Maya .ma file header"""
        timestamp = datetime.now().strftime("%a, %b %d, %Y %I:%M:%S %p")
        return [
            f"//Maya ASCII {self.maya_version} scene",
            f"//Name: {self.shot_name}.ma",
            f"//Last modified: {timestamp}",
            "//Codeset: UTF-8",
            ""
        ]

    def _generate_requirements(self, has_vertex_anim=False, source_file_type='alembic'):
        """Generate requirements section"""
        lines = [f'requires maya "{self.maya_version}";']
        if has_vertex_anim:
            if source_file_type == 'alembic':
                lines.append('requires -nodeType "AlembicNode" "AbcImport" "1.0";')
            else:
                # USD requires mayaUsdPlugin for USD Stage nodes
                lines.append('requires -nodeType "mayaUsdProxyShape" "mayaUsdPlugin" "0.1.0";')
        lines.append('requires "stereoCamera" "10.0";')
        lines.append("")
        return lines

    def _generate_units(self, fps, frame_count):
        """Generate units and playback range"""
        return [
            'currentUnit -l centimeter -a degree -t film;',
            'fileInfo "application" "maya";',
            f'fileInfo "product" "Maya {self.maya_version}";',
            f'fileInfo "version" "{self.maya_version}";',
            f'playbackOptions -min 1 -max {frame_count} -ast 1 -aet {frame_count};',
            ""
        ]

    def _generate_file_info(self, source_filename=None, source_file_type='alembic'):
        """Generate file metadata"""
        lines = []
        if source_filename:
            if source_file_type == 'alembic':
                lines.append(f'fileInfo "sourceAlembic" "{self._mel_escape_string(source_filename)}";')
            else:
                lines.append(f'fileInfo "sourceUSD" "{self._mel_escape_string(source_filename)}";')
        lines.append("")
        return lines

    # === DEFAULT MAYA NODES ===

    def _generate_default_nodes(self):
        """Generate default Maya scene nodes"""
        return [
            '// Default Maya nodes',
            'createNode transform -s -n "persp";',
            '    setAttr ".t" -type "double3" 28 21 28;',
            '    setAttr ".r" -type "double3" -27.9 45 0;',
            'createNode camera -s -n "perspShape" -p "persp";',
            '    setAttr -k off ".v";',
            '    setAttr ".fl" 35;',
            '    setAttr ".coi" 44.8;',
            '    setAttr ".imn" -type "string" "persp";',
            '    setAttr ".den" -type "string" "persp_depth";',
            '    setAttr ".man" -type "string" "persp_mask";',
            '    setAttr ".hc" -type "string" "viewSet -p %camera";',
            'createNode transform -s -n "top";',
            '    setAttr ".t" -type "double3" 0 1000.1 0;',
            '    setAttr ".r" -type "double3" -90 0 0;',
            'createNode camera -s -n "topShape" -p "top";',
            '    setAttr -k off ".v";',
            '    setAttr ".rnd" no;',
            '    setAttr ".coi" 1000.1;',
            '    setAttr ".ow" 30;',
            '    setAttr ".imn" -type "string" "top";',
            '    setAttr ".den" -type "string" "top_depth";',
            '    setAttr ".man" -type "string" "top_mask";',
            '    setAttr ".hc" -type "string" "viewSet -t %camera";',
            '    setAttr ".o" yes;',
            'createNode transform -s -n "front";',
            '    setAttr ".t" -type "double3" 0 0 1000.1;',
            'createNode camera -s -n "frontShape" -p "front";',
            '    setAttr -k off ".v";',
            '    setAttr ".rnd" no;',
            '    setAttr ".coi" 1000.1;',
            '    setAttr ".ow" 30;',
            '    setAttr ".imn" -type "string" "front";',
            '    setAttr ".den" -type "string" "front_depth";',
            '    setAttr ".man" -type "string" "front_mask";',
            '    setAttr ".hc" -type "string" "viewSet -f %camera";',
            '    setAttr ".o" yes;',
            'createNode transform -s -n "side";',
            '    setAttr ".t" -type "double3" 1000.1 0 0;',
            '    setAttr ".r" -type "double3" 0 90 0;',
            'createNode camera -s -n "sideShape" -p "side";',
            '    setAttr -k off ".v";',
            '    setAttr ".rnd" no;',
            '    setAttr ".coi" 1000.1;',
            '    setAttr ".ow" 30;',
            '    setAttr ".imn" -type "string" "side";',
            '    setAttr ".den" -type "string" "side_depth";',
            '    setAttr ".man" -type "string" "side_mask";',
            '    setAttr ".hc" -type "string" "viewSet -s %camera";',
            '    setAttr ".o" yes;',
            '',
            '// Shading nodes',
            'createNode lightLinker -s -n "lightLinker1";',
            'createNode shapeEditorManager -n "shapeEditorManager";',
            'createNode poseInterpolatorManager -n "poseInterpolatorManager";',
            'createNode displayLayerManager -n "layerManager";',
            'createNode displayLayer -n "defaultLayer";',
            'createNode renderLayerManager -n "renderLayerManager";',
            'createNode renderLayer -n "defaultRenderLayer";',
            '    setAttr ".g" yes;',
            '',
            '// Shading groups',
            'createNode shadingEngine -n "initialShadingGroup" -s;',
            '    setAttr ".ihi" 0;',
            '    setAttr ".ro" yes;',
            'createNode materialInfo -n "initialMaterialInfo";',
            'createNode lambert -n "lambert1" -s;',
            '',
        ]

    # === SCENE GENERATION ===

    def _generate_scene_nodes(self, scene_data: SceneData, source_file_path, source_file_type):
        """Generate all scene content nodes from SceneData

        Reconstructs the original scene hierarchy by:
        1. Using centralized hierarchy from SceneData (or building locally as fallback)
        2. Creating intermediate group nodes first
        3. Creating cameras, meshes, and locators with proper parenting
        """
        lines = ['// Scene content', '']

        # Use centralized hierarchy from SceneData if available (new architecture)
        # Fall back to local methods for backward compatibility
        if scene_data.hierarchy:
            hierarchy_groups = scene_data.hierarchy.get_intermediate_groups()
        else:
            hierarchy_groups = self._get_hierarchy_groups(scene_data)

        # Create intermediate hierarchy groups first (parents before children)
        if hierarchy_groups:
            lines.append('// Hierarchy groups')
            for group_name, parent_name in hierarchy_groups:
                if group_name not in self.created_nodes:
                    # Ensure parent exists if specified and not yet created
                    if parent_name and parent_name not in self.created_nodes:
                        lines.append(f'createNode transform -n "{parent_name}";')
                        self.created_nodes.add(parent_name)

                    if parent_name and parent_name in self.created_nodes:
                        lines.append(f'createNode transform -n "{group_name}" -p "{parent_name}";')
                    else:
                        lines.append(f'createNode transform -n "{group_name}";')
                    self.created_nodes.add(group_name)
                    self.log(f"  Creating hierarchy group: {group_name}")
            lines.append('')

        # Process cameras
        for cam in scene_data.cameras:
            # Use parent_name for Alembic (cameraShape -> camera), else use name
            display_name = cam.parent_name if cam.parent_name else cam.name
            cam_name = self._sanitize_name(display_name)

            # Use centralized hierarchy or fall back to local method
            if scene_data.hierarchy:
                parent = scene_data.hierarchy.get_node_parent_from_path(cam.full_path, display_name)
            else:
                parent = self._get_node_parent(cam.full_path, None, display_name)

            self.log(f"  Processing camera: {cam_name}" + (f" (parent: {parent})" if parent else ""))
            lines.extend(self._export_camera(cam, cam_name, parent))
            self.created_nodes.add(cam_name)
            lines.append('')

        # Process meshes
        for mesh in scene_data.meshes:
            # Use parent_name for Alembic (CubeShape -> Cube), else use name
            display_name = mesh.parent_name if mesh.parent_name else mesh.name
            mesh_name = self._sanitize_name(display_name)

            # Use centralized hierarchy or fall back to local method
            if scene_data.hierarchy:
                parent = scene_data.hierarchy.get_node_parent_from_path(mesh.full_path, display_name)
            else:
                parent = self._get_node_parent(mesh.full_path, None, display_name)

            # Log geometry info for debugging
            geom = mesh.geometry
            verts = len(geom.positions) if geom and geom.positions else 0
            faces = len(geom.counts) if geom and geom.counts else 0
            if verts == 0:
                self.log(f"  WARNING: Mesh {mesh_name} has 0 vertices - will be invisible")

            if mesh.animation_type == AnimationType.VERTEX_ANIMATED:
                self.log(f"  Processing vertex-animated mesh: {mesh_name} ({verts} verts, {faces} faces)" + (f" (parent: {parent})" if parent else ""))
                lines.extend(self._export_vertex_animated_mesh(
                    mesh, mesh_name, source_file_path, source_file_type, parent
                ))
            else:
                self.log(f"  Processing mesh: {mesh_name} ({verts} verts, {faces} faces)" + (f" (parent: {parent})" if parent else ""))
                is_animated = mesh.animation_type == AnimationType.TRANSFORM_ONLY
                lines.extend(self._export_static_mesh(mesh, mesh_name, is_animated, parent))
            self.created_nodes.add(mesh_name)
            lines.append('')

        # Process transforms (locators/trackers)
        for transform in scene_data.transforms:
            xform_name = self._sanitize_name(transform.name)

            # Skip if no keyframes
            if not transform.keyframes:
                continue

            # Use centralized hierarchy or fall back to local method
            if scene_data.hierarchy:
                parent = scene_data.hierarchy.get_node_parent_from_path(transform.full_path, xform_name)
            else:
                parent = self._get_node_parent(transform.full_path, None, xform_name)

            self.log(f"  Processing locator: {xform_name}" + (f" (parent: {parent})" if parent else ""))
            lines.extend(self._export_locator(transform, xform_name, parent))
            self.created_nodes.add(xform_name)
            lines.append('')

        return lines

    def _export_camera(self, cam_data, cam_name, parent_name=None):
        """Export camera with animation from CameraData

        Args:
            cam_data: CameraData from SceneData
            cam_name: Sanitized camera name
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        # Create transform (with parent if specified)
        if parent_name and parent_name in self.created_nodes:
            lines.append(f'createNode transform -n "{cam_name}" -p "{parent_name}";')
        else:
            lines.append(f'createNode transform -n "{cam_name}";')

        # Get camera properties
        focal_length = cam_data.properties.focal_length
        # Alembic/SceneData stores aperture in cm, Maya expects inches (1 inch = 2.54 cm)
        h_aperture = cam_data.properties.h_aperture / 2.54
        v_aperture = cam_data.properties.v_aperture / 2.54

        # Create camera shape
        shape_name = f"{cam_name}Shape"
        lines.extend([
            f'createNode camera -n "{shape_name}" -p "{cam_name}";',
            f'    setAttr -k off ".v";',
            f'    setAttr ".fl" {focal_length};',
            f'    setAttr ".coi" 5;',
            f'    setAttr ".imn" -type "string" "{cam_name}";',
            f'    setAttr ".den" -type "string" "{cam_name}_depth";',
            f'    setAttr ".man" -type "string" "{cam_name}_mask";',
            f'    setAttr ".hfa" {h_aperture};',
            f'    setAttr ".vfa" {v_aperture};',
        ])

        # Add animation from keyframes
        lines.extend(self._animate_transform_from_keyframes(cam_data.keyframes, cam_name))

        return lines

    def _export_static_mesh(self, mesh_data, mesh_name, is_animated, parent_name=None):
        """Export mesh with native Maya geometry from MeshData

        Args:
            mesh_data: MeshData from SceneData
            mesh_name: Sanitized mesh name
            is_animated: Whether the mesh has transform animation
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        # Create transform (with parent if specified)
        if parent_name and parent_name in self.created_nodes:
            lines.append(f'createNode transform -n "{mesh_name}" -p "{parent_name}";')
        else:
            lines.append(f'createNode transform -n "{mesh_name}";')

        # Set initial transform values from first keyframe (using Maya-compatible rotation)
        if mesh_data.keyframes:
            kf = mesh_data.keyframes[0]
            pos = kf.position
            rot = kf.rotation_maya  # Use Maya-compatible rotation
            scale = kf.scale
            lines.append(f'    setAttr ".t" -type "double3" {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f};')
            lines.append(f'    setAttr ".r" -type "double3" {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f};')
            lines.append(f'    setAttr ".s" -type "double3" {scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f};')

        # Get mesh geometry from SceneData
        positions = mesh_data.geometry.positions
        indices = mesh_data.geometry.indices
        counts = mesh_data.geometry.counts

        # Create mesh shape
        shape_name = f"{mesh_name}Shape"
        self.mesh_shapes.append(shape_name)

        lines.append(f'createNode mesh -n "{shape_name}" -p "{mesh_name}";')
        lines.append(f'    setAttr -k off ".v";')
        lines.append(f'    setAttr ".vir" yes;')
        lines.append(f'    setAttr ".vif" yes;')

        num_verts = len(positions)
        num_faces = len(counts)

        # Build edges from face data
        edges = []
        edge_map = {}
        idx_offset = 0

        for count in counts:
            face_verts = [indices[idx_offset + i] for i in range(count)]
            for i in range(count):
                v1, v2 = face_verts[i], face_verts[(i + 1) % count]
                edge_key = (min(v1, v2), max(v1, v2))
                if edge_key not in edge_map:
                    edge_map[edge_key] = len(edges)
                    edges.append((v1, v2))
            idx_offset += count

        num_edges = len(edges)

        # Build the mesh data in official Maya format using setAttr -type mesh
        mesh_data_parts = []

        # Vertices: "v" count x y z x y z ...
        mesh_data_parts.append(f'"v" {num_verts}')
        for pos in positions:
            mesh_data_parts.append(f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}')

        # Vertex normals: "vn" 0 (required, set to 0)
        mesh_data_parts.append('"vn" 0')

        # Edges: "e" count v1 v2 "smooth" v1 v2 "smooth" ...
        mesh_data_parts.append(f'"e" {num_edges}')
        for v1, v2 in edges:
            mesh_data_parts.append(f'{v1} {v2} "smooth"')

        # Faces: "face" "l" edgeCount edge1 edge2 ... "face" "l" ...
        idx_offset = 0
        for face_idx, count in enumerate(counts):
            face_verts = [indices[idx_offset + i] for i in range(count)]
            # Reverse winding order - Alembic uses opposite convention from Maya
            face_verts.reverse()

            # Get edge indices for this face
            face_edges = []
            for i in range(count):
                v1, v2 = face_verts[i], face_verts[(i + 1) % count]
                edge_key = (min(v1, v2), max(v1, v2))
                edge_idx = edge_map[edge_key]
                # Check edge direction - negative means reversed
                if edges[edge_idx][0] == v1:
                    face_edges.append(edge_idx)
                else:
                    face_edges.append(-edge_idx - 1)

            edge_str = ' '.join(str(e) for e in face_edges)
            mesh_data_parts.append(f'"face" "l" {count} {edge_str}')
            idx_offset += count

        # Write mesh data as single setAttr command
        lines.append(f'    setAttr ".o" -type "mesh"')
        lines.append(f'        {" ".join(mesh_data_parts)};')

        # Animate transform if needed
        if is_animated:
            lines.extend(self._animate_transform_from_keyframes(mesh_data.keyframes, mesh_name))

        return lines

    def _export_vertex_animated_mesh(self, mesh_data, mesh_name, source_file_path, source_file_type, parent_name=None):
        """Export vertex-animated mesh with embedded first-frame geometry

        Vertex-animated meshes are exported with their first-frame geometry embedded
        directly in the Maya file. This ensures the mesh is visible in Maya even
        without the original source file. A comment notes how to reconnect animation.

        Args:
            mesh_data: MeshData from SceneData
            mesh_name: Sanitized mesh name
            source_file_path: Path to source file (for reference comment)
            source_file_type: 'alembic' or 'usd'
            parent_name: Optional parent node name for hierarchy
        """
        lines = []

        # Create transform (with parent if specified)
        if parent_name and parent_name in self.created_nodes:
            lines.append(f'createNode transform -n "{mesh_name}" -p "{parent_name}";')
        else:
            lines.append(f'createNode transform -n "{mesh_name}";')

        # Set initial transform values from first keyframe (using Maya-compatible rotation)
        if mesh_data.keyframes:
            kf = mesh_data.keyframes[0]
            pos = kf.position
            rot = kf.rotation_maya  # Use Maya-compatible rotation
            scale = kf.scale
            lines.append(f'    setAttr ".t" -type "double3" {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f};')
            lines.append(f'    setAttr ".r" -type "double3" {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f};')
            lines.append(f'    setAttr ".s" -type "double3" {scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f};')

        # Get mesh geometry from SceneData (first frame)
        positions = mesh_data.geometry.positions
        indices = mesh_data.geometry.indices
        counts = mesh_data.geometry.counts

        # Create mesh shape with embedded geometry
        shape_name = f"{mesh_name}Shape"
        self.mesh_shapes.append(shape_name)

        lines.append(f'createNode mesh -n "{shape_name}" -p "{mesh_name}";')
        lines.append(f'    setAttr -k off ".v";')
        lines.append(f'    setAttr ".vir" yes;')
        lines.append(f'    setAttr ".vif" yes;')

        num_verts = len(positions)
        num_faces = len(counts)

        # Build edges from face data
        edges = []
        edge_map = {}
        idx_offset = 0

        for count in counts:
            face_verts = [indices[idx_offset + i] for i in range(count)]
            for i in range(count):
                v1, v2 = face_verts[i], face_verts[(i + 1) % count]
                edge_key = (min(v1, v2), max(v1, v2))
                if edge_key not in edge_map:
                    edge_map[edge_key] = len(edges)
                    edges.append((v1, v2))
            idx_offset += count

        num_edges = len(edges)

        # Build the mesh data in official Maya format using setAttr -type mesh
        mesh_data_parts = []

        # Vertices: "v" count x y z x y z ...
        mesh_data_parts.append(f'"v" {num_verts}')
        for pos in positions:
            mesh_data_parts.append(f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}')

        # Vertex normals: "vn" 0 (required, set to 0)
        mesh_data_parts.append('"vn" 0')

        # Edges: "e" count v1 v2 "smooth" v1 v2 "smooth" ...
        mesh_data_parts.append(f'"e" {num_edges}')
        for v1, v2 in edges:
            mesh_data_parts.append(f'{v1} {v2} "smooth"')

        # Faces: "face" "l" edgeCount edge1 edge2 ... "face" "l" ...
        idx_offset = 0
        for face_idx, count in enumerate(counts):
            face_verts = [indices[idx_offset + i] for i in range(count)]
            # Reverse winding order - Alembic uses opposite convention from Maya
            face_verts.reverse()

            # Get edge indices for this face
            face_edges = []
            for i in range(count):
                v1, v2 = face_verts[i], face_verts[(i + 1) % count]
                edge_key = (min(v1, v2), max(v1, v2))
                edge_idx = edge_map[edge_key]
                # Check edge direction - negative means reversed
                if edges[edge_idx][0] == v1:
                    face_edges.append(edge_idx)
                else:
                    face_edges.append(-edge_idx - 1)

            edge_str = ' '.join(str(e) for e in face_edges)
            mesh_data_parts.append(f'"face" "l" {count} {edge_str}')
            idx_offset += count

        # Write mesh data as single setAttr command
        lines.append(f'    setAttr ".o" -type "mesh"')
        lines.append(f'        {" ".join(mesh_data_parts)};')

        # Add comment about vertex animation
        if source_file_path:
            source_name = source_file_path.split('/')[-1].split('\\')[-1]
            lines.extend([
                f'',
                f'// NOTE: This mesh has vertex animation (deformation)',
                f'// First-frame geometry is embedded above',
                f'// To restore animation, use AlembicNode with original file:',
                f'//   Source: {source_name}',
                f'//   Object path: {mesh_data.full_path}',
            ])

        return lines

    def _export_locator(self, transform_data, locator_name, parent_name=None):
        """Export locator with animation from TransformData

        Creates a Maya locator node with animated transform.

        Args:
            transform_data: TransformData from SceneData
            locator_name: Sanitized name for the locator
            parent_name: Optional parent node name for hierarchy

        Returns:
            list: Maya ASCII lines for this locator
        """
        lines = []

        # Create transform (with parent if specified)
        if parent_name and parent_name in self.created_nodes:
            lines.append(f'createNode transform -n "{locator_name}" -p "{parent_name}";')
        else:
            lines.append(f'createNode transform -n "{locator_name}";')

        # Set initial transform values from first keyframe (using Maya-compatible rotation)
        if transform_data.keyframes:
            kf = transform_data.keyframes[0]
            pos = kf.position
            rot = kf.rotation_maya  # Use Maya-compatible rotation
            scale = kf.scale
            lines.append(f'    setAttr ".t" -type "double3" {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f};')
            lines.append(f'    setAttr ".r" -type "double3" {rot[0]:.6f} {rot[1]:.6f} {rot[2]:.6f};')
            lines.append(f'    setAttr ".s" -type "double3" {scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f};')

        # Create locator shape
        shape_name = f"{locator_name}Shape"
        lines.extend([
            f'createNode locator -n "{shape_name}" -p "{locator_name}";',
            f'    setAttr -k off ".v";',
        ])

        # Add animation from keyframes
        lines.extend(self._animate_transform_from_keyframes(transform_data.keyframes, locator_name))

        return lines

    def _animate_transform_from_keyframes(self, keyframes, node_name):
        """Create animation curves from pre-extracted keyframes"""
        lines = []

        if not keyframes:
            return lines

        # Extract values from keyframes (using Maya-compatible rotation)
        times = [kf.frame for kf in keyframes]
        tx = [kf.position[0] for kf in keyframes]
        ty = [kf.position[1] for kf in keyframes]
        tz = [kf.position[2] for kf in keyframes]
        rx = [kf.rotation_maya[0] for kf in keyframes]
        ry = [kf.rotation_maya[1] for kf in keyframes]
        rz = [kf.rotation_maya[2] for kf in keyframes]
        sx = [kf.scale[0] for kf in keyframes]
        sy = [kf.scale[1] for kf in keyframes]
        sz = [kf.scale[2] for kf in keyframes]

        def is_animated(vals):
            return len(set(round(v, 6) for v in vals)) > 1

        curves = [
            ('translateX', 'TL', 'tx', tx),
            ('translateY', 'TL', 'ty', ty),
            ('translateZ', 'TL', 'tz', tz),
            ('rotateX', 'TA', 'rx', rx),
            ('rotateY', 'TA', 'ry', ry),
            ('rotateZ', 'TA', 'rz', rz),
            ('scaleX', 'TU', 'sx', sx),
            ('scaleY', 'TU', 'sy', sy),
            ('scaleZ', 'TU', 'sz', sz),
        ]

        for attr, curve_type, short, vals in curves:
            if is_animated(vals):
                curve_name = f"{node_name}_{attr}"
                lines.append(f'createNode animCurve{curve_type} -n "{curve_name}";')
                lines.append(f'    setAttr ".tan" 18;')
                lines.append(f'    setAttr ".wgt" no;')
                lines.append(f'    setAttr -s {len(times)} ".ktv[0:{len(times)-1}]"')
                for i, (t, v) in enumerate(zip(times, vals)):
                    suffix = ';' if i == len(times) - 1 else ''
                    lines.append(f'        {t} {v:.6f}{suffix}')
                lines.append(f'connectAttr "{curve_name}.o" "{node_name}.{short}";')

        return lines

    # === SHADING CONNECTIONS ===

    def _generate_shading_connections(self):
        """Connect meshes to default shading group"""
        lines = ['// Shading connections', '']

        for i, shape in enumerate(self.mesh_shapes):
            lines.append(f'connectAttr "{shape}.iog" ":initialShadingGroup.dsm" -na;')

        lines.append('')
        return lines

    def _generate_default_connections(self):
        """Generate default Maya scene connections"""
        return [
            '// Default connections',
            'connectAttr "layerManager.dli[0]" "defaultLayer.id";',
            'connectAttr "renderLayerManager.rlmi[0]" "defaultRenderLayer.rlid";',
            'connectAttr ":defaultArnoldDisplayDriver.msg" ":defaultArnoldRenderOptions.drivers" -na;',
            'connectAttr ":defaultArnoldFilter.msg" ":defaultArnoldRenderOptions.filter";',
            'connectAttr ":defaultArnoldDriver.msg" ":defaultArnoldRenderOptions.driver";',
            'connectAttr "lambert1.oc" "initialShadingGroup.ss";',
            'connectAttr "initialShadingGroup.msg" "initialMaterialInfo.sg";',
            'connectAttr "lambert1.msg" "initialMaterialInfo.m";',
            '// End of file',
        ]

    # === HIERARCHY UTILITIES ===

    def _build_hierarchy_map(self, scene_data: SceneData):
        """Build hierarchy map from full_path data

        Parses full_path strings to determine parent-child relationships.
        This allows reconstruction of the original scene hierarchy.

        Args:
            scene_data: SceneData with full_path information

        Returns:
            dict: Mapping of node_name -> parent_name (sanitized names)
        """
        hierarchy = {}  # node_name -> parent_name
        all_paths = []

        # Collect all full paths
        for cam in scene_data.cameras:
            all_paths.append(cam.full_path)
        for mesh in scene_data.meshes:
            all_paths.append(mesh.full_path)
        for xform in scene_data.transforms:
            all_paths.append(xform.full_path)

        # Parse paths to build hierarchy
        # Path format: "/GroupA/GroupB/ObjectName" or "/GroupA/Transform/ShapeNode"
        for full_path in all_paths:
            parts = [p for p in full_path.split('/') if p]
            if len(parts) < 2:
                continue

            # Build relationships for all intermediate nodes
            for i in range(1, len(parts)):
                child = self._sanitize_name(parts[i])
                parent = self._sanitize_name(parts[i - 1])

                # Skip if child == parent (shouldn't happen, but safety check)
                if child != parent:
                    hierarchy[child] = parent

        return hierarchy

    def _get_hierarchy_groups(self, scene_data: SceneData):
        """Get list of hierarchy groups that need to be created

        Finds intermediate transform nodes that aren't cameras, meshes, or locators
        but are parents in the hierarchy.

        Args:
            scene_data: SceneData instance

        Returns:
            list: List of (group_name, parent_name) tuples in creation order (parents first)
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
        group_depths = {}  # group_name -> depth (for sorting)

        for item in list(scene_data.cameras) + list(scene_data.meshes) + list(scene_data.transforms):
            full_path = item.full_path
            parts = [p for p in full_path.split('/') if p]

            # Walk up the path, finding groups that aren't in known_nodes
            for i, part in enumerate(parts[:-1]):  # Exclude the leaf node
                sanitized = self._sanitize_name(part)

                # If this intermediate node is not a known node, it's a group
                if sanitized not in known_nodes:
                    # Determine parent (if any)
                    if i > 0:
                        parent = self._sanitize_name(parts[i - 1])
                    else:
                        parent = None

                    hierarchy_groups[sanitized] = parent
                    group_depths[sanitized] = i

        # Sort groups by depth (parents before children)
        sorted_groups = sorted(hierarchy_groups.items(), key=lambda x: group_depths.get(x[0], 0))

        return sorted_groups

    def _get_node_parent(self, full_path, hierarchy_map, display_name=None):
        """Get the immediate parent node name for a given object

        Detects shape nodes by comparing the last path component to the
        display_name. If they differ, the object is a shape node and the
        parent is the grandparent (skipping the shape's transform).

        Args:
            full_path: Full hierarchy path (e.g., "/Group/SubGroup/Object")
            hierarchy_map: Hierarchy map from _build_hierarchy_map
            display_name: Display name of this node (parent_name or name)

        Returns:
            str or None: Parent node name (sanitized), or None if at root
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

    # === UTILITIES ===

    def _sanitize_name(self, name):
        """Sanitize name for Maya"""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"obj_{sanitized}"
        return sanitized or "unnamed"

    def _mel_escape_string(self, s):
        """Escape string for MEL"""
        if s is None:
            return ""
        s = str(s).replace('\\', '/')
        s = s.replace('"', '\\"')
        return s
