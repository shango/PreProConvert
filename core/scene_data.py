#!/usr/bin/env python3
"""
Scene Data Module
Format-agnostic data structures for scene representation.

This module defines the intermediate data structures that decouple
readers (Alembic, USD) from exporters (AE, USD, Maya MA). Readers
extract scene data into these structures, and exporters consume them
without knowledge of the source format.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
import re


class AnimationType(Enum):
    """Animation classification for mesh objects"""
    STATIC = "static"
    TRANSFORM_ONLY = "transform_only"
    VERTEX_ANIMATED = "vertex_animated"
    BLEND_SHAPE = "blend_shape"  # Vertex animation via blend shapes (exportable to FBX)


# === ROTATION DECOMPOSITION FUNCTIONS ===
# These extract XYZ Euler angles from a 3x3 rotation matrix.
# Moved from alembic_reader.py to enable lazy computation in Keyframe.

def decompose_rotation_ae(rotation_matrix: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Decompose 3x3 rotation matrix to XYZ Euler angles (After Effects compatible)

    Uses column-major decomposition convention for After Effects compatibility.

    Args:
        rotation_matrix: 9 floats representing row-major 3x3 rotation matrix

    Returns:
        (rx, ry, rz) in degrees
    """
    import math

    # Unpack 3x3 matrix (row-major storage)
    rot = [
        [rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]],
        [rotation_matrix[3], rotation_matrix[4], rotation_matrix[5]],
        [rotation_matrix[6], rotation_matrix[7], rotation_matrix[8]]
    ]

    # Column-major decomposition for After Effects
    sy_test = math.sqrt(rot[0][0]**2 + rot[1][0]**2)

    if sy_test > 1e-6:
        # Normal case
        x = math.atan2(rot[2][1], rot[2][2])
        y = math.atan2(-rot[2][0], sy_test)
        z = math.atan2(rot[1][0], rot[0][0])
    else:
        # Gimbal lock case
        x = math.atan2(-rot[1][2], rot[1][1])
        y = math.atan2(-rot[2][0], sy_test)
        z = 0

    return (math.degrees(x), math.degrees(y), math.degrees(z))


def decompose_rotation_maya(rotation_matrix: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Decompose 3x3 rotation matrix to XYZ Euler angles (Maya/USD compatible)

    Uses row-major decomposition convention for Maya/USD compatibility.

    Args:
        rotation_matrix: 9 floats representing row-major 3x3 rotation matrix

    Returns:
        (rx, ry, rz) in degrees
    """
    import math

    # Unpack 3x3 matrix (row-major storage)
    rot = [
        [rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]],
        [rotation_matrix[3], rotation_matrix[4], rotation_matrix[5]],
        [rotation_matrix[6], rotation_matrix[7], rotation_matrix[8]]
    ]

    # Row-major decomposition for Maya/USD
    cy = math.sqrt(rot[0][0]**2 + rot[0][1]**2)

    if cy > 1e-6:
        # Normal case
        x = math.atan2(rot[1][2], rot[2][2])
        y = math.atan2(-rot[0][2], cy)
        z = math.atan2(-rot[0][1], rot[0][0])  # Negated for correct sign
    else:
        # Gimbal lock case
        x = math.atan2(-rot[2][1], rot[1][1])
        y = math.atan2(-rot[0][2], cy)
        z = 0

    return (math.degrees(x), math.degrees(y), math.degrees(z))


def transform_vertices_to_local(
    positions: List[Tuple[float, float, float]],
    world_position: Tuple[float, float, float],
    rotation_matrix: Optional[Tuple[float, ...]] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> List[Tuple[float, float, float]]:
    """Transform world-space vertices to local space

    Applies the inverse of the object's world transform to convert
    world-space vertex positions to local (object) space.

    The transformation order for world-to-local is:
    1. Translate by -position (move to origin)
    2. Rotate by inverse rotation
    3. Scale by 1/scale

    Args:
        positions: List of vertex positions in world space [(x,y,z), ...]
        world_position: Object's world position (translation)
        rotation_matrix: Optional 3x3 rotation matrix (9 floats, row-major).
                        If None, only translation is applied.
        scale: Object's scale factors [sx, sy, sz]

    Returns:
        List of vertex positions in local space
    """
    import math

    local_positions = []

    # Precompute inverse scale (avoid division by zero)
    inv_scale = (
        1.0 / scale[0] if abs(scale[0]) > 1e-10 else 1.0,
        1.0 / scale[1] if abs(scale[1]) > 1e-10 else 1.0,
        1.0 / scale[2] if abs(scale[2]) > 1e-10 else 1.0
    )

    # Precompute inverse rotation matrix (transpose of orthonormal rotation)
    if rotation_matrix is not None:
        # For orthonormal rotation matrix, inverse = transpose
        # Row-major 3x3: [r00,r01,r02, r10,r11,r12, r20,r21,r22]
        # Transpose swaps rows and columns
        inv_rot = (
            rotation_matrix[0], rotation_matrix[3], rotation_matrix[6],  # Column 0 -> Row 0
            rotation_matrix[1], rotation_matrix[4], rotation_matrix[7],  # Column 1 -> Row 1
            rotation_matrix[2], rotation_matrix[5], rotation_matrix[8],  # Column 2 -> Row 2
        )
    else:
        inv_rot = None

    for pos in positions:
        # Step 1: Translate to origin (subtract world position)
        x = pos[0] - world_position[0]
        y = pos[1] - world_position[1]
        z = pos[2] - world_position[2]

        # Step 2: Apply inverse rotation (if provided)
        if inv_rot is not None:
            rx = inv_rot[0] * x + inv_rot[1] * y + inv_rot[2] * z
            ry = inv_rot[3] * x + inv_rot[4] * y + inv_rot[5] * z
            rz = inv_rot[6] * x + inv_rot[7] * y + inv_rot[8] * z
            x, y, z = rx, ry, rz

        # Step 3: Apply inverse scale
        x *= inv_scale[0]
        y *= inv_scale[1]
        z *= inv_scale[2]

        local_positions.append((x, y, z))

    return local_positions


@dataclass
class Keyframe:
    """Single animation keyframe with transform data

    Supports two creation modes:
    1. Direct: rotation_ae and rotation_maya provided directly
    2. Lazy: rotation_matrix provided, rotations computed on first access

    Attributes:
        frame: 1-based frame number
        position: [x, y, z] translation in scene units
        scale: [sx, sy, sz] scale multipliers
        rotation_ae: [rx, ry, rz] degrees - After Effects compatible (computed lazily if matrix provided)
        rotation_maya: [rx, ry, rz] degrees - Maya/USD compatible (computed lazily if matrix provided)
        rotation_matrix: Optional 3x3 rotation matrix (9 floats, row-major) for lazy decomposition
    """
    frame: int
    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    # Internal storage for rotation - underscore prefix for lazy property pattern
    _rotation_ae: Optional[Tuple[float, float, float]] = None
    _rotation_maya: Optional[Tuple[float, float, float]] = None
    _rotation_matrix: Optional[Tuple[float, ...]] = field(default=None, repr=False)

    @property
    def rotation_ae(self) -> Tuple[float, float, float]:
        """Get After Effects compatible rotation (lazy computed if matrix provided)"""
        if self._rotation_ae is None and self._rotation_matrix is not None:
            # Lazy compute and cache
            object.__setattr__(self, '_rotation_ae', decompose_rotation_ae(self._rotation_matrix))
        return self._rotation_ae

    @property
    def rotation_maya(self) -> Tuple[float, float, float]:
        """Get Maya/USD compatible rotation (lazy computed if matrix provided)"""
        if self._rotation_maya is None and self._rotation_matrix is not None:
            # Lazy compute and cache
            object.__setattr__(self, '_rotation_maya', decompose_rotation_maya(self._rotation_matrix))
        return self._rotation_maya

    @classmethod
    def create(cls, frame: int, position: Tuple[float, float, float],
               scale: Tuple[float, float, float],
               rotation_ae: Tuple[float, float, float] = None,
               rotation_maya: Tuple[float, float, float] = None,
               rotation_matrix: Tuple[float, ...] = None) -> 'Keyframe':
        """Create a Keyframe with either direct rotations or matrix for lazy computation

        Args:
            frame: 1-based frame number
            position: Translation [x, y, z]
            scale: Scale [sx, sy, sz]
            rotation_ae: AE-compatible rotation (optional if matrix provided)
            rotation_maya: Maya-compatible rotation (optional if matrix provided)
            rotation_matrix: 3x3 rotation matrix for lazy decomposition (9 floats)

        Returns:
            Keyframe instance
        """
        return cls(
            frame=frame,
            position=position,
            scale=scale,
            _rotation_ae=rotation_ae,
            _rotation_maya=rotation_maya,
            _rotation_matrix=rotation_matrix
        )


@dataclass
class CameraProperties:
    """Camera-specific optical properties

    Attributes:
        focal_length: Focal length in mm
        h_aperture: Horizontal aperture in cm (Alembic convention)
        v_aperture: Vertical aperture in cm (Alembic convention)
    """
    focal_length: float
    h_aperture: float
    v_aperture: float


@dataclass
class CameraData:
    """Complete camera data with animation

    Attributes:
        name: Camera object name
        parent_name: Parent transform name if camera is nested, None otherwise
        full_path: Full hierarchy path (e.g., "/World/Camera/CameraShape")
        properties: Camera optical properties (focal length, apertures)
        keyframes: Pre-extracted animation keyframes for all frames
    """
    name: str
    parent_name: Optional[str]
    full_path: str
    properties: CameraProperties
    keyframes: List[Keyframe]


@dataclass
class MeshGeometry:
    """Static mesh geometry data (first frame)

    Vertex positions are always stored in local (object) space.
    Readers are responsible for extracting vertices in local space.

    Attributes:
        positions: List of vertex positions in LOCAL space as [x, y, z] tuples
        indices: Face vertex indices (flattened)
        counts: Number of vertices per face
    """
    positions: List[Tuple[float, float, float]]
    indices: List[int]
    counts: List[int]


@dataclass
class BlendShapeTarget:
    """Single blend shape target with delta positions

    Attributes:
        name: Target name (e.g., "smile", "blink")
        vertex_indices: List of affected vertex indices (sparse storage)
        deltas: Delta positions for each affected vertex [(dx,dy,dz), ...]
        full_weight: Weight value at which target is fully applied (default 1.0)
    """
    name: str
    vertex_indices: List[int]
    deltas: List[Tuple[float, float, float]]
    full_weight: float = 1.0


@dataclass
class BlendShapeWeightKey:
    """Keyframe for blend shape weight animation

    Attributes:
        frame: Frame number
        weight: Weight value (0.0 to 1.0)
    """
    frame: int
    weight: float


@dataclass
class BlendShapeChannel:
    """Blend shape channel controlling one or more targets

    A channel represents a single animatable weight that controls one target
    (or multiple targets for in-between/progressive morphs).

    Attributes:
        name: Channel name (often same as target name)
        targets: List of shape targets (usually 1, multiple for in-betweens)
        weight_animation: Optional animated weights, None if static
        default_weight: Static weight if not animated (0.0 to 1.0)
    """
    name: str
    targets: List[BlendShapeTarget]
    weight_animation: Optional[List[BlendShapeWeightKey]] = None
    default_weight: float = 0.0


@dataclass
class BlendShapeDeformer:
    """Complete blend shape deformer with all channels

    Attributes:
        name: Deformer node name
        channels: All blend shape channels
        base_mesh_name: Name of the mesh being deformed
    """
    name: str
    channels: List[BlendShapeChannel]
    base_mesh_name: str


@dataclass
class MeshData:
    """Complete mesh data with animation and geometry

    Attributes:
        name: Mesh object name
        parent_name: Parent transform name if mesh is nested
        full_path: Full hierarchy path
        animation_type: Classification (STATIC, TRANSFORM_ONLY, VERTEX_ANIMATED, BLEND_SHAPE)
        keyframes: Transform animation keyframes (empty if static)
        geometry: First frame geometry (positions, indices, counts)
        vertex_positions_per_frame: Per-frame vertex positions if vertex-animated
        blend_shapes: Blend shape deformer data if mesh has blend shapes
    """
    name: str
    parent_name: Optional[str]
    full_path: str
    animation_type: AnimationType
    keyframes: List[Keyframe]
    geometry: MeshGeometry
    vertex_positions_per_frame: Optional[Dict[int, List[Tuple[float, float, float]]]] = None
    blend_shapes: Optional[BlendShapeDeformer] = None

    def get_local_positions(self) -> List[Tuple[float, float, float]]:
        """Get vertex positions in local (object) space

        SceneData always stores vertices in local space, so this method
        simply returns geometry.positions. Kept for backward compatibility.

        Returns:
            List of vertex positions in local space
        """
        return self.geometry.positions


@dataclass
class TransformData:
    """Transform/locator data with animation

    Used for pure transforms that don't have camera or mesh shapes attached.
    These become nulls/locators in the exported scene.

    Attributes:
        name: Transform object name
        parent_name: Parent transform name if nested
        full_path: Full hierarchy path
        keyframes: Animation keyframes for all frames
    """
    name: str
    parent_name: Optional[str]
    full_path: str
    keyframes: List[Keyframe]


@dataclass
class SceneMetadata:
    """Scene-level metadata

    Attributes:
        width: Render width in pixels
        height: Render height in pixels
        fps: Frames per second
        frame_count: Total number of frames
        start_frame: First frame number (e.g., 1 or 1001 for VFX pipelines)
        footage_path: Path to associated footage file (if embedded in scene)
        source_file_path: Absolute path to the source file
        source_format_name: Human-readable format name ("Alembic" or "USD")
    """
    width: int
    height: int
    fps: float
    frame_count: int
    start_frame: int
    footage_path: Optional[str]
    source_file_path: str
    source_format_name: str


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in export formats (FBX, Maya, USD)

    Replaces non-alphanumeric characters with underscores and ensures
    the name doesn't start with a digit.

    Args:
        name: Original name to sanitize

    Returns:
        Sanitized name safe for all export formats
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"obj_{sanitized}"
    return sanitized or "unnamed"


@dataclass
class HierarchyNode:
    """Node in the pre-computed scene hierarchy tree

    Represents a single node (camera, mesh, transform, or intermediate group)
    in the scene hierarchy with links to parent and children.

    Attributes:
        name: Sanitized name (safe for export formats)
        original_name: Original unsanitized name
        full_path: Full hierarchy path from source (e.g., "/Group/SubGroup/Object")
        node_type: Type of node ("camera", "mesh", "transform", "group")
        depth: Depth in hierarchy (0 = root level)
        parent: Parent node reference (None for root-level nodes)
        children: List of child node references
    """
    name: str
    original_name: str
    full_path: str
    node_type: str = "group"  # "camera", "mesh", "transform", "group"
    depth: int = 0
    parent: Optional['HierarchyNode'] = None
    children: List['HierarchyNode'] = field(default_factory=list)


@dataclass
class HierarchyTree:
    """Pre-computed scene hierarchy tree

    Centralizes hierarchy parsing that was previously duplicated across
    FBX, USD, and Maya exporters. Built once by the reader, consumed by
    all exporters.

    Attributes:
        nodes_by_name: Quick lookup by sanitized name
        nodes_by_path: Quick lookup by original full_path
        intermediate_groups: Groups that aren't cameras/meshes/transforms
    """
    nodes_by_name: Dict[str, HierarchyNode] = field(default_factory=dict)
    nodes_by_path: Dict[str, HierarchyNode] = field(default_factory=dict)
    intermediate_groups: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    def get_parent(self, node_name: str) -> Optional[str]:
        """Get parent name for a node

        Args:
            node_name: Sanitized node name

        Returns:
            Parent's sanitized name, or None if no parent
        """
        node = self.nodes_by_name.get(node_name)
        if node and node.parent:
            return node.parent.name
        return None

    def get_intermediate_groups(self) -> List[Tuple[str, Optional[str]]]:
        """Get hierarchy groups that need to be created as Null/Xform nodes

        These are nodes that appear in full_path but aren't cameras, meshes,
        or transforms themselves. Sorted by depth (parents first).

        Returns:
            List of (group_name, parent_name) tuples in creation order
        """
        return self.intermediate_groups

    def get_node_parent_from_path(self, full_path: str) -> Optional[str]:
        """Get parent node name from a full_path

        Handles the Alembic convention where shapes have transform parents.
        For "/Group/Transform/Shape", returns "Group" (grandparent).
        For "/Group/Transform", returns "Group" (parent).

        Args:
            full_path: Full hierarchy path

        Returns:
            Sanitized parent name, or None if root-level
        """
        parts = [p for p in full_path.split('/') if p]
        if len(parts) < 2:
            return None

        obj_name = parts[-1]
        if obj_name.endswith('Shape') and len(parts) >= 3:
            return _sanitize_name(parts[-3])
        elif len(parts) >= 2:
            return _sanitize_name(parts[-2])

        return None

    @classmethod
    def build_from_scene_items(
        cls,
        cameras: List['CameraData'],
        meshes: List['MeshData'],
        transforms: List['TransformData']
    ) -> 'HierarchyTree':
        """Build hierarchy tree from scene items

        Parses full_path strings from all scene objects to build the
        complete hierarchy tree with parent-child relationships.

        Args:
            cameras: List of camera data
            meshes: List of mesh data
            transforms: List of transform data

        Returns:
            Populated HierarchyTree instance
        """
        tree = cls()

        # Collect known nodes (cameras, meshes, transforms we'll create)
        known_nodes = set()

        for cam in cameras:
            display_name = cam.parent_name if cam.parent_name else cam.name
            sanitized = _sanitize_name(display_name)
            known_nodes.add(sanitized)
            tree.nodes_by_name[sanitized] = HierarchyNode(
                name=sanitized,
                original_name=display_name,
                full_path=cam.full_path,
                node_type="camera"
            )
            tree.nodes_by_path[cam.full_path] = tree.nodes_by_name[sanitized]

        for mesh in meshes:
            display_name = mesh.parent_name if mesh.parent_name else mesh.name
            sanitized = _sanitize_name(display_name)
            known_nodes.add(sanitized)
            tree.nodes_by_name[sanitized] = HierarchyNode(
                name=sanitized,
                original_name=display_name,
                full_path=mesh.full_path,
                node_type="mesh"
            )
            tree.nodes_by_path[mesh.full_path] = tree.nodes_by_name[sanitized]

        for xform in transforms:
            sanitized = _sanitize_name(xform.name)
            known_nodes.add(sanitized)
            tree.nodes_by_name[sanitized] = HierarchyNode(
                name=sanitized,
                original_name=xform.name,
                full_path=xform.full_path,
                node_type="transform"
            )
            tree.nodes_by_path[xform.full_path] = tree.nodes_by_name[sanitized]

        # Find intermediate groups from paths
        hierarchy_groups = {}  # group_name -> parent_name
        group_depths = {}  # group_name -> depth

        all_items = list(cameras) + list(meshes) + list(transforms)
        for item in all_items:
            parts = [p for p in item.full_path.split('/') if p]

            for i, part in enumerate(parts[:-1]):
                sanitized = _sanitize_name(part)

                if sanitized not in known_nodes and sanitized not in hierarchy_groups:
                    parent = _sanitize_name(parts[i - 1]) if i > 0 else None
                    hierarchy_groups[sanitized] = parent
                    group_depths[sanitized] = i

                    # Create node for group
                    group_path = '/' + '/'.join(parts[:i + 1])
                    tree.nodes_by_name[sanitized] = HierarchyNode(
                        name=sanitized,
                        original_name=part,
                        full_path=group_path,
                        node_type="group",
                        depth=i
                    )

        # Build parent-child relationships
        for name, node in tree.nodes_by_name.items():
            parent_name = tree.get_node_parent_from_path(node.full_path)
            if parent_name and parent_name in tree.nodes_by_name:
                parent_node = tree.nodes_by_name[parent_name]
                node.parent = parent_node
                if node not in parent_node.children:
                    parent_node.children.append(node)

        # Sort groups by depth (parents first)
        sorted_groups = sorted(hierarchy_groups.items(), key=lambda x: group_depths.get(x[0], 0))
        tree.intermediate_groups = sorted_groups

        return tree


@dataclass
class AnimationCategories:
    """Pre-categorized mesh names by animation type

    This mirrors the output of AnimationDetector.analyze_scene() for
    backward compatibility and easy access.

    Attributes:
        vertex_animated: List of mesh names with raw vertex deformation (not exportable to FBX)
        blend_shape: List of mesh names with blend shape deformation (exportable to FBX)
        transform_only: List of mesh names with only transform animation
        static: List of mesh names with no animation
    """
    vertex_animated: List[str] = field(default_factory=list)
    blend_shape: List[str] = field(default_factory=list)
    transform_only: List[str] = field(default_factory=list)
    static: List[str] = field(default_factory=list)


@dataclass
class SceneData:
    """Complete scene data extracted from input file

    This is the format-agnostic intermediate representation that decouples
    readers from exporters. All animation is pre-extracted for all frames.

    Attributes:
        metadata: Scene-level information (resolution, fps, source file)
        cameras: All cameras with animation and properties
        meshes: All meshes with animation, geometry, and categorization
        transforms: Pure transforms/locators without shapes
        animation_categories: Quick lookup for mesh animation types
        hierarchy: Pre-computed hierarchy tree (optional for backward compat)
    """
    metadata: SceneMetadata
    cameras: List[CameraData]
    meshes: List[MeshData]
    transforms: List[TransformData]
    animation_categories: AnimationCategories
    hierarchy: Optional[HierarchyTree] = None

    def get_mesh_by_name(self, name: str) -> Optional[MeshData]:
        """Find mesh by name

        Args:
            name: Mesh name to find

        Returns:
            MeshData if found, None otherwise
        """
        for mesh in self.meshes:
            if mesh.name == name:
                return mesh
        return None

    def get_camera_by_name(self, name: str) -> Optional[CameraData]:
        """Find camera by name

        Args:
            name: Camera name to find

        Returns:
            CameraData if found, None otherwise
        """
        for cam in self.cameras:
            if cam.name == name:
                return cam
        return None

    def get_transform_by_name(self, name: str) -> Optional[TransformData]:
        """Find transform by name

        Args:
            name: Transform name to find

        Returns:
            TransformData if found, None otherwise
        """
        for xform in self.transforms:
            if xform.name == name:
                return xform
        return None
