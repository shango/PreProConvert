#!/usr/bin/env python3
"""
Maya ASCII Reader Module
Pure Python parser for Maya ASCII (.ma) files implementing the BaseReader interface.

No Maya installation required - parses the text format directly.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from .base_reader import BaseReader


class MayaNode:
    """Wrapper providing Alembic-compatible interface for parsed Maya nodes"""

    def __init__(self, name: str, node_type: str, parent_name: Optional[str] = None):
        self.name = name
        self.node_type = node_type  # 'transform', 'camera', 'mesh', 'animCurveTL', etc.
        self.parent_name = parent_name
        self.attributes: Dict[str, Any] = {}
        self.children: List['MayaNode'] = []
        self._parent: Optional['MayaNode'] = None
        self._full_path: Optional[str] = None

    def getName(self) -> str:
        """Alembic-compatible: Get node name"""
        return self.name

    def getFullName(self) -> str:
        """Alembic-compatible: Get full hierarchy path"""
        if self._full_path is not None:
            return self._full_path

        # Build path from parent chain
        parts = [self.name]
        current = self._parent
        while current is not None:
            parts.insert(0, current.name)
            current = current._parent
        self._full_path = "/" + "/".join(parts)
        return self._full_path

    def getParent(self) -> Optional['MayaNode']:
        """Alembic-compatible: Get parent node"""
        return self._parent

    def getHeader(self):
        """Alembic-compatible: Return self for type checking"""
        return self

    def __repr__(self):
        return f"MayaNode({self.name}, {self.node_type})"


class MayaAnimCurve:
    """Parsed animation curve with keyframe data"""

    def __init__(self, name: str, curve_type: str):
        self.name = name
        self.curve_type = curve_type  # 'TL' (translate), 'TA' (angle/rotate), 'TU' (unitless/scale)
        self.keyframes: List[Tuple[float, float]] = []  # [(frame, value), ...]
        self.target_node: Optional[str] = None
        self.target_attr: Optional[str] = None
        self.pre_infinity = 'constant'
        self.post_infinity = 'constant'

    def get_value_at_frame(self, frame: float) -> float:
        """Get interpolated value at a specific frame"""
        if not self.keyframes:
            return 0.0

        # Sort keyframes by frame number
        sorted_kf = sorted(self.keyframes, key=lambda x: x[0])

        # Before first keyframe
        if frame <= sorted_kf[0][0]:
            return sorted_kf[0][1]

        # After last keyframe
        if frame >= sorted_kf[-1][0]:
            return sorted_kf[-1][1]

        # Find surrounding keyframes and interpolate (linear)
        for i in range(len(sorted_kf) - 1):
            f1, v1 = sorted_kf[i]
            f2, v2 = sorted_kf[i + 1]
            if f1 <= frame <= f2:
                # Linear interpolation
                t = (frame - f1) / (f2 - f1) if f2 != f1 else 0
                return v1 + t * (v2 - v1)

        return sorted_kf[-1][1]


class MayaBlendShapeData:
    """Parsed blend shape deformer data"""

    def __init__(self, name: str):
        self.name = name
        # targets[target_idx] = {weight_idx: {'deltas': [...], 'components': [...]}}
        self.targets: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self.weight_aliases: Dict[int, str] = {}  # target_idx -> weight alias name
        self.connected_mesh: Optional[str] = None

    def add_deltas(self, target_idx: int, weight_idx: int, deltas: List[Tuple[float, float, float]]):
        """Add delta positions for a target at a weight index"""
        if target_idx not in self.targets:
            self.targets[target_idx] = {}
        if weight_idx not in self.targets[target_idx]:
            self.targets[target_idx][weight_idx] = {'deltas': [], 'components': []}
        self.targets[target_idx][weight_idx]['deltas'] = deltas

    def add_components(self, target_idx: int, weight_idx: int, components: List[int]):
        """Add component indices for a target at a weight index"""
        if target_idx not in self.targets:
            self.targets[target_idx] = {}
        if weight_idx not in self.targets[target_idx]:
            self.targets[target_idx][weight_idx] = {'deltas': [], 'components': []}
        self.targets[target_idx][weight_idx]['components'] = components


class MayaScene:
    """Container for parsed Maya scene data"""

    def __init__(self):
        self.nodes: Dict[str, MayaNode] = {}
        self.anim_curves: Dict[str, MayaAnimCurve] = {}
        self.blend_shapes: Dict[str, MayaBlendShapeData] = {}  # blendShape nodes
        self.connections: List[Tuple[str, str]] = []  # [(source, dest), ...]
        self.fps: float = 24.0
        self.start_frame: float = 1.0
        self.end_frame: float = 120.0
        self.linear_unit: str = 'cm'
        self.angular_unit: str = 'deg'

    def get_node(self, name: str) -> Optional[MayaNode]:
        return self.nodes.get(name)

    def get_cameras(self) -> List[MayaNode]:
        return [n for n in self.nodes.values() if n.node_type == 'camera']

    def get_meshes(self) -> List[MayaNode]:
        return [n for n in self.nodes.values() if n.node_type == 'mesh']

    def get_transforms(self) -> List[MayaNode]:
        return [n for n in self.nodes.values() if n.node_type == 'transform']

    # Attribute name aliases (short -> long and long -> short)
    ATTR_ALIASES = {
        # Short to long
        'tx': 'translateX', 'ty': 'translateY', 'tz': 'translateZ',
        'rx': 'rotateX', 'ry': 'rotateY', 'rz': 'rotateZ',
        'sx': 'scaleX', 'sy': 'scaleY', 'sz': 'scaleZ',
        # Long to short
        'translateX': 'tx', 'translateY': 'ty', 'translateZ': 'tz',
        'rotateX': 'rx', 'rotateY': 'ry', 'rotateZ': 'rz',
        'scaleX': 'sx', 'scaleY': 'sy', 'scaleZ': 'sz',
    }

    def get_anim_curve_for_attr(self, node_name: str, attr: str) -> Optional[MayaAnimCurve]:
        """Find animation curve connected to a specific node attribute

        Checks both short (tx, rx, sx) and long (translateX, rotateX, scaleX)
        attribute name formats.
        """
        # Build list of attribute names to check (original + alias)
        attrs_to_check = [attr]
        if attr in self.ATTR_ALIASES:
            attrs_to_check.append(self.ATTR_ALIASES[attr])

        for curve in self.anim_curves.values():
            if curve.target_node == node_name and curve.target_attr in attrs_to_check:
                return curve
        return None


class MayaASCIIParser:
    """Pure Python parser for Maya ASCII (.ma) file format"""

    def __init__(self):
        self.scene = MayaScene()
        self._current_node: Optional[MayaNode] = None
        self._current_curve: Optional[MayaAnimCurve] = None
        self._current_blend_shape: Optional[MayaBlendShapeData] = None

    def parse(self, file_path: str) -> MayaScene:
        """Parse a Maya ASCII file and return structured scene data"""
        self.scene = MayaScene()
        self._current_node = None
        self._current_curve = None
        self._current_blend_shape = None

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Process line by line, handling line continuations
        lines = self._preprocess_lines(content)

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('createNode '):
                self._parse_create_node(line)
            elif line.startswith('setAttr '):
                self._parse_set_attr(line)
            elif line.startswith('connectAttr '):
                self._parse_connect_attr(line)
            elif line.startswith('currentUnit '):
                self._parse_current_unit(line)
            elif line.startswith('playbackOptions '):
                self._parse_playback_options(line)

        # Build node hierarchy and link animations
        self._build_hierarchy()
        self._link_animations()

        return self.scene

    def _preprocess_lines(self, content: str) -> List[str]:
        """Preprocess content to handle multi-line statements"""
        lines = []
        current_line = ""

        for line in content.split('\n'):
            stripped = line.strip()

            # Skip empty lines and comments when not accumulating
            if not current_line and (not stripped or stripped.startswith('//')):
                continue

            current_line += " " + stripped if current_line else stripped

            # Check if statement is complete (ends with semicolon)
            if current_line.rstrip().endswith(';'):
                lines.append(current_line)
                current_line = ""

        return lines

    def _parse_create_node(self, line: str):
        """Parse createNode command: createNode type -n "name" [-p "parent"];"""
        # Extract node type
        match = re.match(r'createNode\s+(\w+)', line)
        if not match:
            return
        node_type = match.group(1)

        # Extract node name
        name_match = re.search(r'-n\s+"([^"]+)"', line)
        if not name_match:
            name_match = re.search(r"-n\s+'([^']+)'", line)
        name = name_match.group(1) if name_match else f"unnamed_{len(self.scene.nodes)}"

        # Extract parent name
        parent_match = re.search(r'-p\s+"([^"]+)"', line)
        if not parent_match:
            parent_match = re.search(r"-p\s+'([^']+)'", line)
        parent_name = parent_match.group(1) if parent_match else None

        # Handle animation curves specially
        if node_type.startswith('animCurve'):
            curve_type = node_type[9:]  # Extract TL, TA, TU, etc.
            curve = MayaAnimCurve(name, curve_type)
            self.scene.anim_curves[name] = curve
            self._current_curve = curve
            self._current_node = None
            self._current_blend_shape = None
        elif node_type == 'blendShape':
            # Handle blendShape deformer nodes
            bs_data = MayaBlendShapeData(name)
            self.scene.blend_shapes[name] = bs_data
            self._current_blend_shape = bs_data
            self._current_node = None
            self._current_curve = None
        else:
            node = MayaNode(name, node_type, parent_name)
            self.scene.nodes[name] = node
            self._current_node = node
            self._current_curve = None
            self._current_blend_shape = None

    def _parse_set_attr(self, line: str):
        """Parse setAttr command for attributes"""
        if self._current_curve:
            self._parse_anim_curve_attr(line)
        elif self._current_blend_shape:
            self._parse_blend_shape_attr(line)
        elif self._current_node:
            self._parse_node_attr(line)

    def _parse_anim_curve_attr(self, line: str):
        """Parse animation curve attributes (keyframes)"""
        curve = self._current_curve
        if not curve:
            return

        # Parse keyframe time-value pairs: setAttr -s N ".ktv[0:N]" frame1 value1 frame2 value2 ...
        ktv_match = re.search(r'\.ktv\[', line)
        if ktv_match:
            # Extract all numbers after the attribute specification
            # Find where the attribute ends (after the "]")
            bracket_end = line.find(']', ktv_match.start())
            if bracket_end != -1:
                values_str = line[bracket_end + 1:].rstrip(';').strip()
                # Remove quotes if present
                values_str = values_str.strip('"').strip("'")
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
                # Pairs of (frame, value)
                for i in range(0, len(numbers) - 1, 2):
                    frame = float(numbers[i])
                    value = float(numbers[i + 1])
                    curve.keyframes.append((frame, value))

    def _parse_blend_shape_attr(self, line: str):
        """Parse blendShape node attributes (targets, deltas, components)"""
        bs = self._current_blend_shape
        if not bs:
            return

        # Parse inputPointsTarget (delta positions)
        # Format: setAttr ".it[0].itg[0].iti[6000].ipt" -type "pointArray" N dx dy dz dx dy dz ...
        # Alternative short names: .inputTarget[0].inputTargetGroup[0].inputTargetItem[6000].inputPointsTarget
        ipt_match = re.search(r'\.(?:it|inputTarget)\[(\d+)\]\.(?:itg|inputTargetGroup)\[(\d+)\]\.(?:iti|inputTargetItem)\[(\d+)\]\.(?:ipt|inputPointsTarget)', line)
        if ipt_match and '-type "pointArray"' in line:
            geom_idx = int(ipt_match.group(1))  # Usually 0 (first deformed geometry)
            target_idx = int(ipt_match.group(2))  # Target shape index
            weight_idx = int(ipt_match.group(3))  # Weight index (6000 = 1.0 weight)

            # Extract pointArray: -type "pointArray" N x y z x y z ...
            pa_match = re.search(r'-type\s+"pointArray"\s+(\d+)\s+', line)
            if pa_match:
                count = int(pa_match.group(1))
                # Get everything after the count
                start_idx = pa_match.end()
                values_str = line[start_idx:].rstrip(';').strip()
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)

                # Group into triplets (x, y, z deltas)
                deltas = []
                for i in range(0, min(len(numbers), count * 3), 3):
                    if i + 2 < len(numbers):
                        deltas.append((float(numbers[i]), float(numbers[i + 1]), float(numbers[i + 2])))

                if deltas:
                    bs.add_deltas(target_idx, weight_idx, deltas)
            return

        # Parse inputComponentsTarget (affected vertex indices)
        # Format: setAttr ".it[0].itg[0].iti[6000].ict" -type "componentList" N "vtx[0:99]" "vtx[150]" ...
        ict_match = re.search(r'\.(?:it|inputTarget)\[(\d+)\]\.(?:itg|inputTargetGroup)\[(\d+)\]\.(?:iti|inputTargetItem)\[(\d+)\]\.(?:ict|inputComponentsTarget)', line)
        if ict_match and '-type "componentList"' in line:
            geom_idx = int(ict_match.group(1))
            target_idx = int(ict_match.group(2))
            weight_idx = int(ict_match.group(3))

            # Extract all vertex specifications: "vtx[N]" or "vtx[N:M]"
            vtx_specs = re.findall(r'"vtx\[(\d+)(?::(\d+))?\]"', line)
            components = []
            for spec in vtx_specs:
                start = int(spec[0])
                end = int(spec[1]) if spec[1] else start
                components.extend(range(start, end + 1))

            if components:
                bs.add_components(target_idx, weight_idx, components)
            return

        # Parse weight alias (target name)
        # Format: addAttr -ci true -k true -sn "smile" -ln "smile" -at "double" -min 0 -max 1;
        # Or via alias: aliasAttr "smile" ".w[0]";
        alias_match = re.search(r'aliasAttr\s+"([^"]+)"\s+"\.w\[(\d+)\]"', line)
        if alias_match:
            alias_name = alias_match.group(1)
            weight_idx = int(alias_match.group(2))
            bs.weight_aliases[weight_idx] = alias_name
            return

    def _parse_node_attr(self, line: str):
        """Parse node attributes (transform, geometry, camera properties)"""
        node = self._current_node
        if not node:
            return

        # Extract attribute name
        attr_match = re.search(r'setAttr\s+"\.([^"]+)"', line)
        if not attr_match:
            attr_match = re.search(r"setAttr\s+'\.([^']+)'", line)
        if not attr_match:
            attr_match = re.search(r'setAttr\s+\.(\w+)', line)
        if not attr_match:
            return

        attr_name = attr_match.group(1)

        # Handle different attribute types
        if '-type "double3"' in line or "-type 'double3'" in line:
            self._parse_double3_attr(node, attr_name, line)
        elif '-type "float3"' in line or "-type 'float3'" in line:
            self._parse_float3_array_attr(node, attr_name, line)
        elif '.vt[' in line or '.vrts[' in line:
            self._parse_vertex_attr(node, line)
        elif '.fc[' in line or '.face[' in line:
            self._parse_face_attr(node, line)
        elif '.ed[' in line or '.edge[' in line:
            self._parse_edge_attr(node, line)
        elif '.pnts[' in line:
            self._parse_points_attr(node, line)
        else:
            # Simple numeric attribute
            self._parse_simple_attr(node, attr_name, line)

    def _parse_double3_attr(self, node: MayaNode, attr_name: str, line: str):
        """Parse double3 attribute (translate, rotate, scale)"""
        # Find numbers after -type "double3"
        type_idx = line.find('double3"')
        if type_idx == -1:
            type_idx = line.find("double3'")
        if type_idx != -1:
            values_str = line[type_idx + 8:].rstrip(';').strip()
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
            if len(numbers) >= 3:
                node.attributes[attr_name] = [float(numbers[0]), float(numbers[1]), float(numbers[2])]

    def _parse_float3_array_attr(self, node: MayaNode, attr_name: str, line: str):
        """Parse float3 array attribute (vertices, normals, etc.)"""
        type_idx = line.find('float3"')
        if type_idx == -1:
            type_idx = line.find("float3'")
        if type_idx != -1:
            values_str = line[type_idx + 7:].rstrip(';').strip()
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
            # Group into triplets
            values = []
            for i in range(0, len(numbers) - 2, 3):
                values.append([float(numbers[i]), float(numbers[i + 1]), float(numbers[i + 2])])
            if values:
                if attr_name not in node.attributes:
                    node.attributes[attr_name] = []
                node.attributes[attr_name].extend(values)

    def _parse_vertex_attr(self, node: MayaNode, line: str):
        """Parse mesh vertex positions

        Handles two Maya formats:
        1. With type: setAttr ".vt[0:N]" -type "float3" x y z x y z ...
        2. Raw numbers: setAttr ".vt[0:N]" x y z x y z ...
        """
        if 'vertices' not in node.attributes:
            node.attributes['vertices'] = []

        # Find where the numeric data starts
        # Skip past attribute index like ".vt[0:4]"
        bracket_match = re.search(r'\.vt\[\d+:\d+\]', line)
        if not bracket_match:
            bracket_match = re.search(r'\.vrts\[\d+:\d+\]', line)
        if not bracket_match:
            bracket_match = re.search(r'\.vt\[\d+\]', line)

        if bracket_match:
            # Start parsing after the bracket
            data_start = bracket_match.end()
            values_str = line[data_start:].rstrip(';').strip()

            # Skip past -type "float3" if present
            if '-type' in values_str:
                type_match = re.search(r'-type\s+["\']?\w+["\']?\s*', values_str)
                if type_match:
                    values_str = values_str[type_match.end():]

            # Extract all floating point numbers
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
            for i in range(0, len(numbers) - 2, 3):
                node.attributes['vertices'].append(
                    [float(numbers[i]), float(numbers[i + 1]), float(numbers[i + 2])]
                )

    def _parse_face_attr(self, node: MayaNode, line: str):
        """Parse mesh face definitions (polyFaces format)

        Maya ASCII stores faces in polyFaces format:
        setAttr ".fc[0:N]" -type "polyFaces"
            f 3 e1 e2 e3    # triangle with 3 edge indices
            f 4 e1 e2 e3 e4 # quad with 4 edge indices

        Edge indices can be positive (forward) or negative (reversed).
        For negative index -N: use edge at index (N-1) reversed.
        """
        if 'polyfaces_raw' not in node.attributes:
            node.attributes['polyfaces_raw'] = []
            node.attributes['face_format'] = 'unknown'

        # Detect polyFaces format
        if '-type "polyFaces"' in line or '-type \'polyFaces\'' in line:
            node.attributes['face_format'] = 'polyFaces'

        # Parse "f N e1 e2 ..." entries (face definitions)
        # Each face starts with 'f' followed by vertex count and edge indices
        face_pattern = r'f\s+(\d+)((?:\s+[-]?\d+)+)'
        face_matches = re.findall(face_pattern, line)

        for match in face_matches:
            vertex_count = int(match[0])
            edge_indices_str = match[1].strip()
            edge_indices = [int(e) for e in edge_indices_str.split()]

            if len(edge_indices) == vertex_count:
                node.attributes['polyfaces_raw'].append({
                    'count': vertex_count,
                    'edges': edge_indices
                })

    def _parse_edge_attr(self, node: MayaNode, line: str):
        """Parse mesh edge definitions

        Maya ASCII stores edges as triplets: start_vertex end_vertex smooth_flag
        Format: setAttr ".ed[0:N]" v1 v2 smooth v3 v4 smooth ...
        """
        if 'edges_raw' not in node.attributes:
            node.attributes['edges_raw'] = []

        # Extract all numbers from the line (skip attribute index range like [0:7])
        # Find where the actual data starts (after the closing bracket or after attribute name)
        data_start = line.find(']')
        if data_start == -1:
            data_start = line.find('.ed')
        if data_start != -1:
            data_str = line[data_start + 1:].rstrip(';').strip()
            numbers = re.findall(r'[-+]?\d+', data_str)

            # Parse triplets: (start_vertex, end_vertex, smooth_flag)
            for i in range(0, len(numbers) - 2, 3):
                start_v = int(numbers[i])
                end_v = int(numbers[i + 1])
                # smooth_flag = int(numbers[i + 2])  # Not needed for vertex extraction
                node.attributes['edges_raw'].append((start_v, end_v))

    def _parse_points_attr(self, node: MayaNode, line: str):
        """Parse point offsets (deformation deltas)"""
        if 'point_offsets' not in node.attributes:
            node.attributes['point_offsets'] = []

        if '-type "float3"' in line or "-type 'float3'" in line:
            type_idx = max(line.find('float3"'), line.find("float3'"))
            if type_idx != -1:
                values_str = line[type_idx + 7:].rstrip(';').strip()
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
                for i in range(0, len(numbers) - 2, 3):
                    node.attributes['point_offsets'].append(
                        [float(numbers[i]), float(numbers[i + 1]), float(numbers[i + 2])]
                    )

    def _parse_simple_attr(self, node: MayaNode, attr_name: str, line: str):
        """Parse simple numeric attribute"""
        # Get all numbers after the attribute name
        attr_idx = line.find(attr_name)
        if attr_idx != -1:
            values_str = line[attr_idx + len(attr_name):].rstrip(';').strip()
            # Remove any remaining flags
            values_str = re.sub(r'-\w+\s+"[^"]*"', '', values_str)
            values_str = re.sub(r"-\w+\s+'[^']*'", '', values_str)
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', values_str)
            if len(numbers) == 1:
                node.attributes[attr_name] = float(numbers[0])
            elif len(numbers) > 1:
                node.attributes[attr_name] = [float(n) for n in numbers]

    def _parse_connect_attr(self, line: str):
        """Parse connectAttr command: connectAttr "source.attr" "dest.attr";"""
        # Extract source and destination
        match = re.search(r'connectAttr\s+"([^"]+)"\s+"([^"]+)"', line)
        if not match:
            match = re.search(r"connectAttr\s+'([^']+)'\s+'([^']+)'", line)
        if match:
            source = match.group(1)
            dest = match.group(2)
            self.scene.connections.append((source, dest))

    def _parse_current_unit(self, line: str):
        """Parse currentUnit command for units"""
        # currentUnit -l centimeter -a degree -t film;
        linear_match = re.search(r'-l\s+(\w+)', line)
        if linear_match:
            self.scene.linear_unit = linear_match.group(1)

        angular_match = re.search(r'-a\s+(\w+)', line)
        if angular_match:
            self.scene.angular_unit = angular_match.group(1)

        time_match = re.search(r'-t\s+(\w+)', line)
        if time_match:
            time_unit = time_match.group(1)
            # Convert time unit to FPS
            fps_map = {
                'film': 24.0, 'pal': 25.0, 'ntsc': 30.0, 'show': 48.0,
                'palf': 50.0, 'ntscf': 60.0, 'game': 15.0,
                '2fps': 2.0, '3fps': 3.0, '4fps': 4.0, '5fps': 5.0,
                '6fps': 6.0, '8fps': 8.0, '10fps': 10.0, '12fps': 12.0,
                '16fps': 16.0, '20fps': 20.0, '40fps': 40.0,
                '75fps': 75.0, '80fps': 80.0, '100fps': 100.0,
                '120fps': 120.0, '125fps': 125.0, '150fps': 150.0,
                '200fps': 200.0, '240fps': 240.0, '250fps': 250.0,
                '300fps': 300.0, '375fps': 375.0, '400fps': 400.0,
                '500fps': 500.0, '600fps': 600.0, '750fps': 750.0,
                '1200fps': 1200.0, '1500fps': 1500.0, '2000fps': 2000.0,
                '3000fps': 3000.0, '6000fps': 6000.0,
                '23.976fps': 23.976, '29.97fps': 29.97, '29.97df': 29.97,
                '47.952fps': 47.952, '59.94fps': 59.94,
            }
            self.scene.fps = fps_map.get(time_unit, 24.0)

    def _parse_playback_options(self, line: str):
        """Parse playbackOptions for frame range"""
        # playbackOptions -min 1 -max 120 -ast 1 -aet 120;
        min_match = re.search(r'-min\s+([-+]?\d*\.?\d+)', line)
        max_match = re.search(r'-max\s+([-+]?\d*\.?\d+)', line)
        ast_match = re.search(r'-ast\s+([-+]?\d*\.?\d+)', line)
        aet_match = re.search(r'-aet\s+([-+]?\d*\.?\d+)', line)

        if min_match:
            self.scene.start_frame = float(min_match.group(1))
        if max_match:
            self.scene.end_frame = float(max_match.group(1))
        # Animation start/end take precedence if present
        if ast_match:
            self.scene.start_frame = float(ast_match.group(1))
        if aet_match:
            self.scene.end_frame = float(aet_match.group(1))

    def _build_hierarchy(self):
        """Build parent-child relationships between nodes"""
        for node in self.scene.nodes.values():
            if node.parent_name and node.parent_name in self.scene.nodes:
                parent = self.scene.nodes[node.parent_name]
                node._parent = parent
                parent.children.append(node)

    def _link_animations(self):
        """Link animation curves to their target nodes/attributes"""
        for source, dest in self.scene.connections:
            # Parse source (e.g., "pCube1_translateX.output" or "pCube1_translateX.o")
            source_parts = source.split('.')
            if len(source_parts) >= 1:
                curve_name = source_parts[0]
                if curve_name in self.scene.anim_curves:
                    curve = self.scene.anim_curves[curve_name]
                    # Parse destination (e.g., "pCube1.translateX" or "pCube1.tx")
                    dest_parts = dest.split('.')
                    if len(dest_parts) >= 2:
                        curve.target_node = dest_parts[0]
                        curve.target_attr = dest_parts[1]

        # Also link blend shapes to their target meshes
        self._link_blend_shapes()

    def _link_blend_shapes(self):
        """Link blendShape deformers to their target meshes via connections"""
        for source, dest in self.scene.connections:
            # Look for blendShape output connections:
            # blendShape1.outputGeometry[0] -> meshShape.inMesh
            # or: blendShape1.og[0] -> meshShape.i
            source_parts = source.split('.')
            if len(source_parts) >= 1:
                bs_name = source_parts[0]
                if bs_name in self.scene.blend_shapes:
                    # Check if destination is a mesh's input
                    dest_parts = dest.split('.')
                    if len(dest_parts) >= 1:
                        mesh_name = dest_parts[0]
                        # Check if the destination looks like a mesh input
                        attr = dest_parts[1] if len(dest_parts) > 1 else ''
                        if attr in ('inMesh', 'i', 'inputGeometry', 'ig'):
                            self.scene.blend_shapes[bs_name].connected_mesh = mesh_name


class MayaReader(BaseReader):
    """Maya ASCII reader implementing BaseReader interface

    Parses .ma files without requiring Maya installation.
    Supports:
    - Transform animation via animCurve nodes
    - Camera properties (focal length, apertures)
    - Mesh geometry (vertices, faces)
    - Baked vertex animation (point offsets)
    - Blend shape deformers with delta targets
    """

    def __init__(self, ma_file: str):
        """Initialize reader and parse Maya ASCII file

        Args:
            ma_file: Path to Maya ASCII (.ma) file
        """
        super().__init__(ma_file)
        parser = MayaASCIIParser()
        self.scene = parser.parse(str(self.file_path))
        self.fps = self.scene.fps

    def get_format_name(self) -> str:
        """Return human-readable format name"""
        return "Maya"

    def get_all_objects(self) -> List[MayaNode]:
        """Get all objects in the scene hierarchy (cached)"""
        if self._objects_cache is None:
            self._objects_cache = list(self.scene.nodes.values())
        return self._objects_cache

    def get_cameras(self) -> List[MayaNode]:
        """Get all camera objects in the scene"""
        return self.scene.get_cameras()

    def get_meshes(self) -> List[MayaNode]:
        """Get all mesh objects in the scene"""
        return self.scene.get_meshes()

    def get_transforms(self) -> List[MayaNode]:
        """Get all transform objects in the scene"""
        return self.scene.get_transforms()

    def get_parent_map(self) -> Dict[str, MayaNode]:
        """Build parent-child relationship map (cached)"""
        if self._parent_map_cache is None:
            self._parent_map_cache = {}
            for node in self.scene.nodes.values():
                if node._parent:
                    self._parent_map_cache[node.name] = node._parent
        return self._parent_map_cache

    def detect_frame_count(self, fps: int = 24) -> int:
        """Auto-detect frame count from file playback options

        Args:
            fps: Frames per second (used for fallback)

        Returns:
            int: Number of frames in the animation
        """
        frame_count = int(self.scene.end_frame - self.scene.start_frame + 1)
        return max(1, frame_count)

    def get_transform_at_time(self, obj: MayaNode, time_seconds: float,
                              maya_compat: bool = False) -> Tuple[List[float], List[float], List[float]]:
        """Get transform data at a specific time

        Samples animation curves if present, otherwise uses static values.

        Args:
            obj: MayaNode object (transform or shape with parent transform)
            time_seconds: Time in seconds to sample
            maya_compat: If True, use Maya-compatible rotation (already native)

        Returns:
            tuple: (translation, rotation, scale)
        """
        frame = time_seconds * self.fps

        # Find the transform node
        if obj.node_type in ('camera', 'mesh'):
            # Shape node - get parent transform
            transform_node = obj._parent
        else:
            transform_node = obj

        if not transform_node:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        # Get translation
        translation = self._get_animated_value(transform_node, 't', frame, [0.0, 0.0, 0.0])

        # Get rotation
        rotation = self._get_animated_value(transform_node, 'r', frame, [0.0, 0.0, 0.0])

        # Get scale
        scale = self._get_animated_value(transform_node, 's', frame, [1.0, 1.0, 1.0])

        return translation, rotation, scale

    def _get_animated_value(self, node: MayaNode, attr_base: str, frame: float,
                            default: List[float]) -> List[float]:
        """Get animated or static value for a transform attribute

        Args:
            node: Transform node
            attr_base: Attribute base name ('t', 'r', or 's')
            frame: Frame number to sample
            default: Default value if not found

        Returns:
            List of 3 floats [x, y, z]
        """
        result = list(default)

        # Map short names to full names for animation curve lookup
        attr_map = {
            't': ['translateX', 'translateY', 'translateZ', 'tx', 'ty', 'tz'],
            'r': ['rotateX', 'rotateY', 'rotateZ', 'rx', 'ry', 'rz'],
            's': ['scaleX', 'scaleY', 'scaleZ', 'sx', 'sy', 'sz'],
        }

        attrs = attr_map.get(attr_base, [])
        components = ['X', 'Y', 'Z', 'x', 'y', 'z']

        # Try to find animation curves for each component
        for i, component in enumerate(['X', 'Y', 'Z']):
            idx = i % 3
            curve = None

            # Try different attribute name formats
            for attr_name in [f'{attr_base}{component.lower()}',
                              f'{attr_base}{component}',
                              f'translate{component}' if attr_base == 't' else None,
                              f'rotate{component}' if attr_base == 'r' else None,
                              f'scale{component}' if attr_base == 's' else None]:
                if attr_name:
                    curve = self.scene.get_anim_curve_for_attr(node.name, attr_name)
                    if curve:
                        break

            if curve:
                result[idx] = curve.get_value_at_frame(frame)
            else:
                # Fall back to static attribute
                if attr_base in node.attributes:
                    static_val = node.attributes[attr_base]
                    if isinstance(static_val, list):
                        # Handle nested list case: [[x, y, z]] from float3 parsing
                        # vs flat list case: [x, y, z] from double3 parsing
                        if len(static_val) > 0 and isinstance(static_val[0], list):
                            # Nested list - flatten first element
                            if len(static_val[0]) > idx:
                                result[idx] = float(static_val[0][idx])
                        elif len(static_val) > idx:
                            # Flat list - access directly
                            result[idx] = float(static_val[idx])

        return result

    def get_mesh_data_at_time(self, mesh_obj: MayaNode, time_seconds: float) -> Dict[str, Any]:
        """Get mesh geometry data at a specific time

        Args:
            mesh_obj: MayaNode mesh object
            time_seconds: Time in seconds to sample

        Returns:
            dict: Mesh data with 'positions', 'indices', 'counts'
        """
        # Get base vertices
        vertices = mesh_obj.attributes.get('vertices', [])

        # Apply point offsets if present (baked vertex animation)
        offsets = mesh_obj.attributes.get('point_offsets', [])
        if offsets and len(offsets) == len(vertices):
            vertices = [
                [v[0] + o[0], v[1] + o[1], v[2] + o[2]]
                for v, o in zip(vertices, offsets)
            ]

        # Get face data using polyFaces format (edge-based)
        indices, counts = self._parse_face_data(mesh_obj, len(vertices))

        return {
            'positions': vertices,
            'indices': indices,
            'counts': counts
        }

    def _parse_face_data(self, mesh_obj: MayaNode, vertex_count: int) -> Tuple[List[int], List[int]]:
        """Parse Maya face data into indices and counts

        Handles Maya's polyFaces format which uses edge indices:
        - Positive edge index N: use start vertex of edge N
        - Negative edge index -N: use end vertex of edge (N-1)

        Args:
            mesh_obj: MayaNode with parsed mesh attributes
            vertex_count: Total number of vertices (for bounds checking)

        Returns:
            tuple: (indices, counts)
        """
        polyfaces_raw = mesh_obj.attributes.get('polyfaces_raw', [])
        edges_raw = mesh_obj.attributes.get('edges_raw', [])

        if not polyfaces_raw:
            return [], []

        indices = []
        counts = []

        for face_data in polyfaces_raw:
            face_vertex_count = face_data['count']
            edge_indices = face_data['edges']

            face_verts = []
            for edge_idx in edge_indices:
                if edge_idx >= 0:
                    # Positive: use start vertex of edge at this index
                    if edge_idx < len(edges_raw):
                        vert = edges_raw[edge_idx][0]  # start vertex
                        face_verts.append(vert)
                else:
                    # Negative: use end vertex of edge at (abs(edge_idx) - 1)
                    actual_edge_idx = abs(edge_idx) - 1
                    if 0 <= actual_edge_idx < len(edges_raw):
                        vert = edges_raw[actual_edge_idx][1]  # end vertex
                        face_verts.append(vert)

            if len(face_verts) == face_vertex_count:
                indices.extend(face_verts)
                counts.append(face_vertex_count)

        return indices, counts

    def get_camera_properties(self, cam_obj: MayaNode, time_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get camera properties at a specific time

        Args:
            cam_obj: MayaNode camera object
            time_seconds: Time in seconds (None for first sample)

        Returns:
            dict: Camera properties with 'focal_length', 'h_aperture', 'v_aperture'
        """
        # Get focal length (Maya attribute: .fl or .focalLength)
        focal_length = cam_obj.attributes.get('fl',
                       cam_obj.attributes.get('focalLength', 35.0))

        # Get apertures (Maya: .chs/.cvs for camera aperture, or .hfa/.vfa)
        # Maya stores in inches, convert to cm (Alembic convention)
        h_aperture_inch = cam_obj.attributes.get('hfa',
                          cam_obj.attributes.get('horizontalFilmAperture', 1.417))
        v_aperture_inch = cam_obj.attributes.get('vfa',
                          cam_obj.attributes.get('verticalFilmAperture', 0.945))

        # Convert inches to cm
        h_aperture = h_aperture_inch * 2.54
        v_aperture = v_aperture_inch * 2.54

        return {
            'focal_length': float(focal_length),
            'h_aperture': float(h_aperture),
            'v_aperture': float(v_aperture)
        }

    def _get_full_path(self, obj: MayaNode) -> str:
        """Get full hierarchy path for a Maya node

        Args:
            obj: MayaNode object

        Returns:
            str: Full path like "/World/Camera/CameraShape"
        """
        return obj.getFullName()

    def _uses_shape_transform_pattern(self):
        """Maya uses shape/transform pairs like Alembic.

        Camera and mesh nodes are shape nodes under transform nodes.
        The transform name is the meaningful display name.
        """
        return True

    def _is_organizational_group(self, obj: MayaNode) -> bool:
        """Check if transform is just an organizational container

        Args:
            obj: MayaNode object

        Returns:
            bool: True if object is organizational only
        """
        if obj.node_type != 'transform':
            return False

        # Check if it has any animation
        has_animation = False
        for attr in ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz',
                     'translateX', 'translateY', 'translateZ',
                     'rotateX', 'rotateY', 'rotateZ',
                     'scaleX', 'scaleY', 'scaleZ']:
            if self.scene.get_anim_curve_for_attr(obj.name, attr):
                has_animation = True
                break

        # If no animation and has children but no direct shapes, it's organizational
        if not has_animation:
            has_shape_child = any(
                child.node_type in ('camera', 'mesh')
                for child in obj.children
            )
            has_children = len(obj.children) > 0

            if has_children and not has_shape_child:
                return True

        return False

    def get_blend_shape_for_mesh(self, mesh_name: str) -> Optional['BlendShapeDeformer']:
        """Get blend shape deformer data for a mesh

        Args:
            mesh_name: Name of the mesh to find blend shapes for

        Returns:
            BlendShapeDeformer if mesh has blend shapes, None otherwise
        """
        from core.scene_data import (
            BlendShapeDeformer, BlendShapeChannel, BlendShapeTarget, BlendShapeWeightKey
        )

        # Find blend shape connected to this mesh
        for bs_data in self.scene.blend_shapes.values():
            if bs_data.connected_mesh == mesh_name:
                channels = []

                # Process each target group
                for target_idx, weight_items in bs_data.targets.items():
                    # Get target name from alias or generate one
                    target_name = bs_data.weight_aliases.get(target_idx, f"target_{target_idx}")

                    # Usually there's one weight item per target (at index 6000 = weight 1.0)
                    for weight_idx, data in weight_items.items():
                        deltas = data.get('deltas', [])
                        components = data.get('components', [])

                        if not deltas:
                            continue

                        # If no components specified, assume sequential indices
                        if not components:
                            components = list(range(len(deltas)))

                        # Ensure we have matching counts
                        if len(components) != len(deltas):
                            # Adjust to minimum of both
                            min_len = min(len(components), len(deltas))
                            components = components[:min_len]
                            deltas = deltas[:min_len]

                        # Calculate full weight from index: weight = (index / 1000) - 5
                        full_weight = (weight_idx / 1000.0) - 5.0
                        if full_weight <= 0:
                            full_weight = 1.0

                        target = BlendShapeTarget(
                            name=target_name,
                            vertex_indices=components,
                            deltas=deltas,
                            full_weight=full_weight
                        )

                        # Check for weight animation
                        weight_animation = None
                        weight_attr = f"w[{target_idx}]"
                        # Also check alias name
                        for attr_name in [weight_attr, target_name]:
                            curve = self.scene.get_anim_curve_for_attr(bs_data.name, attr_name)
                            if curve and curve.keyframes:
                                weight_animation = [
                                    BlendShapeWeightKey(frame=int(kf[0]), weight=kf[1])
                                    for kf in curve.keyframes
                                ]
                                break

                        channel = BlendShapeChannel(
                            name=target_name,
                            targets=[target],
                            weight_animation=weight_animation,
                            default_weight=0.0
                        )
                        channels.append(channel)

                if channels:
                    return BlendShapeDeformer(
                        name=bs_data.name,
                        channels=channels,
                        base_mesh_name=mesh_name
                    )

        return None
