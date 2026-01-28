#!/usr/bin/env python3
"""
After Effects Exporter Module
Exports SceneData to After Effects JSX + OBJ format with vertex animation filtering

v2.5.0 - Refactored to use SceneData instead of reader objects.
         Now format-agnostic - works with any input format!
         USD to After Effects export is now supported.
"""

import os
import re
from pathlib import Path

from .base_exporter import BaseExporter
from core.scene_data import SceneData, AnimationType


class AfterEffectsExporter(BaseExporter):
    """After Effects JSX + OBJ exporter with vertex animation filtering

    After Effects can only handle transform animation (position, rotation, scale).
    Meshes with vertex animation (deformation) must be skipped entirely.

    v2.5.0: Now works with SceneData instead of reader objects - format-agnostic.
            Supports Alembic and USD input files.

    Exports:
    - Single JSX file with scene setup and animation
    - Multiple OBJ files (one per mesh, frame 1 only)
    """

    def get_format_name(self):
        return "After Effects JSX + OBJ"

    def get_file_extension(self):
        return "jsx"

    def export(self, scene_data: SceneData, output_path, shot_name):
        """Export to After Effects JSX format

        Args:
            scene_data: SceneData instance with pre-extracted animation and geometry
            output_path: Output directory path
            shot_name: Shot name for composition and file naming

        Returns:
            dict: Export results with keys:
                - 'success': bool
                - 'jsx_file': Path to created JSX file
                - 'obj_files': List of created OBJ file paths
                - 'skipped_meshes': List of skipped mesh names (vertex animated)
                - 'message': Status message
        """
        try:
            output_dir = self.validate_output_path(output_path)

            # Extract info from SceneData
            fps = scene_data.metadata.fps
            frame_count = scene_data.metadata.frame_count
            start_frame = scene_data.metadata.start_frame
            comp_width = scene_data.metadata.width
            comp_height = scene_data.metadata.height
            footage_path = scene_data.metadata.footage_path
            source_filename = Path(scene_data.metadata.source_file_path).name

            # Get skipped meshes (vertex animated)
            skipped_meshes = scene_data.animation_categories.vertex_animated
            if skipped_meshes:
                self.log(f"Skipping {len(skipped_meshes)} meshes with vertex animation:")
                for mesh_name in skipped_meshes:
                    self.log(f"  - {mesh_name}")

            duration = frame_count / fps

            # Generate JSX
            jsx_lines = []

            # Header
            jsx_lines.extend(self._generate_header(shot_name, source_filename))

            # Helper functions
            jsx_lines.extend(self._generate_helper_functions())

            # Main function start
            jsx_lines.append("function SceneImportFunction() {")
            jsx_lines.append("")
            jsx_lines.append("app.exitAfterLaunchAndEval = false;")
            jsx_lines.append("")
            jsx_lines.append("app.beginUndoGroup('Scene Import');")
            jsx_lines.append("")

            # Create composition
            jsx_lines.append(f"var comp = app.project.items.addComp('{shot_name}', {comp_width}, {comp_height}, 1.0, {duration}, {fps});")
            jsx_lines.append(f"comp.displayStartFrame = {start_frame};")
            jsx_lines.append("")

            # Import footage if available
            if footage_path:
                jsx_lines.extend(self._generate_footage_import(footage_path, shot_name))

            # Process cameras
            for camera in scene_data.cameras:
                cam_name = camera.parent_name if camera.parent_name else camera.name
                self.log(f"Processing camera: {cam_name}")
                camera_jsx = self._process_camera(camera, cam_name, frame_count, fps, comp_width, comp_height)
                jsx_lines.extend(camera_jsx)
                jsx_lines.append("")

            # Process meshes (skip vertex-animated ones)
            for mesh in scene_data.meshes:
                mesh_name = mesh.parent_name if mesh.parent_name else mesh.name

                # Skip if mesh has vertex animation
                if mesh.animation_type == AnimationType.VERTEX_ANIMATED:
                    self.log(f"Skipping vertex-animated mesh: {mesh_name}")
                    continue

                self.log(f"Processing geometry: {mesh_name}")
                geom_jsx = self._process_geometry(mesh, mesh_name, frame_count, fps, output_dir, comp_width, comp_height)
                jsx_lines.extend(geom_jsx)
                jsx_lines.append("")

            # Process transforms (locators/nulls)
            for transform in scene_data.transforms:
                self.log(f"Processing locator: {transform.name}")
                loc_jsx = self._process_locator(transform, transform.name, frame_count, fps, comp_width, comp_height)
                jsx_lines.extend(loc_jsx)
                jsx_lines.append("")

            # Footer
            jsx_lines.extend(self._generate_footer())

            # Write JSX file
            jsx_file = output_dir / f"{shot_name}.jsx"
            with open(jsx_file, 'w') as f:
                f.write("\n".join(jsx_lines))

            # Collect OBJ files
            obj_files = list(output_dir.glob('*.obj'))

            self.log(f"\n✓ JSX written to: {jsx_file}")
            self.log(f"✓ Frame range: {start_frame} - {start_frame + frame_count - 1} ({frame_count} frames)")
            self.log(f"✓ Duration: {duration}s @ {fps} fps")
            self.log(f"✓ OBJ files exported: {len(obj_files)}")

            return {
                'success': True,
                'jsx_file': str(jsx_file),
                'obj_files': [str(f) for f in obj_files],
                'skipped_meshes': skipped_meshes,
                'message': f"Exported {len(obj_files)} OBJ files, skipped {len(skipped_meshes)} vertex-animated meshes",
                'files': [str(jsx_file)] + [str(f) for f in obj_files]
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

    def _generate_header(self, shot_name, source_filename):
        """Generate JSX file header"""
        lines = []
        lines.append("// Auto-generated JSX from scene data")
        lines.append(f"// Exported from: {source_filename}")
        lines.append("// Y-up coordinate system, 1:1 scale")
        lines.append("")
        lines.append("app.activate();")
        lines.append("")
        return lines

    def _generate_helper_functions(self):
        """Generate JSX helper functions"""
        lines = []
        lines.append("function findComp(nm) {")
        lines.append("    var i, n, prjitm;")
        lines.append("")
        lines.append("    prjitm = app.project.items;")
        lines.append("    n = prjitm.length;")
        lines.append("    for (i = 1; i <= n; i++) {")
        lines.append("        if (prjitm[i].name == nm)")
        lines.append("            return prjitm[i];")
        lines.append("    }")
        lines.append("    return null;")
        lines.append("}")
        lines.append("")
        lines.append("function firstComp() {")
        lines.append("    var i, n, prjitm;")
        lines.append("")
        lines.append("    if (app.project.activeItem.typeName == \"Composition\")")
        lines.append("        return app.project.activeItem;")
        lines.append("")
        lines.append("    prjitm = app.project.items;")
        lines.append("    n = prjitm.length;")
        lines.append("    for (i = 1; i <= n; i++) {")
        lines.append("        if (prjitm[i].typeName == \"Composition\")")
        lines.append("            return prjitm[i];")
        lines.append("    }")
        lines.append("    return null;")
        lines.append("}")
        lines.append("")
        lines.append("function firstSelectedComp(items) {")
        lines.append("    var i, itm, subitm;")
        lines.append("")
        lines.append("    for (i = 1; i <= items.length; i++) {")
        lines.append("        itm = items[i];")
        lines.append("        if (itm instanceof CompItem && itm.selected)")
        lines.append("            return itm;")
        lines.append("        if (itm instanceof FolderItem) {")
        lines.append("            subitm = firstSelectedComp(itm.items);")
        lines.append("            if (subitm)")
        lines.append("                return subitm;")
        lines.append("        }")
        lines.append("    };")
        lines.append("    return null;")
        lines.append("}")
        lines.append("")
        lines.append("function deselectAll(items) {")
        lines.append("    var i, itm;")
        lines.append("")
        lines.append("    for (i = 1; i <= items.length; i++) {")
        lines.append("        itm = items[i];")
        lines.append("        if (itm instanceof FolderItem)")
        lines.append("            deselectAll(itm.items);")
        lines.append("        itm.selected = false;")
        lines.append("    };")
        lines.append("}")
        lines.append("")
        return lines

    def _generate_footage_import(self, footage_path, shot_name):
        """Generate JSX code for footage import"""
        lines = []
        lines.append("// Import footage file from scene metadata")
        lines.append(f"var footagePath = '{footage_path.replace(chr(92), '/')}';")
        lines.append("var footageFile = new File(footagePath);")
        lines.append("if (footageFile.exists) {")
        lines.append("    var footageImportOptions = new ImportOptions(footageFile);")
        lines.append("    var footageItem = app.project.importFile(footageImportOptions);")
        lines.append("    footageItem.selected = false;")
        lines.append(f"    footageItem.name = '{shot_name}_Footage';")
        lines.append("    // Add footage to composition as background layer")
        lines.append("    var footageLayer = comp.layers.add(footageItem);")
        lines.append(f"    footageLayer.name = '{shot_name}_Footage';")
        lines.append("    footageLayer.moveToEnd();  // Move to bottom of layer stack")
        lines.append("} else {")
        lines.append(f"    alert('Warning: Footage file not found at path: ' + footagePath);")
        lines.append("}")
        lines.append("")
        return lines

    def _generate_footer(self):
        """Generate JSX file footer"""
        lines = []
        lines.append("// Make comp the current open composition")
        lines.append("comp.selected = true;")
        lines.append("deselectAll(app.project.items);")
        lines.append("comp.selected = true;")
        lines.append("comp.openInViewer();")
        lines.append("")
        lines.append("app.endUndoGroup();")
        lines.append("alert('Scene import complete!');")
        lines.append("")
        lines.append("} // End SceneImportFunction")
        lines.append("")
        lines.append("SceneImportFunction();")
        return lines

    def _is_animated(self, keyframes):
        """Check if keyframes have any variation (not static)"""
        if len(keyframes) <= 1:
            return False

        first = keyframes[0]
        for kf in keyframes[1:]:
            # Check position
            if (abs(kf.position[0] - first.position[0]) > 0.0001 or
                abs(kf.position[1] - first.position[1]) > 0.0001 or
                abs(kf.position[2] - first.position[2]) > 0.0001):
                return True
            # Check rotation (using AE rotation)
            if (abs(kf.rotation_ae[0] - first.rotation_ae[0]) > 0.0001 or
                abs(kf.rotation_ae[1] - first.rotation_ae[1]) > 0.0001 or
                abs(kf.rotation_ae[2] - first.rotation_ae[2]) > 0.0001):
                return True
            # Check scale
            if (abs(kf.scale[0] - first.scale[0]) > 0.0001 or
                abs(kf.scale[1] - first.scale[1]) > 0.0001 or
                abs(kf.scale[2] - first.scale[2]) > 0.0001):
                return True

        return False

    def _process_camera(self, camera, name, frame_count, fps, comp_width, comp_height):
        """Process camera and return JSX with array-based animation"""
        jsx = []

        # Get camera properties from SceneData
        focal_length = camera.properties.focal_length
        h_aperture = camera.properties.h_aperture * 10  # cm to mm

        # Calculate AE zoom value
        ae_zoom = focal_length * comp_width / h_aperture

        layer_var = f"camera_{self._sanitize_var_name(name)}"
        layer_name = self._escape_layer_name(name)

        # Log animation info for debugging
        if camera.keyframes:
            first_kf = camera.keyframes[0]
            last_kf = camera.keyframes[-1] if len(camera.keyframes) > 1 else first_kf
            self.log(f"  Camera {name} frame 1: pos={first_kf.position}, rot={first_kf.rotation_ae}")
            if len(camera.keyframes) > 1:
                self.log(f"  Camera {name} frame {last_kf.frame}: pos={last_kf.position}, rot={last_kf.rotation_ae}")
                pos_changed = first_kf.position != last_kf.position
                rot_changed = first_kf.rotation_ae != last_kf.rotation_ae
                self.log(f"    Position animated: {pos_changed}, Rotation animated: {rot_changed}")

        # Create camera
        jsx.append(f"var {layer_var} = comp.layers.addCamera('{layer_name}', [0, 0]);")
        jsx.append(f"{layer_var}.autoOrient = AutoOrientType.NO_AUTO_ORIENT;")

        # Generate array definitions
        jsx.append(f"var timesArray = new Array();")
        jsx.append(f"var posArray = new Array();")
        jsx.append(f"var rotXArray = new Array();")
        jsx.append(f"var rotYArray = new Array();")
        jsx.append(f"var rotZArray = new Array();")

        # Coordinate system transformation (Y-up to AE composition space)
        comp_center_x = comp_width / 2
        comp_center_y = comp_height / 2

        for kf in camera.keyframes:
            # AE time: frame 1 = time 0, frame 2 = time 1/fps, etc.
            time_seconds = (kf.frame - 1) / fps

            # Transform coordinates for AE
            x_ae = kf.position[0] * 10 + comp_center_x
            y_ae = -kf.position[1] * 10 + comp_center_y
            z_ae = -kf.position[2] * 10

            jsx.append(f"timesArray.push({time_seconds:.10f});")
            jsx.append(f"posArray.push([{x_ae:.10f}, {y_ae:.10f}, {z_ae:.10f}]);")
            jsx.append(f"rotXArray.push({-kf.rotation_ae[0]:.10f});")
            jsx.append(f"rotYArray.push({kf.rotation_ae[1]:.10f});")
            jsx.append(f"rotZArray.push({kf.rotation_ae[2]:.10f});")

        # Apply arrays to properties
        jsx.append(f"{layer_var}.position.setValuesAtTimes(timesArray, posArray);")
        jsx.append(f"{layer_var}.rotationX.setValuesAtTimes(timesArray, rotXArray);")
        jsx.append(f"{layer_var}.rotationY.setValuesAtTimes(timesArray, rotYArray);")
        jsx.append(f"{layer_var}.rotationZ.setValuesAtTimes(timesArray, rotZArray);")

        # Set camera zoom
        jsx.append(f"{layer_var}.zoom.setValue({ae_zoom:.10f});")

        return jsx

    def _process_geometry(self, mesh, name, frame_count, fps, output_dir, comp_width, comp_height):
        """Process geometry mesh with OBJ export and transform"""
        jsx = []
        layer_var = f"mesh_{self._sanitize_var_name(name)}"
        layer_name = self._escape_layer_name(name)

        # Export OBJ from SceneData geometry - use sanitized name for filename
        safe_filename = self._sanitize_var_name(name)
        obj_filename = f"{safe_filename}.obj"
        obj_path = os.path.join(output_dir, obj_filename)
        self._export_mesh_to_obj(mesh, obj_path)

        # Generate OBJ import code
        jsx.append(f"var importOptions = new ImportOptions();")
        jsx.append(f"importOptions.file = File(new File($.fileName).parent.fsName + '/{obj_filename}');")
        jsx.append(f"var objFootage = app.project.importFile(importOptions);")
        jsx.append(f"objFootage.selected = false;")
        jsx.append(f"app.beginSuppressDialogs();")
        jsx.append(f"var {layer_var} = comp.layers.add(objFootage);")
        jsx.append(f"{layer_var}.name = '{layer_name}';")
        jsx.append(f"app.endSuppressDialogs(true);")

        # Set anchor point to [0,0,0]
        jsx.append(f"{layer_var}.anchorPoint.setValue([0, 0, 0]);")

        keyframes = mesh.keyframes

        # Check if animated or static
        if keyframes and self._is_animated(keyframes):
            # Animated - use setValuesAtTimes
            jsx.append(f"var timesArray = new Array();")
            jsx.append(f"var posArray = new Array();")
            jsx.append(f"var rotXArray = new Array();")
            jsx.append(f"var rotYArray = new Array();")
            jsx.append(f"var rotZArray = new Array();")
            jsx.append(f"var scaleArray = new Array();")

            comp_center_x = comp_width / 2
            comp_center_y = comp_height / 2

            for kf in keyframes:
                time_seconds = (kf.frame - 1) / fps

                x_ae = kf.position[0] * 10 + comp_center_x
                y_ae = -kf.position[1] * 10 + comp_center_y
                z_ae = -kf.position[2] * 10

                # AE scale is in percent, compensate for world-scale OBJ vertices
                scale = [s * 2 for s in kf.scale]

                jsx.append(f"timesArray.push({time_seconds:.10f});")
                jsx.append(f"posArray.push([{x_ae:.10f}, {y_ae:.10f}, {z_ae:.10f}]);")
                jsx.append(f"rotXArray.push({-kf.rotation_ae[0]:.10f});")
                jsx.append(f"rotYArray.push({kf.rotation_ae[1]:.10f});")
                jsx.append(f"rotZArray.push({kf.rotation_ae[2]:.10f});")
                jsx.append(f"scaleArray.push([{scale[0]:.10f}, {scale[1]:.10f}, {scale[2]:.10f}]);")

            jsx.append(f"{layer_var}.position.setValuesAtTimes(timesArray, posArray);")
            jsx.append(f"{layer_var}.rotationX.setValuesAtTimes(timesArray, rotXArray);")
            jsx.append(f"{layer_var}.rotationY.setValuesAtTimes(timesArray, rotYArray);")
            jsx.append(f"{layer_var}.rotationZ.setValuesAtTimes(timesArray, rotZArray);")
            jsx.append(f"{layer_var}.scale.setValuesAtTimes(timesArray, scaleArray);")
        elif keyframes:
            # Static - use setValue with first frame values
            kf = keyframes[0]

            comp_center_x = comp_width / 2
            comp_center_y = comp_height / 2

            x_ae = kf.position[0] * 10 + comp_center_x
            y_ae = -kf.position[1] * 10 + comp_center_y
            z_ae = -kf.position[2] * 10

            # AE scale is in percent, compensate for world-scale OBJ vertices
            scale = [s * 2 for s in kf.scale]

            jsx.append(f"{layer_var}.scale.setValue([{scale[0]:.10f}, {scale[1]:.10f}, {scale[2]:.10f}]);")
            jsx.append(f"{layer_var}.position.setValue([{x_ae:.10f}, {y_ae:.10f}, {z_ae:.10f}]);")
            jsx.append(f"{layer_var}.rotationX.setValue({-kf.rotation_ae[0]:.10f});")
            jsx.append(f"{layer_var}.rotationY.setValue({kf.rotation_ae[1]:.10f});")
            jsx.append(f"{layer_var}.rotationZ.setValue({kf.rotation_ae[2]:.10f});")

        jsx.append("")
        return jsx

    def _process_locator(self, transform, name, frame_count, fps, comp_width, comp_height):
        """Process locator/transform as 3D null"""
        jsx = []
        layer_var = f"locator_{self._sanitize_var_name(name)}"
        layer_name = self._escape_layer_name(name)

        jsx.append(f"var {layer_var} = comp.layers.addNull();")
        jsx.append(f"{layer_var}.name = '{layer_name}';")
        jsx.append(f"{layer_var}.threeDLayer = true;")
        jsx.append(f"{layer_var}.shy = true;")
        jsx.append(f"{layer_var}.label = 13;")  # 13 = yellow in AE

        keyframes = transform.keyframes

        # Check if animated or static
        if keyframes and self._is_animated(keyframes):
            # Animated
            jsx.append(f"var timesArray = new Array();")
            jsx.append(f"var posArray = new Array();")
            jsx.append(f"var rotXArray = new Array();")
            jsx.append(f"var rotYArray = new Array();")
            jsx.append(f"var rotZArray = new Array();")

            comp_center_x = comp_width / 2
            comp_center_y = comp_height / 2

            for kf in keyframes:
                time_seconds = (kf.frame - 1) / fps

                x_ae = kf.position[0] * 10 + comp_center_x
                y_ae = -kf.position[1] * 10 + comp_center_y
                z_ae = -kf.position[2] * 10

                jsx.append(f"timesArray.push({time_seconds:.10f});")
                jsx.append(f"posArray.push([{x_ae:.10f}, {y_ae:.10f}, {z_ae:.10f}]);")
                jsx.append(f"rotXArray.push({-kf.rotation_ae[0]:.10f});")
                jsx.append(f"rotYArray.push({kf.rotation_ae[1]:.10f});")
                jsx.append(f"rotZArray.push({kf.rotation_ae[2]:.10f});")

            jsx.append(f"{layer_var}.position.setValuesAtTimes(timesArray, posArray);")
            jsx.append(f"{layer_var}.rotationX.setValuesAtTimes(timesArray, rotXArray);")
            jsx.append(f"{layer_var}.rotationY.setValuesAtTimes(timesArray, rotYArray);")
            jsx.append(f"{layer_var}.rotationZ.setValuesAtTimes(timesArray, rotZArray);")
        elif keyframes:
            # Static
            kf = keyframes[0]

            comp_center_x = comp_width / 2
            comp_center_y = comp_height / 2

            x_ae = kf.position[0] * 10 + comp_center_x
            y_ae = -kf.position[1] * 10 + comp_center_y
            z_ae = -kf.position[2] * 10

            jsx.append(f"{layer_var}.position.setValue([{x_ae:.10f}, {y_ae:.10f}, {z_ae:.10f}]);")
            jsx.append(f"{layer_var}.property('Anchor Point').setValue([0.00, 0.00, 0.00]);")
            jsx.append(f"{layer_var}.scale.setValue([{kf.scale[0] * 2:.10f}, {kf.scale[1] * 2:.10f}, {kf.scale[2] * 2:.10f}]);")

        jsx.append("")
        return jsx

    def _export_mesh_to_obj(self, mesh, obj_path):
        """Export a mesh to OBJ file using SceneData geometry"""
        try:
            geometry = mesh.geometry

            with open(obj_path, 'w') as f:
                f.write(f"# Exported from scene data\n")
                f.write(f"# Object: {mesh.name}\n\n")

                # Write vertices
                for v in geometry.positions:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                # Write faces
                f.write("\n")
                idx = 0
                for count in geometry.counts:
                    # OBJ indices are 1-based
                    face_verts = [str(geometry.indices[idx + i] + 1) for i in range(count)]
                    f.write(f"f {' '.join(face_verts)}\n")
                    idx += count

            return True
        except Exception as e:
            self.log(f"Warning: Could not export mesh to OBJ: {e}")
            return False

    def _sanitize_var_name(self, name):
        """Sanitize a name for use as a JavaScript variable name

        JavaScript variable names can only contain letters, digits, underscores, and $.
        They cannot start with a digit.
        """
        # Replace all non-alphanumeric characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a digit
        if sanitized and sanitized[0].isdigit():
            sanitized = f"obj_{sanitized}"
        return sanitized or "unnamed"

    def _escape_layer_name(self, name):
        """Escape a name for use in a JavaScript string literal

        Escapes quotes and backslashes for safe use in JSX strings.
        """
        # Escape backslashes first, then single quotes
        escaped = name.replace('\\', '\\\\').replace("'", "\\'")
        return escaped
