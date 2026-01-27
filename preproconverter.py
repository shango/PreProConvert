#!/usr/bin/env python3
"""
PreProConverter - Multi-Format Scene Converter
Coordinates multi-format export using modular readers and exporters
Supports Alembic (.abc) and USD (.usd, .usda, .usdc) input

v2.6.2 - SceneData architecture: readers extract all data into format-agnostic
structure, exporters work only with SceneData (no direct reader access).
"""

from pathlib import Path

# Import readers module
from readers import create_reader, get_file_type, AlembicReader

# Import exporters
from exporters.ae_exporter import AfterEffectsExporter
from exporters.usd_exporter import USDExporter
from exporters.maya_ma_exporter import MayaMAExporter
from exporters.fbx_exporter import FBXExporter


class PreProConverter:
    """Multi-format scene converter (orchestrator/facade)

    This class coordinates the conversion process:
    1. Read input file ONCE (via readers module - supports Alembic and USD)
    2. Analyze animation types ONCE (via AnimationDetector)
    3. Export to selected formats (via format-specific exporters)

    Input formats supported:
    - Alembic (.abc)
    - USD (.usd, .usda, .usdc)

    Output formats supported:
    - After Effects JSX + OBJ (skips vertex-animated meshes)
    - USD .usdc (with vertex animation support)
    - Maya MA .ma (native Maya ASCII with source file references)
    - FBX .fbx (for Unreal Engine)
    """

    def __init__(self, progress_callback=None):
        """Initialize converter

        Args:
            progress_callback: Optional function to call for progress updates
                              Signature: callback(message: str, progress: int = None) -> None
        """
        self.progress_callback = progress_callback

    def log(self, message, progress=None):
        """Send progress updates to callback

        Args:
            message: Progress message
            progress: Optional progress percentage (0-100)
        """
        if self.progress_callback:
            self.progress_callback(message, progress)
        print(message)

    def convert_multi_format(self, input_file, output_dir, shot_name, fps=24, frame_count=None,
                            export_ae=True, export_usd=True, export_maya_ma=True, export_fbx=True):
        """Convert scene file to multiple formats

        This is the main entry point for v2.5.0 multi-format export.
        Uses SceneData architecture - all scene data extracted once, passed to all exporters.

        Args:
            input_file: Path to input scene file (.abc, .usd, .usda, .usdc)
            output_dir: Output directory for all formats
            shot_name: Shot name for file/folder naming
            fps: Frames per second (default: 24)
            frame_count: Number of frames (None = auto-detect)
            export_ae: Export to After Effects JSX + OBJ
            export_usd: Export to USD (.usdc)
            export_maya_ma: Export to Maya MA (.ma)
            export_fbx: Export to FBX (.fbx) for Unreal Engine

        Returns:
            dict: Results with keys:
                - 'success': bool
                - 'ae': AE export results (if export_ae=True)
                - 'usd': USD export results (if export_usd=True)
                - 'maya_ma': Maya MA export results (if export_maya_ma=True)
                - 'fbx': FBX export results (if export_fbx=True)
                - 'message': Summary message
        """
        try:
            # Detect file type for logging
            input_path = Path(input_file)
            file_type = get_file_type(str(input_path))
            format_name = "Alembic" if file_type == 'alembic' else "USD"

            self.log(f"\n{'='*60}", progress=0)
            self.log(f"PreProConverter v2.6.2 - VFX-Experts")
            self.log(f"{'='*60}")
            self.log(f"Input: {input_file} ({format_name})")
            self.log(f"Output: {output_dir}")
            self.log(f"Shot: {shot_name}")
            self.log(f"{'='*60}\n")

            results = {
                'success': False,
                'message': ''
            }

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Calculate number of selected exports for progress distribution
            selected_exports = sum([export_ae, export_usd, export_maya_ma, export_fbx])
            export_progress_per_format = 65 // max(selected_exports, 1)  # 30-95% for exports

            # Step 1: Read input file ONCE (auto-detect format)
            self.log(f"Step 1/4: Reading {format_name} file...", progress=5)
            reader = create_reader(input_file)

            # Auto-detect frame count if not provided
            if frame_count is None:
                frame_count = reader.detect_frame_count(fps)
                self.log(f"  Auto-detected frame count: {frame_count}")
            else:
                self.log(f"  Frame count: {frame_count}")

            self.log(f"  FPS: {fps}", progress=10)

            # Step 2: Extract all scene data ONCE (includes animation analysis)
            self.log("\nStep 2/4: Extracting scene data and analyzing animation...", progress=15)
            scene_data = reader.extract_scene_data(fps, frame_count)

            # Log animation summary
            categories = scene_data.animation_categories
            self.log(f"\nAnimation Analysis:")
            self.log(f"  - Vertex Animated: {len(categories.vertex_animated)} meshes")
            self.log(f"  - Transform Only: {len(categories.transform_only)} meshes")
            self.log(f"  - Static: {len(categories.static)} meshes")
            self.log(f"  - Cameras: {len(scene_data.cameras)}")
            self.log(f"  - Transforms/Locators: {len(scene_data.transforms)}")

            if categories.vertex_animated:
                self.log(f"\n  Vertex Animated Meshes:")
                for name in categories.vertex_animated:
                    self.log(f"    - {name}")

            # Step 3: Export to selected formats
            self.log("\nStep 3/4: Exporting to selected formats...", progress=30)

            current_export = 0

            # Export to After Effects
            if export_ae:
                current_export += 1
                progress = 30 + (current_export * export_progress_per_format)
                self.log(f"\n--- After Effects Export ---", progress=progress - export_progress_per_format + 5)
                ae_dir = output_path / f"{shot_name}_ae"
                exporter = AfterEffectsExporter(self.progress_callback)
                results['ae'] = exporter.export(scene_data, ae_dir, shot_name)
                self.log(f"After Effects export complete", progress=progress)

            # Export to USD
            if export_usd:
                current_export += 1
                progress = 30 + (current_export * export_progress_per_format)
                self.log(f"\n--- USD Export ---", progress=progress - export_progress_per_format + 5)
                usd_dir = output_path / f"{shot_name}_usd"
                exporter = USDExporter(self.progress_callback)
                results['usd'] = exporter.export(scene_data, usd_dir, shot_name)
                self.log(f"USD export complete", progress=progress)

            # Export to Maya MA
            if export_maya_ma:
                current_export += 1
                progress = 30 + (current_export * export_progress_per_format)
                self.log(f"\n--- Maya MA Export ---", progress=progress - export_progress_per_format + 5)
                maya_dir = output_path / f"{shot_name}_maya"
                exporter = MayaMAExporter(self.progress_callback)
                results['maya_ma'] = exporter.export(scene_data, maya_dir, shot_name)
                self.log(f"Maya MA export complete", progress=progress)

            # Export to FBX (for Unreal Engine)
            if export_fbx:
                current_export += 1
                progress = 30 + (current_export * export_progress_per_format)
                self.log(f"\n--- FBX Export (Unreal Engine) ---", progress=progress - export_progress_per_format + 5)
                fbx_dir = output_path / f"{shot_name}_fbx"
                exporter = FBXExporter(self.progress_callback)
                results['fbx'] = exporter.export(scene_data, fbx_dir, shot_name)
                self.log(f"FBX export complete", progress=progress)

            # Step 4: Summary
            self.log(f"\n{'='*60}", progress=95)
            self.log(f"Export Complete!")
            self.log(f"{'='*60}")

            success_count = sum(1 for key in ['ae', 'usd', 'maya_ma', 'fbx']
                              if key in results and results[key].get('success', False))

            results['success'] = success_count > 0
            results['message'] = f"Successfully exported {success_count} format(s)"

            self.log(f"\nSummary:")
            if 'ae' in results:
                status = "✓" if results['ae'].get('success') else "✗"
                self.log(f"  {status} After Effects: {results['ae'].get('message', 'N/A')}")
            if 'usd' in results:
                status = "✓" if results['usd'].get('success') else "✗"
                self.log(f"  {status} USD: {results['usd'].get('message', 'N/A')}")
            if 'maya_ma' in results:
                status = "✓" if results['maya_ma'].get('success') else "✗"
                self.log(f"  {status} Maya MA: {results['maya_ma'].get('message', 'N/A')}")
            if 'fbx' in results:
                status = "✓" if results['fbx'].get('success') else "✗"
                self.log(f"  {status} FBX: {results['fbx'].get('message', 'N/A')}")

            self.log(f"\n{results['message']}")
            self.log(f"{'='*60}\n")

            return results

        except Exception as e:
            self.log(f"\nERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return {
                'success': False,
                'message': f"Conversion failed: {str(e)}"
            }

    def convert(self, abc_file, jsx_file, fps=24, frame_count=120, comp_name="AlembicScene"):
        """Legacy conversion method for backward compatibility with v2.0.0

        This method maintains the original API for existing code/scripts.
        It internally calls convert_multi_format() with AE-only settings.

        Args:
            abc_file: Path to input scene file (.abc, .usd, .usda, .usdc)
            jsx_file: Path to output JSX file
            fps: Frames per second (default: 24)
            frame_count: Number of frames (default: 120)
            comp_name: Composition name (default: "AlembicScene")

        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        self.log("Running in legacy mode (v2.0.0 compatibility)")
        self.log("Note: Using new v2.5.0 SceneData architecture internally")

        # Extract output directory from JSX file path
        jsx_path = Path(jsx_file)
        output_dir = jsx_path.parent

        # Use the parent directory as output, but name it with the comp name
        result = self.convert_multi_format(
            input_file=abc_file,
            output_dir=output_dir,
            shot_name=comp_name,
            fps=fps,
            frame_count=frame_count,
            export_ae=True,
            export_usd=False,
            export_maya_ma=False,
            export_fbx=False
        )

        # If successful, move/rename the JSX file to match the expected path
        if result.get('success') and 'ae' in result:
            ae_result = result['ae']
            if ae_result.get('success'):
                # The AE exporter creates a folder like "comp_name_ae"
                # We need to move the JSX file to the specified jsx_file path
                generated_jsx = Path(ae_result['jsx_file'])
                if generated_jsx != jsx_path:
                    # Move JSX and OBJ files to the specified location
                    import shutil
                    jsx_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(generated_jsx), str(jsx_path))

                    # Move OBJ files
                    generated_dir = generated_jsx.parent
                    for obj_file in generated_dir.glob('*.obj'):
                        shutil.move(str(obj_file), str(jsx_path.parent / obj_file.name))

                    # Remove the generated folder if empty
                    try:
                        generated_dir.rmdir()
                    except:
                        pass

                self.log(f"\n✓ Conversion complete (legacy mode)")
                return True

        return False

    def detect_frame_count(self, input_file, fps=24):
        """Detect frame count from scene file

        This is a convenience method that wraps the reader's detect_frame_count()

        Args:
            input_file: Path to scene file (.abc, .usd, .usda, .usdc)
            fps: Frames per second

        Returns:
            int: Number of frames
        """
        reader = create_reader(input_file)
        return reader.detect_frame_count(fps)
