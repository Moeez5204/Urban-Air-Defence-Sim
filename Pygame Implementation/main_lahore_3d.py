"""
Lahore 3D Urban Defense Visualization
Main file with Good Drone Defense System Integration
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lahore_model_builder import LahoreModelBuilder
from lahore_3d_renderer import Lahore3DRenderer


def find_json_files():
    """Find JSON files in project directory"""
    project_root = Path(__file__).parent.parent  # Go up from PyGame folder

    json_files = {
        'building_data': None,
        'strategic_features': None,
        'threat_data': None
    }

    # Check in current directory first
    current_dir = Path(__file__).parent

    # Look for building data
    possible_building_files = [
        'building_data_3.1.1.json',
        'JSON files/building_data_3.1.1.json',
        '../building_data_3.1.1.json',
        '../JSON files/building_data_3.1.1.json'
    ]

    for file_path in possible_building_files:
        full_path = current_dir / file_path
        if full_path.exists():
            json_files['building_data'] = str(full_path)
            break

    # Look for strategic features
    possible_strategic_files = [
        'strategic_features_3.1.4.json',
        'JSON files/strategic_features_3.1.4.json',
        '../strategic_features_3.1.4.json',
        '../JSON files/strategic_features_3.1.4.json'
    ]

    for file_path in possible_strategic_files:
        full_path = current_dir / file_path
        if full_path.exists():
            json_files['strategic_features'] = str(full_path)
            break

    # Look for threat data
    possible_threat_files = [
        'lahore_3d_data.json',
        'JSON files/lahore_3d_data.json',
        '../lahore_3d_data.json',
        '../JSON files/lahore_3d_data.json',
        'urban_tracking_data.json',
        'JSON files/urban_tracking_data.json'
    ]

    for file_path in possible_threat_files:
        full_path = current_dir / file_path
        if full_path.exists():
            json_files['threat_data'] = str(full_path)
            break

    return json_files


def test_drone_system():
    """Test the good drone system independently"""
    print("\n" + "=" * 60)
    print("TESTING DRONE DEFENSE SYSTEM")
    print("=" * 60)

    try:
        # Try to import and test
        print("Attempting to import good drone controller...")
        from good_drone_controller import test_good_drone_system

        print("Running drone system test...")
        controller = test_good_drone_system()

        if controller:
            print("\n✓ Drone system test passed!")
            print(f"✓ {len(controller.drones)} drones initialized")
            print(f"✓ {controller.engagements} engagements simulated")
            print(f"✓ {controller.threats_neutralized} threats neutralized")
            print("\nYou can now run the full visualization with good drones!")
            return True
        else:
            print("\n⚠ Drone system test failed")
            return False

    except ImportError as e:
        print(f"\n❌ Could not import good_drone_controller: {e}")
        print("\nMake sure good_drone_controller.py is in the same directory")
        print("and all dependencies are installed:")
        print("  pip install numpy dataclasses-json")
        return False
    except Exception as e:
        print(f"\n❌ Error testing drone system: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to build and render Lahore 3D model"""
    print("=" * 60)
    print("LAHORE 3D URBAN DEFENSE VISUALIZATION")
    print("WITH GOOD DRONE DEFENSE SYSTEM")
    print("=" * 60)

    # Find JSON files
    print("\nLooking for JSON data files...")
    json_files = find_json_files()

    files_found = sum(1 for v in json_files.values() if v is not None)

    if files_found == 0:
        print("\nNo JSON files found in project directory.")
        print("The visualization will use synthetic data.")
        print("\nTo use real data, please ensure these files exist:")
        print("  - building_data_3.1.1.json")
        print("  - strategic_features_3.1.4.json")
        print("  - lahore_3d_data.json")
        print("\nFiles should be in the project root or JSON files/ directory.")
    else:
        print(f"\nFound {files_found} JSON file(s).")

    # Step 1: Build 3D model
    print("\n" + "=" * 50)
    print("[1/3] Building 3D Model...")
    print("=" * 50)

    builder = LahoreModelBuilder()
    model = builder.build_complete_model()

    # Step 2: Initialize 3D renderer
    print("\n" + "=" * 50)
    print("[2/3] Initializing 3D Visualization...")
    print("=" * 50)

    try:
        renderer = Lahore3DRenderer(screen_width=1400, screen_height=900)

        # Step 3: Initialize good drone defense system
        print("\n" + "=" * 50)
        print("[3/3] Initializing Defense Systems...")
        print("=" * 50)

        # Initialize good drones
        drone_init_success = renderer.initialize_good_drones(num_drones=4)

        if drone_init_success:
            print("\n" + "=" * 50)
            print("✓ DEFENSE SYSTEMS READY")
            print("=" * 50)
            print("Key Controls:")
            print("  D: Toggle good drones on/off")
            print("  M: Show drone status report (console)")
            print("  I: Reinitialize drones")
            print("\nVisual Guide:")
            print("  Green spheres: Your defense drones")
            print("  Red/Orange spheres: Enemy threats")
            print("  Green pyramids: Defended assets")
            print("  Colored troughs: Urban canyons")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("⚠ Good drones not available")
            print("=" * 50)
            print("To enable good drone defense system:")
            print("1. Make sure good_drone_controller.py is in same directory")
            print("2. Install dependencies:")
            print("   pip install numpy dataclasses-json")
            print("3. Run test first:")
            print("   Uncomment test_drone_system() in this file")
            print("=" * 50)

        # Display enhanced control guide
        print("\n" + "=" * 50)
        print("ENHANCED CONTROL SUMMARY:")
        print("=" * 50)
        print("Camera Controls:")
        print("  Mouse Drag: Rotate camera")
        print("  Scroll: Zoom in/out")
        print("  Arrow Keys: Fine camera control")
        print("  +/-: Zoom fine adjustment")
        print("  R: Reset camera")
        print("\nDisplay Toggles:")
        print("  B: Toggle Buildings")
        print("  C: Toggle Canyons")
        print("  T: Toggle Threats (Enemies)")
        print("  A: Toggle Assets")
        print("  D: Toggle Good Drones")
        print("  W: Wireframe Mode")
        print("  G: Toggle Ground")
        print("  X: Toggle Axes")
        print("\nDrone Controls:")
        print("  I: Initialize/Reinitialize Drones")
        print("  M: Drone Status Report")
        print("\nSystem:")
        print("  ESC: Exit")
        print("=" * 50)

        # Run the visualization
        print("\nStarting visualization...")
        renderer.run(model)

    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

        # Provide helpful installation instructions
        print("\n" + "=" * 50)
        print("TROUBLESHOOTING:")
        print("=" * 50)
        print("1. Install required packages:")
        print("   pip install PyOpenGL PyOpenGL-accelerate")
        print("\n2. For good drone system:")
        print("   pip install numpy dataclasses-json")
        print("\n3. Ensure JSON files are in correct location:")
        print("   - In project root or JSON_files/ folder")
        print("   - Required: building_data_3.1.1.json")
        print("   - Optional: strategic_features_3.1.4.json")
        print("   - Optional: lahore_3d_data.json")
        print("\n4. Run test first (uncomment in code):")
        print("   test_drone_system()")
        print("=" * 50)

    print("\n" + "=" * 60)
    print("Visualization session ended.")
    print("=" * 60)


if __name__ == "__main__":
    # Uncomment to test drone system first (recommended)
    # test_drone_system()

    # Run main visualization
    main()