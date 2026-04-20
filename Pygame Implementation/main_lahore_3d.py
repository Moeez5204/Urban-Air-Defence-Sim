import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lahore_model_builder import LahoreModelBuilder
from lahore_3d_renderer import Lahore3DRenderer
from conclusion import PostMissionAnalyzer  # Changed import


def find_json_files():
    project_root = Path(__file__).parent.parent

    json_files = {
        'building_data': None,
        'strategic_features': None,
        'threat_data': None
    }

    current_dir = Path(__file__).parent
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


    try:
        from good_drone_controller import test_integrated_system

        print("Running drone system test")
        controller = test_integrated_system()

        if controller:
            print("Drone system test passed!")
            return True
        else:
            print("Drone system test failed")
            return False

    except ImportError as e:
        return False
    except Exception as e:
        print(f"\n Error testing drone system: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to build and render Lahore 3D model"""
    print("=" * 60)
    print("LAHORE 3D URBAN DEFENSE VISUALIZATION")
    print("WITH POST-MISSION MOVEMENT ANALYSIS")
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

    # Build 3D model
    print("\n" + "=" * 50)
    print("[1/3] Building 3D Model...")
    print("=" * 50)

    builder = LahoreModelBuilder()
    model = builder.build_complete_model()

    # Initialize 3D renderer
    print("\n" + "=" * 50)
    print("[2/3] Initializing 3D Visualization...")
    print("=" * 50)

    renderer = None
    post_mission_analyzer = None  # New variable for post-mission analysis

    try:
        renderer = Lahore3DRenderer(screen_width=1400, screen_height=900)

        # Initialize good drone defense system
        print("\n" + "=" * 50)
        print("[3/3] Initializing Defense Systems...")
        print("=" * 50)

        # Initialize good drones
        drone_init_success = renderer.initialize_good_drones(num_drones=4)

        if drone_init_success:
            print("✓ DEFENSE SYSTEMS READY")
            print("Key Controls:")
            print("  D: Toggle good drones on/off")
            print("  M: Show drone status report (console)")
            print("  I: Reinitialize drones")
            print("Guide:")
            print("  Green spheres: Your defense drones")
            print("  Red/Orange spheres: Enemy threats")
            print("  Green pyramids: Defended assets")
            print("  Colored troughs: Urban canyons")
            print("=" * 50)


        print("POST-MISSION ANALYSIS SYSTEM")

        if renderer and renderer.bad_drone_controller:
            post_mission_analyzer = PostMissionAnalyzer()
            post_mission_analyzer.enable_recording(renderer.bad_drone_controller)

            print("Post-mission recording enabled")
            print("All enemy movements will be recorded")
            print("After simulation, analyze any enemy with:")
            print("analyzer.analyze_enemy('Enemy_02')")
        else:
            print("Cannot enable post-mission analysis:")
            print("Bad drone controller not available")

        print("Camera Controls:")
        print("Mouse Drag: Rotate camera")
        print("Scroll: Zoom in/out")
        print("Arrow Keys: Fine camera control")
        print("+/-: Zoom fine adjustment")
        print("R: Reset camera")
        print("Display Toggles:")
        print("B: Toggle Buildings")
        print("C: Toggle Canyons")
        print("T: Toggle Threats (Enemies)")
        print("A: Toggle Assets")
        print("D: Toggle Good Drones")
        print("W: Wireframe Mode")
        print("G: Toggle Ground")
        print("X: Toggle Axes")
        print("Drone Controls:")
        print("I: Initialize/Reinitialize Drones")
        print("M: Drone Status Report")
        print("System:")
        print("ESC: Exit")

        # Run the visualization
        print("\nStarting visualization")
        renderer.run(model)

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()



    finally:
        print("\n" + "=" * 60)
        print("POST-MISSION ANALYSIS")
        print("=" * 60)

        if post_mission_analyzer:
            try:
                if hasattr(post_mission_analyzer, 'enemy_history') and post_mission_analyzer.enemy_history:
                    print("Enemies recorded during mission:")
                    for enemy_id in post_mission_analyzer.enemy_history.keys():
                        print(f"  - {enemy_id}")

                    # Simple menu
                    while True:
                        print("POST-MISSION ANALYSIS MENU")
                        print("-" * 40)
                        print("1. Analyze specific enemy movement")
                        print("2. Show simple report for all enemies")
                        print("3. Exit analysis")

                        choice = input("\nSelect option (1-3): ").strip()

                        if choice == '1':
                            enemy_id = input("Enter enemy ID to analyze (e.g., Enemy_01): ").strip()
                            if enemy_id in post_mission_analyzer.enemy_history:
                                print(f"Analyzing {enemy_id}...")
                                post_mission_analyzer.analyze_enemy(enemy_id)
                                print(f"Analysis complete for {enemy_id}")
                                print("Returning to menu...")
                            else:
                                print(f" No data found for {enemy_id}")
                                print(f"Available enemies: {list(post_mission_analyzer.enemy_history.keys())}")

                        elif choice == '2':
                            print("Simple report ")
                            for enemy_id in post_mission_analyzer.enemy_history.keys():
                                post_mission_analyzer.simple_report(enemy_id)
                                print("-" * 40)

                        elif choice == '3':
                            print("\nExiting post-mission analysis")
                            break

                        else:
                            print("\nInvalid choice. Please enter 1, 2, or 3.")
                else:
                    print("No enemy data recorded during mission")

            except Exception as e:
                print(f"\nError during post-mission analysis: {e}")
                print("You can still analyze enemies manually:")
                print("analyzer.analyze_enemy('Enemy_02')")

        else:
            print(" Post-mission analysis was not enabled")
            print("To enable next time, ensure bad_drone_controller is available")

        print("MISSION COMPLETE - SYSTEM SUMMARY")

        try:
            # Basic summary
            if model:
                print(f"\nModel Statistics:")
                print(f"  Buildings: {len(model.buildings) if hasattr(model, 'buildings') else 0}")
                print(f"  Canyons: {len(model.canyons) if hasattr(model, 'canyons') else 0}")
                print(f"  Threats: {len(model.threats) if hasattr(model, 'threats') else 0}")
                print(f"  Assets: {len(model.assets) if hasattr(model, 'assets') else 0}")

            if renderer and renderer.good_drone_controller:
                print(f"\nDrone System Performance:")
                print(f"  Defense Drones: {len(renderer.good_drone_controller.drones)}")

            if renderer and renderer.bad_drone_controller:
                print(f"  Enemy Drones: {len(renderer.bad_drone_controller.enemies)}")

            if post_mission_analyzer and hasattr(post_mission_analyzer, 'enemy_history'):
                recorded_enemies = len(post_mission_analyzer.enemy_history)
                print(f"\nPost-Mission Analysis:")
                print(f"  Enemies recorded: {recorded_enemies}")
                if recorded_enemies > 0:
                    print("  To analyze: analyzer.analyze_enemy('Enemy_ID')")

        except Exception as e:
            print(f"Could not generate final summary: {e}")



if __name__ == "__main__":
    main()