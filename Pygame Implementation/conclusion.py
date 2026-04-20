# conclusion.py - Post-Mission 3D Analysis with Live Animation
import json
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class PostMissionAnalyzer:

    def __init__(self):
        self.enemy_history = {}
        self.prediction_history = {}
        self.timestamps = []

    def enable_recording(self, bad_drone_controller):
        print("Post mission analysis")

        original_update = bad_drone_controller.update_enemies

        def recording_update(delta_time=0.016):
            result = original_update(delta_time)

            current_time = time.time()
            if not self.timestamps:
                self.start_time = current_time
            rel_time = current_time - self.start_time
            self.timestamps.append(rel_time)

            for enemy in bad_drone_controller.enemies:
                if enemy.id not in self.enemy_history:
                    self.enemy_history[enemy.id] = []

                self.enemy_history[enemy.id].append({
                    'time': rel_time,
                    'position': enemy.position
                })

                predicted_pos = (
                    enemy.position[0] + enemy.velocity[0] * 2.0,
                    enemy.position[1] + enemy.velocity[1] * 2.0,
                    enemy.position[2] + enemy.velocity[2] * 2.0
                )

                if enemy.id not in self.prediction_history:
                    self.prediction_history[enemy.id] = []

                self.prediction_history[enemy.id].append({
                    'time': rel_time,
                    'position': predicted_pos
                })

            return result

        bad_drone_controller.update_enemies = recording_update
        print(f"Enemies tracked: {[e.id for e in bad_drone_controller.enemies]}")

        return self

    def analyze_enemy(self, enemy_id, default_to_animation=True):
        print(f"ANALYZING ENEMY: {enemy_id}")

        if enemy_id not in self.enemy_history:
            print(f"Error: No data for {enemy_id}")
            print(f"Available enemies: {list(self.enemy_history.keys())}")
            return

        actual_data = self.enemy_history[enemy_id]
        pred_data = self.prediction_history[enemy_id]
        self.show_brief_analysis(enemy_id, actual_data, pred_data)

        if default_to_animation:
            print("LAUNCHING DEFAULT: Live 3D Animation ")
            self.create_live_animation(enemy_id, actual_data, pred_data, speed_factor=2.0)
            self.offer_post_animation_options(enemy_id, actual_data, pred_data)

    def create_live_animation(self, enemy_id, actual_data, pred_data, speed_factor=2.0):


        print(f"LIVE 3D ANIMATION: {enemy_id}")
        print(f"Speed: {speed_factor}x, Frames: {len(actual_data)}")

        actual_x = np.array([d['position'][0] for d in actual_data])
        actual_y = np.array([d['position'][1] for d in actual_data])
        actual_z = np.array([d['position'][2] for d in actual_data])

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10, color='white')
        ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10, color='white')
        ax.set_zlabel('Altitude (m)', fontsize=12, labelpad=10, color='white')
        ax.set_title(f'Enemy {enemy_id}: Live Path Animation ({speed_factor}x Speed)',
                     fontsize=14, fontweight='bold', pad=20, color='white')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')

        actual_line, = ax.plot([], [], [], 'cyan', linewidth=4, label='Actual Path', alpha=0.9)
        actual_point, = ax.plot([], [], [], 'cyan', marker='o', markersize=10,
                                label='Current Position', alpha=1.0, markeredgecolor='white')

        pred_line, = ax.plot([], [], [], 'magenta', linewidth=3, label='Predicted Path',
                             alpha=0.7, linestyle='--')
        pred_point, = ax.plot([], [], [], 'magenta', marker='s', markersize=8,
                              label='Predicted Position', alpha=1.0, markeredgecolor='white')

        legend = ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        for text in legend.get_texts():
            text.set_color('white')
        ax.grid(True, alpha=0.2, color='white')

        all_x = list(actual_x)
        all_y = list(actual_y)
        all_z = list(actual_z)

        pred_x = [d['position'][0] for d in pred_data]
        pred_y = [d['position'][1] for d in pred_data]
        pred_z = [d['position'][2] for d in pred_data]
        all_x += pred_x
        all_y += pred_y
        all_z += pred_z

        x_center = (min(all_x) + max(all_x)) / 2
        y_center = (min(all_y) + max(all_y)) / 2
        z_center = (min(all_z) + max(all_z)) / 2
        max_range = max(max(all_x) - min(all_x),
                        max(all_y) - min(all_y),
                        max(all_z) - min(all_z)) * 0.6

        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)

        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                              color='white', fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='navy', alpha=0.9))

        progress_text = ax.text2D(0.85, 0.95, '', transform=ax.transAxes, fontsize=11,
                                  color='white', fontweight='bold',
                                  bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.9))

        stats_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes, fontsize=10,
                               color='white',
                               bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.9))

        error_lines = []

        def update(frame):
            actual_frame = min(int(frame * speed_factor), len(actual_x) - 1)

            actual_line.set_data(actual_x[:actual_frame + 1], actual_y[:actual_frame + 1])
            actual_line.set_3d_properties(actual_z[:actual_frame + 1])

            actual_point.set_data([actual_x[actual_frame]], [actual_y[actual_frame]])
            actual_point.set_3d_properties([actual_z[actual_frame]])

            for line in error_lines:
                line.remove()
            error_lines.clear()

            pred_frame = min(actual_frame, len(pred_data) - 1)
            pred_line.set_data(pred_x[:pred_frame + 1], pred_y[:pred_frame + 1])
            pred_line.set_3d_properties(pred_z[:pred_frame + 1])

            pred_point.set_data([pred_x[pred_frame]], [pred_y[pred_frame]])
            pred_point.set_3d_properties([pred_z[pred_frame]])

            if actual_frame % 10 == 0:
                for i in range(0, min(actual_frame, pred_frame), 20):
                    line, = ax.plot([actual_x[i], pred_x[i]],
                                    [actual_y[i], pred_y[i]],
                                    [actual_z[i], pred_z[i]],
                                    'gray', linewidth=0.8, alpha=0.5)
                    error_lines.append(line)

            if actual_frame > 0:
                distance = 0
                for i in range(1, actual_frame + 1):
                    dx = actual_x[i] - actual_x[i - 1]
                    dy = actual_y[i] - actual_y[i - 1]
                    dz = actual_z[i] - actual_z[i - 1]
                    distance += np.sqrt(dx * dx + dy * dy + dz * dz)

                current_speed = 0
                if actual_frame > 1:
                    dx = actual_x[actual_frame] - actual_x[actual_frame - 1]
                    dy = actual_y[actual_frame] - actual_y[actual_frame - 1]
                    dz = actual_z[actual_frame] - actual_z[actual_frame - 1]
                    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                    time_diff = actual_data[actual_frame]['time'] - actual_data[actual_frame - 1]['time']
                    if time_diff > 0:
                        current_speed = dist / time_diff

                avg_speed = distance / actual_data[actual_frame]['time'] if actual_data[actual_frame]['time'] > 0 else 0

                if pred_data and actual_frame < len(pred_data):
                    error = np.sqrt(
                        (pred_x[actual_frame] - actual_x[actual_frame]) ** 2 +
                        (pred_y[actual_frame] - actual_y[actual_frame]) ** 2 +
                        (pred_z[actual_frame] - actual_z[actual_frame]) ** 2
                    )
                    error_text = f"Pred Error: {error:.1f}m"
                else:
                    error_text = ""

                stats_text.set_text(
                    f"Distance: {distance:.1f}m\n"
                    f"Avg Speed: {avg_speed:.1f}m/s\n"
                    f"Current Speed: {current_speed:.1f}m/s\n"
                    f"{error_text}"
                )

            mission_time = actual_data[actual_frame]['time']
            progress_pct = (actual_frame / len(actual_x)) * 100

            time_text.set_text(f'Mission Time: {mission_time:.1f}s\n'
                               f'Frame: {actual_frame}/{len(actual_x) - 1}')
            progress_text.set_text(f'Progress: {progress_pct:.1f}%\n'
                                   f'Speed: {speed_factor}x')

            artists = [actual_line, actual_point, time_text, progress_text, stats_text, pred_line, pred_point]
            artists.extend(error_lines)

            return artists

        total_animation_frames = int(len(actual_x) / speed_factor)

        print("Generating animation ")

        anim = animation.FuncAnimation(
            fig, update,
            frames=total_animation_frames,
            interval=40,
            blit=True,
            repeat=False,
            cache_frame_data=False
        )

        try:
            save_video = input("Save animation as video file? (y/n): ").strip().lower()
            if save_video == 'y':
                video_filename = f"{enemy_id}_live_animation.mp4"
                print(f"Saving video to {video_filename}")
                anim.save(video_filename)
                print(f"Video saved as {video_filename}")
        except Exception as e:
            print(f"Could not save video: {e}")


        plt.tight_layout()
        plt.show(block=True)
        print("Animation complete!")

        while True:
            key = input("Press 'Q' to close ").upper()
            if key == 'Q':
                plt.close(fig)
                break
            else:
                print("Please press 'Q' to close")

        return True

    def offer_post_animation_options(self, enemy_id, actual_data, pred_data):
        print("POST-ANIMATION OPTIONS:")
        print("1. View brief summary")
        print("2. Save data to files")
        print("3. Create static 3D plot")
        print("4. Analyze another enemy")
        print("5. Exit analysis")

        choice = input("Select option (1-5): ").strip()

        if choice == '1':
            self.show_brief_analysis(enemy_id, actual_data, pred_data)
            self.offer_post_animation_options(enemy_id, actual_data, pred_data)
        elif choice == '2':
            self.save_all_data(enemy_id, actual_data, pred_data)
            self.offer_post_animation_options(enemy_id, actual_data, pred_data)
        elif choice == '3':
            self.create_static_plot(enemy_id, actual_data, pred_data)
            self.offer_post_animation_options(enemy_id, actual_data, pred_data)
        elif choice == '4':
            return
        elif choice == '5':
            print("Exiting analysis")
            return

    def show_brief_analysis(self, enemy_id, actual_data, pred_data):
        print("Analysis:")

        total_distance = 0
        for i in range(1, len(actual_data)):
            p1 = actual_data[i - 1]['position']
            p2 = actual_data[i]['position']
            total_distance += math.sqrt(
                (p2[0] - p1[0]) ** 2 +
                (p2[1] - p1[1]) ** 2 +
                (p2[2] - p1[2]) ** 2
            )

        duration = actual_data[-1]['time'] - actual_data[0]['time']
        avg_speed = total_distance / duration if duration > 0 else 0

        print(f"Mission Duration: {duration:.1f}s")
        print(f"Total Distance: {total_distance:.1f}m")
        print(f"Average Speed: {avg_speed:.1f}m/s")
        print(f"Predictions Available: {len(pred_data)}")

    def create_static_plot(self, enemy_id, actual_data, pred_data):
        print(f"Creating static 3D plot for {enemy_id}")

        actual_x = [d['position'][0] for d in actual_data]
        actual_y = [d['position'][1] for d in actual_data]
        actual_z = [d['position'][2] for d in actual_data]

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(actual_x, actual_y, actual_z, 'b-', linewidth=3, label='Actual Path', alpha=0.9)

        ax.scatter(actual_x[0], actual_y[0], actual_z[0], c='green', s=200, marker='o', label='Start')
        ax.scatter(actual_x[-1], actual_y[-1], actual_z[-1], c='red', s=200, marker='o', label='End')

        pred_x = [d['position'][0] for d in pred_data]
        pred_y = [d['position'][1] for d in pred_data]
        pred_z = [d['position'][2] for d in pred_data]

        ax.plot(pred_x, pred_y, pred_z, 'r--', linewidth=2, label='Predicted Path', alpha=0.7)
        ax.scatter(pred_x[-1], pred_y[-1], pred_z[-1], c='orange', s=150, marker='s', label='Predicted End')

        ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Altitude (m)', fontsize=12, labelpad=10)
        ax.set_title(f'Enemy {enemy_id}: Actual vs Predicted Movement', fontsize=14, fontweight='bold', pad=20)

        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)

        filename = f"{enemy_id}_static_plot.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Static plot saved as: {filename}")



    def save_all_data(self, enemy_id, actual_data, pred_data):
        dir_name = f"{enemy_id}_analysis"
        os.makedirs(dir_name, exist_ok=True)

        with open(f"{dir_name}/{enemy_id}_positions.txt", 'w') as f:
            f.write(f"ENEMY: {enemy_id} - POSITION DATA\n")
            f.write("Frame, Time(s), X(m), Y(m), Z(m), Pred_X, Pred_Y, Pred_Z, Error(m)\n")

            for i in range(len(actual_data)):
                actual_pos = actual_data[i]['position']
                pred_pos = pred_data[i]['position']
                error = math.sqrt(
                    (pred_pos[0] - actual_pos[0]) ** 2 +
                    (pred_pos[1] - actual_pos[1]) ** 2 +
                    (pred_pos[2] - actual_pos[2]) ** 2
                )
                f.write(f"{i}, {actual_data[i]['time']:.2f}, "
                        f"{actual_pos[0]:.1f}, {actual_pos[1]:.1f}, {actual_pos[2]:.1f}, "
                        f"{pred_pos[0]:.1f}, {pred_pos[1]:.1f}, {pred_pos[2]:.1f}, "
                        f"{error:.1f}\n")

        print(f"Position data saved to: {dir_name}/{enemy_id}_positions.txt")

        with open(f"{dir_name}/{enemy_id}_summary.txt", 'w') as f:
            f.write(f"ENEMY: {enemy_id} - ANALYSIS SUMMARY\n")

            total_distance = 0
            for i in range(1, len(actual_data)):
                p1 = actual_data[i - 1]['position']
                p2 = actual_data[i]['position']
                total_distance += math.sqrt(
                    (p2[0] - p1[0]) ** 2 +
                    (p2[1] - p1[1]) ** 2 +
                    (p2[2] - p1[2]) ** 2
                )

            duration = actual_data[-1]['time'] - actual_data[0]['time']
            avg_speed = total_distance / duration if duration > 0 else 0

            f.write("MOVEMENT STATISTICS:\n")
            f.write(f"Total distance: {total_distance:.1f}m\n")
            f.write(f"Mission duration: {duration:.1f}s\n")
            f.write(f"Average speed: {avg_speed:.1f}m/s\n")
            f.write(f"Data points: {len(actual_data)}\n")

        print(f"Summary saved to: {dir_name}/{enemy_id}_summary.txt")

        data = {
            'enemy_id': enemy_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'actual_positions': [(d['time'], d['position']) for d in actual_data],
            'predicted_positions': [(d['time'], d['position']) for d in pred_data],
        }

        with open(f"{dir_name}/{enemy_id}_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"JSON data saved to: {dir_name}/{enemy_id}_data.json")



class SystemDataAnalyzer(PostMissionAnalyzer):
    pass